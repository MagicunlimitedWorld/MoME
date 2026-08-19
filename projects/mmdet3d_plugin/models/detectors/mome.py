# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------

import mmcv
import copy
import hashlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from mmcv.runner import force_fp32, auto_fp16
from mmdet.core import multi_apply
from mmdet.models import DETECTORS
from mmdet.models.builder import build_backbone
from mmdet3d.core import (Box3DMode, Coord3DMode, bbox3d2result,
                          merge_aug_bboxes_3d, show_result)
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector

from projects.mmdet3d_plugin.models.utils.grid_mask import GridMask
from projects.mmdet3d_plugin.models.utils.qta_router import (
    normalize_locked_route_state,
    rank_greedy_candidates,
    robust_candidate_quality,
    select_conservative_routes,
    select_full_query_oracle_routes,
    select_robust_oracle_actions,
)
from projects.mmdet3d_plugin import SPConvVoxelization


@DETECTORS.register_module()
class MoME(MVXTwoStageDetector):

    def __init__(self,
                 use_grid_mask=False,
                 **kwargs):
        self.local_evidence_rule = kwargs.pop('local_evidence_rule', None)
        pts_voxel_cfg = kwargs.get('pts_voxel_layer', None)
        kwargs['pts_voxel_layer'] = None
        super(MoME, self).__init__(**kwargs)
        
        self.use_grid_mask = use_grid_mask
        self.grid_mask = GridMask(True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        if pts_voxel_cfg:
            self.pts_voxel_layer = SPConvVoxelization(**pts_voxel_cfg)
        self._validate_local_evidence_rule()

    def _validate_local_evidence_rule(self):
        if self.local_evidence_rule is None:
            return
        allowed = {
            'enabled', 'variant', 'thresholds', 'max_overrides_per_frame'
        }
        unexpected = set(self.local_evidence_rule) - allowed
        if unexpected:
            raise ValueError(
                f'Unsupported local_evidence_rule keys: {sorted(unexpected)}'
            )
        if self.local_evidence_rule.get('variant') not in (
            'point_count', 'valid_views'
        ):
            raise ValueError(
                'local_evidence_rule variant must be point_count or valid_views'
            )
        if len(self.local_evidence_rule.get('thresholds', ())) != 3:
            raise ValueError('local_evidence_rule requires three thresholds')
        if int(self.local_evidence_rule.get('max_overrides_per_frame', 0)) < 0:
            raise ValueError('local_evidence_rule cap cannot be negative')

    def init_weights(self):
        """Initialize model weights."""
        super(MoME, self).init_weights()

    @auto_fp16(apply_to=('img'), out_fp32=True) 
    def extract_img_feat(self, img, img_metas):
        """Extract features of images."""
        if self.with_img_backbone and img is not None:
            input_shape = img.shape[-2:]
            # update real input shape of each single img
            for img_meta in img_metas:
                img_meta.update(input_shape=input_shape)

            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_(0)
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.view(B * N, C, H, W)
            if self.use_grid_mask:
                img = self.grid_mask(img)
            img_feats = self.img_backbone(img.float())
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)
        return img_feats

    @force_fp32(apply_to=('pts', 'img_feats'))
    def extract_pts_feat(self, pts, img_feats, img_metas):
        """Extract features of points."""
        if not self.with_pts_bbox:
            return None
        if pts is None:
            return None
        voxels, num_points, coors = self.voxelize(pts)
        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors,
                                                )
        batch_size = coors[-1, 0] + 1
        x = self.pts_middle_encoder(voxel_features, coors, batch_size)
        x = self.pts_backbone(x)
        if self.with_pts_neck:
            x = self.pts_neck(x)
        return x

    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points):
        """Apply dynamic voxelization to points.

        Args:
            points (list[torch.Tensor]): Points of each sample.

        Returns:
            tuple[torch.Tensor]: Concatenated points, number of points
                per voxel, and coordinates.
        """
        voxels, coors, num_points = [], [], []
        for res in points:
            res_voxels, res_coors, res_num_points = self.pts_voxel_layer(res)
            voxels.append(res_voxels)
            coors.append(res_coors)
            num_points.append(res_num_points)
        voxels = torch.cat(voxels, dim=0)
        num_points = torch.cat(num_points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        return voxels, num_points, coors_batch

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      return_query_losses=False,
                      return_route_bundle=False,
                      route_override=None,
                      fixed_targets=None,
                      return_rematched_losses=True):
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.

        Returns:
            dict: Losses of different branches.
        """

        if return_route_bundle:
            return self.forward_qta_route_bundle(
                points,
                img_metas,
                img,
                gt_bboxes_3d,
                gt_labels_3d,
            )
        if return_query_losses:
            return self.forward_qta_probe(
                points,
                img_metas,
                img,
                gt_bboxes_3d,
                gt_labels_3d,
                route_override=route_override,
                fixed_targets=fixed_targets,
                return_query_losses=True,
                return_rematched_losses=return_rematched_losses,
            )

        img_feats, pts_feats = self.extract_feat(
            points, img=img, img_metas=img_metas)
        losses = dict()
        if pts_feats or img_feats:
            losses_pts = self.forward_pts_train(pts_feats, img_feats, gt_bboxes_3d,
                                                gt_labels_3d, img_metas,
                                                gt_bboxes_ignore)
            losses.update(losses_pts)
        return losses

    @force_fp32(apply_to=('pts_feats', 'img_feats'))
    def forward_pts_train(self,
                          pts_feats,
                          img_feats,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          img_metas,
                          gt_bboxes_ignore=None):
        """Forward function for point cloud branch.

        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.

        Returns:
            dict: Losses of each branch.
        """
        if pts_feats is None:
            pts_feats = [None]
        if img_feats is None:
            img_feats = [None]
        outs = self.pts_bbox_head(pts_feats, img_feats, img_metas)
        loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
        losses = self.pts_bbox_head.loss(*loss_inputs)
        return losses

    def forward_test(self,
                     points=None,
                     img_metas=None,
                     img=None, **kwargs):
        """
        Args:
            points (list[torch.Tensor]): the outer list indicates test-time
                augmentations and inner torch.Tensor should have a shape NxC,
                which contains all points in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch
            img (list[torch.Tensor], optional): the outer
                list indicates test-time augmentations and inner
                torch.Tensor should have a shape NxCxHxW, which contains
                all images in the batch. Defaults to None.
        """
        if points is None:
            points = [None]
        if img is None:
            img = [None]
        for var, name in [(points, 'points'), (img, 'img'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))

        return self.simple_test(points[0], img_metas[0], img[0], **kwargs)
    
    @force_fp32(apply_to=('x', 'x_img'))
    def simple_test_pts(self, x, x_img, img_metas, rescale=False,
                        route_override=None, qta_thresholds=None,
                        qta_max_overrides=None):
        """Test function of point cloud branch."""
        outs = self.pts_bbox_head(
            x,
            x_img,
            img_metas,
            route_override=route_override,
            qta_thresholds=qta_thresholds,
            qta_max_overrides=qta_max_overrides,
        )
        bbox_list = self.pts_bbox_head.get_bboxes(
            outs, img_metas, rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ] 
        return bbox_results

    def simple_test(self, points, img_metas, img=None, rescale=False,
                    route_override=None, qta_thresholds=None,
                    qta_max_overrides=None):
        self._annotate_qta_complete_failure(points, img, img_metas)
        if (
            self.local_evidence_rule is not None
            and self.local_evidence_rule.get('enabled', False)
        ):
            if route_override is not None:
                raise ValueError(
                    'explicit route_override cannot be combined with local evidence rule'
                )
            return self._simple_test_local_evidence_rule(
                points, img_metas, img=img, rescale=rescale
            )
        img_feats, pts_feats = self.extract_feat(
            points, img=img, img_metas=img_metas)
        if pts_feats is None:
            pts_feats = [None]
        if img_feats is None:
            img_feats = [None]
        
        bbox_list = [dict() for i in range(len(img_metas))]
        if (pts_feats or img_feats) and self.with_pts_bbox:
            bbox_pts = self.simple_test_pts(
                pts_feats,
                img_feats,
                img_metas,
                rescale=rescale,
                route_override=route_override,
                qta_thresholds=qta_thresholds,
                qta_max_overrides=qta_max_overrides,
            )
            for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
                result_dict['pts_bbox'] = pts_bbox
        if img_feats and self.with_img_bbox:
            bbox_img = self.simple_test_img(
                img_feats, img_metas, rescale=rescale)
            for result_dict, img_bbox in zip(bbox_list, bbox_img):
                result_dict['img_bbox'] = img_bbox
        return bbox_list

    def _simple_test_local_evidence_rule(self, points, img_metas, img=None,
                                         rescale=False):
        """Run the no-network point-count or valid-view routing control."""

        img_feats, pts_feats = self.extract_feat(
            points, img=img, img_metas=img_metas
        )
        if pts_feats is None:
            pts_feats = [None]
        if img_feats is None:
            img_feats = [None]
        base_outs = self.pts_bbox_head(
            pts_feats,
            img_feats,
            img_metas,
            return_router_state=True,
        )
        base_task = base_outs[0][0]
        point_count, valid_views = self.qta_local_evidence(
            base_task['reference_points'], img_metas, points
        )
        point_support = torch.log1p(point_count.clamp_min(0.0))
        view_support = valid_views.clamp_min(0.0)
        route_scores = point_support.new_zeros(
            (*point_support.shape, 3)
        )
        if self.local_evidence_rule['variant'] == 'point_count':
            route_scores[..., 0] = point_support
            route_scores[..., 1] = point_support
            route_scores[..., 2] = -point_support
        else:
            route_scores[..., 0] = view_support
            route_scores[..., 1] = -view_support
            route_scores[..., 2] = view_support
        route_available, hard_bypass = (
            self.pts_bbox_head.transformer._qta_runtime_constraints(
                img_metas, base_task['base_routes'].device
            )
        )
        final_routes, _ = select_conservative_routes(
            base_routes=base_task['base_routes'],
            advantage_scores=route_scores,
            thresholds=self.local_evidence_rule['thresholds'],
            max_overrides_per_frame=int(
                self.local_evidence_rule['max_overrides_per_frame']
            ),
            route_available=route_available,
            hard_bypass=hard_bypass,
        )
        final_outs = self.pts_bbox_head(
            pts_feats,
            img_feats,
            img_metas,
            route_override=final_routes,
        )
        bbox_list = self.pts_bbox_head.get_bboxes(
            final_outs, img_metas, rescale=rescale
        )
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return [dict(pts_bbox=result) for result in bbox_results]

    @staticmethod
    def _camera_sample_is_zero(img_sample, meta):
        """Detect an exactly zero raw image after the configured normalization."""

        if img_sample is None:
            return False
        if isinstance(img_sample, (list, tuple)):
            img_sample = torch.stack(list(img_sample), dim=0)
        if not torch.is_tensor(img_sample) or img_sample.ndim != 4:
            return False
        if torch.count_nonzero(img_sample).item() == 0:
            return True
        norm = meta.get('img_norm_cfg')
        shapes = meta.get('img_shape')
        if not isinstance(norm, dict) or shapes is None:
            return False
        if isinstance(shapes, tuple):
            shapes = [shapes] * img_sample.shape[0]
        mean = img_sample.new_tensor(norm['mean']).view(-1, 1, 1)
        std = img_sample.new_tensor(norm['std']).view(-1, 1, 1)
        expected = -mean / std
        tolerance = 2e-3 if img_sample.dtype == torch.float16 else 1e-5
        for view_index in range(img_sample.shape[0]):
            height, width = shapes[view_index][:2]
            visible = img_sample[view_index, :, :height, :width]
            if visible.shape[0] != expected.shape[0]:
                return False
            if not torch.allclose(
                visible,
                expected.expand_as(visible),
                rtol=0.0,
                atol=tolerance,
            ):
                return False
        return True

    def _annotate_qta_complete_failure(self, points, img, img_metas):
        """Set a transient hard bypass only for exact whole-modality zeros."""

        router = getattr(self.pts_bbox_head.transformer, 'qta_router', None)
        if router is None and self.local_evidence_rule is None:
            return
        if img is not None and torch.is_tensor(img) and img.ndim == 4:
            image_samples = [img]
        elif img is None:
            image_samples = [None] * len(img_metas)
        else:
            image_samples = list(img)
        for batch_index, meta in enumerate(img_metas):
            point_sample = points[batch_index] if points is not None else None
            if hasattr(point_sample, 'tensor'):
                point_sample = point_sample.tensor
            lidar_zero = (
                torch.is_tensor(point_sample)
                and torch.count_nonzero(point_sample).item() == 0
            )
            camera_zero = self._camera_sample_is_zero(
                image_samples[batch_index], meta
            )
            if not (lidar_zero or camera_zero):
                continue
            meta['qta_hard_bypass'] = True
            if lidar_zero and camera_zero:
                meta['qta_route_available'] = [False, False, False]
            elif lidar_zero:
                meta['qta_route_available'] = [False, False, True]
            else:
                meta['qta_route_available'] = [False, True, False]

    def forward_qta_probe(self, points, img_metas, img, gt_bboxes_3d,
                          gt_labels_3d, route_override=None,
                          fixed_targets=None, return_query_losses=True,
                          return_rematched_losses=True):
        """Execute an audit route and expose fixed/rematched query losses.

        This interface is intentionally separate from normal inference.  A
        Stage019 label generator first calls it with ``route_override=0`` and
        retains the returned Fused target bundle, then reuses that bundle for
        LiDAR and Camera executions.  A fresh target bundle provides the
        independent-Hungarian sensitivity result.
        """

        if self.training:
            raise RuntimeError("forward_qta_probe requires model.eval()")
        img_feats, pts_feats = self.extract_feat(
            points, img=img, img_metas=img_metas
        )
        if pts_feats is None:
            pts_feats = [None]
        if img_feats is None:
            img_feats = [None]
        preds_dicts = self.pts_bbox_head(
            pts_feats,
            img_feats,
            img_metas,
            route_override=route_override,
            return_router_state=True,
        )
        first_task = preds_dicts[0][0]
        result = {
            'preds_dicts': preds_dicts,
            'base_routes': first_task['base_routes'],
            'advantage_scores': first_task.get('advantage_scores'),
            'final_routes': first_task['final_routes'],
            'router_features': first_task['router_features'],
            'reference_points': first_task['reference_points'],
        }
        if return_query_losses:
            fixed = self.pts_bbox_head.return_query_losses(
                gt_bboxes_3d,
                gt_labels_3d,
                preds_dicts,
                fixed_targets=fixed_targets,
            )
            result['query_losses'] = {
                key: value for key, value in fixed.items() if key != 'target_bundle'
            }
            result['target_bundle'] = fixed['target_bundle']
            if return_rematched_losses and fixed_targets is not None:
                rematched = self.pts_bbox_head.return_query_losses(
                    gt_bboxes_3d,
                    gt_labels_3d,
                    preds_dicts,
                    fixed_targets=None,
                )
                result['rematched_query_losses'] = {
                    key: value
                    for key, value in rematched.items()
                    if key != 'target_bundle'
                }
        return result

    @staticmethod
    def _points_in_polygon(points, polygon):
        x = points[:, 0]
        y = points[:, 1]
        inside = np.zeros(points.shape[0], dtype=bool)
        previous = len(polygon) - 1
        for current in range(len(polygon)):
            x_current, y_current = polygon[current]
            x_previous, y_previous = polygon[previous]
            crosses = ((y_current > y) != (y_previous > y)) & (
                x
                < (x_previous - x_current)
                * (y - y_current)
                / (y_previous - y_current + 1e-12)
                + x_current
            )
            inside ^= crosses
            previous = current
        return inside

    def qta_affected_query_mask(self, reference_points, img_metas,
                                gt_bboxes_3d):
        """Map recorded procedural fault geometry to MoME query centers."""

        pc_range = reference_points.new_tensor(self.pts_bbox_head.pc_range)
        xyz = reference_points * (pc_range[3:] - pc_range[:3]) + pc_range[:3]
        affected = torch.zeros(
            reference_points.shape[:2],
            device=reference_points.device,
            dtype=torch.bool,
        )
        for batch_index, meta in enumerate(img_metas):
            corruption = meta.get('qta_corruption', {})
            condition = corruption.get('condition', 'clean')
            details = corruption.get('details', {})
            if condition in ('lidar_zero', 'camera_zero'):
                affected[batch_index] = True
            elif condition == 'lidar_sector_missing':
                center = np.deg2rad(details['sector_center_deg'])
                half_width = np.deg2rad(details['sector_width_deg']) / 2.0
                angles = torch.atan2(xyz[batch_index, :, 1], xyz[batch_index, :, 0])
                center_t = angles.new_tensor(center)
                delta = torch.atan2(
                    torch.sin(angles - center_t), torch.cos(angles - center_t)
                )
                affected[batch_index] = delta.abs() <= half_width
            elif condition == 'lidar_object_points_missing':
                selected = details.get('selected_box_indices', [])
                if selected:
                    boxes = gt_bboxes_3d[batch_index]
                    point_box = boxes.points_in_boxes_part(
                        xyz[batch_index], boxes_override=boxes.tensor[:, :7]
                    )
                    for box_index in selected:
                        affected[batch_index] |= point_box == int(box_index)
            elif condition == 'camera_local_occlusion':
                query_xyz = xyz[batch_index].detach().cpu().numpy()
                homogeneous = np.concatenate(
                    [query_xyz, np.ones((query_xyz.shape[0], 1), dtype=query_xyz.dtype)],
                    axis=1,
                )
                inside_any = np.zeros(query_xyz.shape[0], dtype=bool)
                polygons = details.get('polygons_xy', {})
                for view_index in details.get('selected_views', []):
                    projection = homogeneous @ np.asarray(
                        meta['lidar2img'][int(view_index)]
                    ).T
                    valid_depth = projection[:, 2] > 1e-5
                    pixels = projection[:, :2] / np.clip(
                        projection[:, 2:3], 1e-5, None
                    )
                    polygon = np.asarray(polygons[str(view_index)], dtype=np.float64)
                    inside_any |= valid_depth & self._points_in_polygon(pixels, polygon)
                affected[batch_index] = torch.from_numpy(inside_any).to(
                    device=reference_points.device
                )
        return affected

    def _require_qta_probe_state(self):
        router = getattr(self.pts_bbox_head.transformer, 'qta_router', None)
        if router is not None and router.enabled:
            raise RuntimeError(
                'QTA label probes require the candidate override gate to be disabled'
            )

    def qta_local_evidence(self, reference_points, img_metas, points):
        """Return simple per-query LiDAR-count and valid-view controls."""

        pc_range = reference_points.new_tensor(self.pts_bbox_head.pc_range)
        xyz = reference_points * (pc_range[3:] - pc_range[:3]) + pc_range[:3]
        batch_size, num_queries = reference_points.shape[:2]
        lidar_counts = reference_points.new_zeros((batch_size, num_queries))
        valid_views = reference_points.new_zeros((batch_size, num_queries))
        grid_size = 180
        for batch_index in range(batch_size):
            point_tensor = points[batch_index]
            if hasattr(point_tensor, 'tensor'):
                point_tensor = point_tensor.tensor
            point_x = torch.floor(
                (point_tensor[:, 0] - pc_range[0])
                / (pc_range[3] - pc_range[0])
                * grid_size
            ).long()
            point_y = torch.floor(
                (point_tensor[:, 1] - pc_range[1])
                / (pc_range[4] - pc_range[1])
                * grid_size
            ).long()
            valid_point = (
                (point_x >= 0)
                & (point_x < grid_size)
                & (point_y >= 0)
                & (point_y < grid_size)
            )
            flat = point_y[valid_point] * grid_size + point_x[valid_point]
            grid = torch.bincount(
                flat, minlength=grid_size * grid_size
            ).to(reference_points.dtype).reshape(1, 1, grid_size, grid_size)
            window = F.conv2d(
                grid,
                grid.new_ones((1, 1, 5, 5)),
                padding=2,
            )[0, 0]
            query_x = torch.floor(reference_points[batch_index, :, 0] * grid_size)
            query_y = torch.floor(reference_points[batch_index, :, 1] * grid_size)
            query_x = query_x.long().clamp(0, grid_size - 1)
            query_y = query_y.long().clamp(0, grid_size - 1)
            lidar_counts[batch_index] = window[query_y, query_x]

            query_xyz = xyz[batch_index].detach().cpu().numpy()
            homogeneous = np.concatenate(
                [query_xyz, np.ones((num_queries, 1), dtype=query_xyz.dtype)],
                axis=1,
            )
            count = np.zeros(num_queries, dtype=np.float32)
            corruption = img_metas[batch_index].get('qta_corruption', {})
            details = corruption.get('details', {})
            occluded_views = set(details.get('selected_views', []))
            polygons = details.get('polygons_xy', {})
            for view_index, matrix in enumerate(img_metas[batch_index]['lidar2img']):
                projection = homogeneous @ np.asarray(matrix).T
                depth_ok = projection[:, 2] > 1e-5
                pixels = projection[:, :2] / np.clip(
                    projection[:, 2:3], 1e-5, None
                )
                height, width = img_metas[batch_index]['img_shape'][view_index][:2]
                view_valid = (
                    depth_ok
                    & (pixels[:, 0] >= 0)
                    & (pixels[:, 0] < width)
                    & (pixels[:, 1] >= 0)
                    & (pixels[:, 1] < height)
                )
                if view_index in occluded_views and str(view_index) in polygons:
                    polygon = np.asarray(
                        polygons[str(view_index)], dtype=np.float64
                    )
                    view_valid &= ~self._points_in_polygon(pixels, polygon)
                count += view_valid.astype(np.float32)
            valid_views[batch_index] = torch.from_numpy(count).to(
                device=reference_points.device
            )
        return lidar_counts, valid_views

    def forward_qta_route_bundle(self, points, img_metas, img,
                                 gt_bboxes_3d, gt_labels_3d):
        """Run original MoME plus all three complete routes on shared features."""

        if self.training:
            raise RuntimeError('forward_qta_route_bundle requires model.eval()')
        self._require_qta_probe_state()
        img_feats, pts_feats = self.extract_feat(points, img=img, img_metas=img_metas)
        if pts_feats is None:
            pts_feats = [None]
        if img_feats is None:
            img_feats = [None]

        def execute(override, expose_state=False):
            return self.pts_bbox_head(
                pts_feats,
                img_feats,
                img_metas,
                route_override=override,
                return_router_state=expose_state,
            )

        base_preds = execute(None, expose_state=True)
        route_preds = [execute(route) for route in range(3)]
        fused_targets = self.pts_bbox_head.build_query_targets(
            gt_bboxes_3d, gt_labels_3d, route_preds[0]
        )
        route_fixed = [
            self.pts_bbox_head.return_query_losses(
                gt_bboxes_3d,
                gt_labels_3d,
                predictions,
                fixed_targets=fused_targets,
            )
            for predictions in route_preds
        ]
        route_query_losses = torch.stack(
            [item['total'] for item in route_fixed], dim=-1
        )
        base_task = base_preds[0][0]
        base_routes = base_task['base_routes'].long()
        if route_query_losses.shape[:2] != base_routes.shape:
            raise RuntimeError(
                'three-route proxy losses must align with original MoME query ids'
            )
        # Normal MoME inference groups decoded queries by selected expert before
        # concatenation, whereas AQR features and routes retain the original
        # query order.  Therefore L_i(r_i^0) must be gathered from the three
        # complete, query-aligned route executions rather than indexed from the
        # grouped default prediction tensor.
        base_query_losses = route_query_losses.gather(
            dim=-1, index=base_routes.unsqueeze(-1)
        ).squeeze(-1)
        route_rematched_group_losses = torch.stack(
            [
                self.pts_bbox_head.return_query_losses(
                    gt_bboxes_3d,
                    gt_labels_3d,
                    predictions,
                    fixed_targets=None,
                )['total'].sum(dim=1)
                for predictions in route_preds
            ],
            dim=-1,
        )
        affected_query = self.qta_affected_query_mask(
            base_task['reference_points'], img_metas, gt_bboxes_3d
        )
        lidar_point_count, valid_camera_views = self.qta_local_evidence(
            base_task['reference_points'], img_metas, points
        )
        return {
            'router_features': base_task['router_features'],
            'reference_points': base_task['reference_points'],
            'base_routes': base_routes,
            'base_query_losses': base_query_losses,
            'route_query_losses': route_query_losses,
            'valid_mask': torch.isfinite(route_query_losses),
            'positive_mask': route_fixed[0]['positive_mask'],
            'affected_query': affected_query,
            'lidar_point_count': lidar_point_count,
            'valid_camera_views': valid_camera_views,
            'route_rematched_group_losses': route_rematched_group_losses,
        }

    def forward_qta_full_query_oracle(self, points, img_metas, img,
                                      gt_bboxes_3d, gt_labels_3d,
                                      return_predictions=True,
                                      return_raw_base=False,
                                      repeat_base_on_shared_features=False):
        """Apply all unique positive-GT expert switches in one decoder pass.

        The three complete expert executions share extracted features and use
        one Fused-Hungarian target bundle, so every ``[B,N,3]`` loss row refers
        to the same original query and GT target.  The Oracle route vector is
        selected without thresholds or labels from any other frame, applied
        simultaneously, and then decoded once using the canonical MoME head.
        """

        if self.training:
            raise RuntimeError('forward_qta_full_query_oracle requires model.eval()')
        if len(img_metas) != 1:
            raise ValueError('full-query Oracle requires batch size 1')
        self._require_qta_probe_state()
        img_feats, pts_feats = self.extract_feat(points, img=img, img_metas=img_metas)
        if pts_feats is None:
            pts_feats = [None]
        if img_feats is None:
            img_feats = [None]

        def execute(override, expose_state=False):
            return self.pts_bbox_head(
                pts_feats,
                img_feats,
                img_metas,
                route_override=override,
                return_router_state=expose_state,
            )

        base_preds = execute(None, expose_state=True)
        shared_feature_base_repeat = (
            execute(None, expose_state=True)
            if repeat_base_on_shared_features
            else None
        )
        route_preds = [execute(route) for route in range(3)]
        fused_targets = self.pts_bbox_head.build_query_targets(
            gt_bboxes_3d, gt_labels_3d, route_preds[0]
        )
        route_fixed = [
            self.pts_bbox_head.return_query_losses(
                gt_bboxes_3d,
                gt_labels_3d,
                predictions,
                fixed_targets=fused_targets,
            )
            for predictions in route_preds
        ]
        route_query_losses = torch.stack(
            [item['total'] for item in route_fixed], dim=-1
        )
        base_task = base_preds[0][0]
        base_routes = base_task['base_routes'].long()
        oracle_routes, switch_mask, query_gains, best_routes = (
            select_full_query_oracle_routes(base_routes, route_query_losses)
        )
        oracle_preds = execute(oracle_routes, expose_state=True)
        base_rematched = self.pts_bbox_head.return_query_losses(
            gt_bboxes_3d, gt_labels_3d, base_preds, fixed_targets=None
        )['total'].sum(dim=1)
        oracle_rematched = self.pts_bbox_head.return_query_losses(
            gt_bboxes_3d, gt_labels_3d, oracle_preds, fixed_targets=None
        )['total'].sum(dim=1)
        output = {
            'base_routes': base_routes,
            'oracle_routes': oracle_routes,
            'route_query_losses': route_query_losses,
            'query_gains': query_gains,
            'best_routes': best_routes,
            'switch_mask': switch_mask,
            'base_rematched_group_losses': base_rematched,
            'oracle_rematched_group_losses': oracle_rematched,
            'router_features': base_task['router_features'],
            'reference_points': base_task['reference_points'],
        }
        if return_predictions:
            base_bbox_list = self.pts_bbox_head.get_bboxes(
                base_preds, img_metas, rescale=False
            )
            oracle_bbox_list = self.pts_bbox_head.get_bboxes(
                oracle_preds, img_metas, rescale=False
            )
            output.update(
                {
                    'base_bbox_results': [
                        bbox3d2result(bboxes, scores, labels)
                        for bboxes, scores, labels in base_bbox_list
                    ],
                    'oracle_bbox_results': [
                        bbox3d2result(bboxes, scores, labels)
                        for bboxes, scores, labels in oracle_bbox_list
                    ],
                }
            )
        if return_raw_base:
            output['base_raw_predictions'] = base_preds
        if shared_feature_base_repeat is not None:
            output['shared_feature_base_repeat_predictions'] = (
                shared_feature_base_repeat
            )
        return output

    @staticmethod
    def _qta_route_hash(routes):
        payload = (
            routes.detach().to(device='cpu', dtype=torch.int8)
            .contiguous().numpy().tobytes()
        )
        return hashlib.sha256(payload).hexdigest()

    @staticmethod
    def _qta_bbox_results(bbox_list):
        return [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]

    @staticmethod
    def _qta_bbox_list_exact(left, right):
        if len(left) != len(right):
            return False
        for left_item, right_item in zip(left, right):
            left_boxes, left_scores, left_labels = left_item
            right_boxes, right_scores, right_labels = right_item
            left_tensor = getattr(left_boxes, 'tensor', left_boxes)
            right_tensor = getattr(right_boxes, 'tensor', right_boxes)
            if not (
                torch.equal(left_tensor, right_tensor)
                and torch.equal(left_scores, right_scores)
                and torch.equal(left_labels, right_labels)
            ):
                return False
        return True

    def forward_qta_context_preserving_output_oracle(
        self,
        points,
        img_metas,
        img,
        gt_bboxes_3d,
        gt_labels_3d,
        epsilon_query_keep=0.0,
        epsilon_query_reconstruction=0.0,
        return_predictions=True,
        return_raw_predictions=False,
    ):
        """Compose complete-context final head outputs without decoder reruns."""

        if self.training:
            raise RuntimeError(
                'forward_qta_context_preserving_output_oracle requires model.eval()'
            )
        if len(img_metas) != 1:
            raise ValueError('context-preserving Oracle requires batch size one')
        self._require_qta_probe_state()
        img_feats, pts_feats = self.extract_feat(
            points, img=img, img_metas=img_metas
        )
        if pts_feats is None:
            pts_feats = [None]
        if img_feats is None:
            img_feats = [None]

        decoder_call_count = 0

        def execute(override):
            nonlocal decoder_call_count
            decoder_call_count += 1
            return self.pts_bbox_head(
                pts_feats,
                img_feats,
                img_metas,
                route_override=override,
                return_router_state=True,
            )

        original_preds = execute(None)
        route_preds = [execute(route) for route in range(3)]
        original_indices = self.pts_bbox_head.validate_s2_prediction_bundle(
            original_preds, 'original_mome', expected_query_count=900
        )
        route_indices = [
            self.pts_bbox_head.validate_s2_prediction_bundle(
                predictions,
                f'full_context_route_{route}',
                expected_query_count=900,
            )
            for route, predictions in enumerate(route_preds)
        ]
        all_keep_preds = self.pts_bbox_head.compose_final_detection_head_outputs(
            original_preds,
            route_preds,
            torch.full_like(original_indices, -1),
            label='all_keep_identity_audit',
        )
        all_keep_tensor_exact = all(
            torch.equal(
                original_preds[0][task_index][field],
                all_keep_preds[0][task_index][field],
            )
            for task_index in range(len(original_preds[0]))
            for field in ('cls_logits', 'center', 'height', 'dim', 'rot', 'vel')
        )
        if not all_keep_tensor_exact:
            raise RuntimeError('all-KEEP final head composition drifted from MoME')
        original_task = original_preds[0][0]
        original_routes = original_task['base_routes'].long()
        if original_routes.shape != original_indices.shape:
            raise RuntimeError('original route vector does not align with query ids')

        keep_targets = self.pts_bbox_head.build_query_targets(
            gt_bboxes_3d, gt_labels_3d, original_preds
        )
        keep_losses = self.pts_bbox_head.query_losses_in_original_order(
            original_preds,
            keep_targets,
            original_indices,
            original_indices,
        )
        route_keep_losses = [
            self.pts_bbox_head.query_losses_in_original_order(
                predictions,
                keep_targets,
                original_indices,
                prediction_indices,
            )
            for predictions, prediction_indices in zip(route_preds, route_indices)
        ]
        route_query_losses = torch.stack(
            [losses['total'] for losses in route_keep_losses], dim=-1
        )
        keep_actions, keep_switch_mask, keep_lower_bounds, keep_best_lb = (
            select_robust_oracle_actions(
                keep_losses['total'],
                route_query_losses,
                epsilon_query_keep,
                original_routes,
            )
        )
        keep_oracle_preds = (
            self.pts_bbox_head.compose_final_detection_head_outputs(
                original_preds,
                route_preds,
                keep_actions,
                label='keep_anchored_output_oracle',
            )
        )

        reconstruction_source_routes = original_routes.clone()
        reconstruction_preds = (
            self.pts_bbox_head.compose_final_detection_head_outputs(
                original_preds,
                route_preds,
                reconstruction_source_routes,
                label='pure_original_route_full_context_reconstruction',
            )
        )
        reconstruction_losses = (
            self.pts_bbox_head.query_losses_in_original_order(
                reconstruction_preds,
                keep_targets,
                original_indices,
                original_indices,
            )
        )
        portfolio_actions, portfolio_switch_mask, portfolio_lower_bounds, portfolio_best_lb = (
            select_robust_oracle_actions(
                reconstruction_losses['total'],
                route_query_losses,
                epsilon_query_reconstruction,
                original_routes,
            )
        )
        portfolio_source_routes = torch.where(
            portfolio_switch_mask,
            portfolio_actions,
            reconstruction_source_routes,
        )
        portfolio_preds = self.pts_bbox_head.compose_final_detection_head_outputs(
            reconstruction_preds,
            route_preds,
            portfolio_source_routes,
            label='pure_three_expert_portfolio_oracle',
        )

        prediction_groups = {
            'original_mome': original_preds,
            'keep_anchored_output_oracle': keep_oracle_preds,
            'pure_original_route_full_context_reconstruction': reconstruction_preds,
            'pure_three_expert_portfolio_oracle': portfolio_preds,
        }
        rematched_losses = {
            name: self.pts_bbox_head.return_query_losses(
                gt_bboxes_3d,
                gt_labels_3d,
                predictions,
                fixed_targets=None,
            )['total'].sum(dim=1)
            for name, predictions in prediction_groups.items()
        }

        fused_targets = self.pts_bbox_head.build_query_targets(
            gt_bboxes_3d, gt_labels_3d, route_preds[0]
        )
        fused_route_losses = torch.stack(
            [
                self.pts_bbox_head.query_losses_in_original_order(
                    predictions,
                    fused_targets,
                    route_indices[0],
                    prediction_indices,
                )['total']
                for predictions, prediction_indices in zip(
                    route_preds, route_indices
                )
            ],
            dim=-1,
        )
        fused_reconstruction_losses = fused_route_losses.gather(
            -1, original_routes.unsqueeze(-1)
        ).squeeze(-1)
        fused_actions, fused_switch_mask, fused_lower_bounds, _ = (
            select_robust_oracle_actions(
                fused_reconstruction_losses,
                fused_route_losses,
                epsilon_query_reconstruction,
                original_routes,
            )
        )

        output = {
            'query_identity': {
                'original_mome': original_indices,
                'full_context_experts': route_indices,
            },
            'original_routes': original_routes,
            'keep_query_losses': keep_losses,
            'full_context_query_losses': route_query_losses,
            'keep_anchored_actions': keep_actions,
            'keep_anchored_switch_mask': keep_switch_mask,
            'keep_anchored_lower_bounds': keep_lower_bounds,
            'keep_anchored_best_lower_bound': keep_best_lb,
            'reconstruction_query_losses': reconstruction_losses,
            'portfolio_actions': portfolio_actions,
            'portfolio_switch_mask': portfolio_switch_mask,
            'portfolio_source_routes': portfolio_source_routes,
            'portfolio_lower_bounds': portfolio_lower_bounds,
            'portfolio_best_lower_bound': portfolio_best_lb,
            'rematched_group_losses': rematched_losses,
            'fused_hungarian_sensitivity': {
                'actions': fused_actions,
                'switch_mask': fused_switch_mask,
                'lower_bounds': fused_lower_bounds,
                'route_agreement': (fused_actions == portfolio_actions),
            },
            'final_detection_head_fields': {
                name: self.pts_bbox_head.final_detection_head_fields(predictions)
                for name, predictions in prediction_groups.items()
            },
            'decoder_call_count': decoder_call_count,
            'all_keep_tensor_exact': all_keep_tensor_exact,
        }
        if return_predictions:
            bbox_lists = {
                name: self.pts_bbox_head.get_bboxes(
                    predictions, img_metas, rescale=False
                )
                for name, predictions in prediction_groups.items()
            }
            all_keep_bbox_list = self.pts_bbox_head.get_bboxes(
                all_keep_preds, img_metas, rescale=False
            )
            all_keep_bbox_exact = self._qta_bbox_list_exact(
                bbox_lists['original_mome'], all_keep_bbox_list
            )
            if not all_keep_bbox_exact:
                raise RuntimeError('all-KEEP bbox decode drifted from MoME')
            output['all_keep_bbox_exact'] = all_keep_bbox_exact
            output['bbox_results'] = {
                name: self._qta_bbox_results(bbox_list)
                for name, bbox_list in bbox_lists.items()
            }
        if return_raw_predictions:
            output['raw_predictions'] = prediction_groups
        return output

    def forward_qta_greedy_joint_oracle(
        self,
        points,
        img_metas,
        img,
        gt_bboxes_3d,
        gt_labels_3d,
        epsilon_query=0.0,
        proxy_overestimate=0.0,
        epsilon_frame=0.0,
        candidate_k=16,
        locked_candidates=None,
        replay_acceptance=None,
        locked_base_routes=None,
        return_predictions=True,
    ):
        """Run the locked GT-guided greedy route search on shared features."""

        if self.training:
            raise RuntimeError(
                'forward_qta_greedy_joint_oracle requires model.eval()'
            )
        if len(img_metas) != 1:
            raise ValueError('greedy joint Oracle requires batch size one')
        candidate_k = int(candidate_k)
        if candidate_k < 0 or candidate_k > 16:
            raise ValueError('candidate_k must lie in [0,16]')
        self._require_qta_probe_state()
        img_feats, pts_feats = self.extract_feat(
            points, img=img, img_metas=img_metas
        )
        if pts_feats is None:
            pts_feats = [None]
        if img_feats is None:
            img_feats = [None]

        decoder_call_count = 0

        def execute(override):
            nonlocal decoder_call_count
            decoder_call_count += 1
            return self.pts_bbox_head(
                pts_feats,
                img_feats,
                img_metas,
                route_override=override,
                return_router_state=True,
            )

        base_preds = execute(locked_base_routes)
        route_preds = [execute(route) for route in range(3)]
        base_indices = self.pts_bbox_head.validate_s2_prediction_bundle(
            base_preds, 'greedy_original_mome', expected_query_count=900
        )
        route_indices = [
            self.pts_bbox_head.validate_s2_prediction_bundle(
                predictions,
                f'greedy_full_context_route_{route}',
                expected_query_count=900,
            )
            for route, predictions in enumerate(route_preds)
        ]
        native_base_routes = base_preds[0][0]['base_routes'].long()
        if locked_base_routes is None:
            base_routes = native_base_routes
        else:
            if 'final_routes' not in base_preds[0][0]:
                raise RuntimeError(
                    'locked replay did not expose the executed final routes'
                )
            executed_routes = base_preds[0][0]['final_routes'].long()
            base_routes = normalize_locked_route_state(
                locked_base_routes, executed_routes
            )
            if not torch.equal(executed_routes, base_routes):
                raise RuntimeError(
                    'locked replay base routes were not executed exactly'
                )
        if base_routes.shape != base_indices.shape:
            raise RuntimeError('greedy base routes do not align with query ids')
        keep_targets = self.pts_bbox_head.build_query_targets(
            gt_bboxes_3d, gt_labels_3d, base_preds
        )
        base_fixed_losses = self.pts_bbox_head.query_losses_in_original_order(
            base_preds, keep_targets, base_indices, base_indices
        )['total']
        route_fixed_losses = torch.stack(
            [
                self.pts_bbox_head.query_losses_in_original_order(
                    predictions,
                    keep_targets,
                    base_indices,
                    prediction_indices,
                )['total']
                for predictions, prediction_indices in zip(
                    route_preds, route_indices
                )
            ],
            dim=-1,
        )
        reconstruction_losses = route_fixed_losses.gather(
            -1, base_routes.unsqueeze(-1)
        ).squeeze(-1)
        full_context_gains = (
            reconstruction_losses.unsqueeze(-1) - route_fixed_losses
        )
        candidate_quality = robust_candidate_quality(
            full_context_gains,
            epsilon_query,
            proxy_overestimate,
            base_routes,
        )
        if locked_candidates is None:
            ranked = rank_greedy_candidates(candidate_quality)
        else:
            ranked = []
            seen = set()
            for item in locked_candidates:
                if isinstance(item, dict):
                    query_id = int(item['query_id'])
                    expert_id = int(item['destination_expert'])
                else:
                    query_id, expert_id = int(item[0]), int(item[1])
                if query_id < 0 or query_id >= base_routes.shape[1]:
                    raise ValueError(f'locked candidate query out of range: {query_id}')
                if expert_id not in (0, 1, 2):
                    raise ValueError(f'locked candidate expert invalid: {expert_id}')
                key = (query_id, expert_id)
                if key in seen:
                    raise ValueError(f'duplicate locked candidate: {key}')
                seen.add(key)
                ranked.append(
                    (
                        query_id,
                        expert_id,
                        float(candidate_quality[0, query_id, expert_id].item()),
                    )
                )
        candidates = ranked[:candidate_k]
        if replay_acceptance is not None and len(replay_acceptance) != len(candidates):
            raise ValueError('replay_acceptance must align with selected candidates')

        base_rematched = self.pts_bbox_head.return_query_losses(
            gt_bboxes_3d, gt_labels_3d, base_preds, fixed_targets=None
        )['total'].sum(dim=1)
        current_routes = base_routes.clone()
        current_preds = base_preds
        current_indices = base_indices
        current_fixed_losses = base_fixed_losses
        current_rematched = base_rematched
        trace = []
        frame_margin = float(torch.as_tensor(epsilon_frame).item())
        if not np.isfinite(frame_margin) or frame_margin < 0:
            raise ValueError('epsilon_frame must be finite and non-negative')
        for step_index, (query_id, expert_id, quality) in enumerate(candidates):
            current_hash = self._qta_route_hash(current_routes)
            if int(current_routes[0, query_id].item()) == expert_id:
                trace.append(
                    {
                        'step': step_index,
                        'query_id': query_id,
                        'destination_expert': expert_id,
                        'quality': quality,
                        'current_route_hash': current_hash,
                        'candidate_route_hash': current_hash,
                        'proxy_full_context_gain': full_context_gains[
                            0, query_id, expert_id
                        ],
                        'actual_query_gain': None,
                        'actual_frame_gain': None,
                        'accepted': False,
                        'reason': 'skipped_noop_after_prior_accept',
                    }
                )
                continue
            candidate_routes = current_routes.clone()
            candidate_routes[0, query_id] = expert_id
            candidate_hash = self._qta_route_hash(candidate_routes)
            candidate_preds = execute(candidate_routes)
            candidate_indices = self.pts_bbox_head.validate_s2_prediction_bundle(
                candidate_preds,
                f'greedy_candidate_step_{step_index}',
                expected_query_count=900,
            )
            candidate_fixed_losses = (
                self.pts_bbox_head.query_losses_in_original_order(
                    candidate_preds,
                    keep_targets,
                    base_indices,
                    candidate_indices,
                )['total']
            )
            candidate_rematched = self.pts_bbox_head.return_query_losses(
                gt_bboxes_3d,
                gt_labels_3d,
                candidate_preds,
                fixed_targets=None,
            )['total'].sum(dim=1)
            actual_query_gain = (
                current_fixed_losses[0, query_id]
                - candidate_fixed_losses[0, query_id]
            )
            actual_frame_gain = current_rematched[0] - candidate_rematched[0]
            would_accept = bool(
                torch.isfinite(actual_frame_gain).item()
                and float(actual_frame_gain.item()) > frame_margin
            )
            trajectory_advance = (
                bool(replay_acceptance[step_index])
                if replay_acceptance is not None
                else would_accept
            )
            if replay_acceptance is None:
                accepted = would_accept
                if accepted and not (
                    float(candidate_rematched[0].item())
                    < float(current_rematched[0].item())
                ):
                    raise RuntimeError(
                        'accepted greedy transition did not strictly lower frame loss'
                    )
            else:
                accepted = False
            if trajectory_advance:
                current_routes = candidate_routes
                current_preds = candidate_preds
                current_indices = candidate_indices
                current_fixed_losses = candidate_fixed_losses
                current_rematched = candidate_rematched
            reason = (
                'accepted_gt_frame_gain_above_margin'
                if accepted
                else 'rejected_nonfinite_frame_gain'
                if not torch.isfinite(actual_frame_gain).item()
                else 'rejected_frame_gain_not_above_margin'
            )
            if replay_acceptance is not None:
                reason = 'calibration_replay_locked_trajectory'
            trace.append(
                {
                    'step': step_index,
                    'query_id': query_id,
                    'destination_expert': expert_id,
                    'quality': quality,
                    'current_route_hash': current_hash,
                    'candidate_route_hash': candidate_hash,
                    'proxy_full_context_gain': full_context_gains[
                        0, query_id, expert_id
                    ],
                    'actual_query_gain': actual_query_gain,
                    'actual_frame_gain': actual_frame_gain,
                    'would_accept': would_accept,
                    'accepted': accepted,
                    'trajectory_advanced': trajectory_advance,
                    'reason': reason,
                }
            )

        output = {
            'K': candidate_k,
            'candidate_quality': candidate_quality,
            'candidate_order': candidates,
            'base_fixed_query_losses': base_fixed_losses,
            'route_fixed_query_losses': route_fixed_losses,
            'reconstruction_fixed_query_losses': reconstruction_losses,
            'full_context_gains': full_context_gains,
            'base_routes': base_routes,
            'native_base_route_hash': self._qta_route_hash(native_base_routes),
            'final_routes': current_routes,
            'base_route_hash': self._qta_route_hash(base_routes),
            'final_route_hash': self._qta_route_hash(current_routes),
            'base_rematched_group_loss': base_rematched,
            'final_rematched_group_loss': current_rematched,
            'trace': trace,
            'accepted_count': sum(int(item['accepted']) for item in trace),
            'decoder_call_count': decoder_call_count,
            'decoder_call_limit': 4 + candidate_k,
            'calibration_replay': replay_acceptance is not None,
            'final_query_indices': current_indices,
        }
        if decoder_call_count > 4 + candidate_k:
            raise RuntimeError('greedy decoder call count exceeded 4+K')
        if return_predictions:
            output['original_mome_greedy_run_bbox_results'] = (
                self._qta_bbox_results(
                    self.pts_bbox_head.get_bboxes(
                        base_preds, img_metas, rescale=False
                    )
                )
            )
            output['greedy_joint_routing_gt_oracle_result_bbox_results'] = (
                self._qta_bbox_results(
                    self.pts_bbox_head.get_bboxes(
                        current_preds, img_metas, rescale=False
                    )
                )
            )
        return output

    def forward_qta_single_query_audit(self, points, img_metas, img,
                                       gt_bboxes_3d, gt_labels_3d,
                                       query_indices, destination_routes,
                                       base_route_override=None,
                                       return_oracle_predictions=False):
        """Audit one-query interventions from an optional locked base route."""

        if self.training:
            raise RuntimeError('forward_qta_single_query_audit requires model.eval()')
        if len(img_metas) != 1:
            raise ValueError('strict single-query audit requires batch size 1')
        if len(query_indices) != len(destination_routes):
            raise ValueError('query_indices and destination_routes must align')
        self._require_qta_probe_state()
        img_feats, pts_feats = self.extract_feat(points, img=img, img_metas=img_metas)
        if pts_feats is None:
            pts_feats = [None]
        if img_feats is None:
            img_feats = [None]

        def execute(override, expose_state=False):
            return self.pts_bbox_head(
                pts_feats,
                img_feats,
                img_metas,
                route_override=override,
                return_router_state=expose_state,
            )

        base_preds = execute(base_route_override, expose_state=True)
        fused_preds = execute(0, expose_state=True)
        fused_targets = self.pts_bbox_head.build_query_targets(
            gt_bboxes_3d, gt_labels_3d, fused_preds
        )
        base_rematched = self.pts_bbox_head.return_query_losses(
            gt_bboxes_3d, gt_labels_3d, base_preds, fixed_targets=None
        )['total'].sum(dim=1)
        base_task = base_preds[0][0]
        live_base_routes = base_task['base_routes']
        base_routes = base_task['final_routes']
        base_output_query_indices = base_task['output_query_indices']
        fused_output_query_indices = fused_preds[0][0]['output_query_indices']
        base_query_losses = {}
        for query_index in set(int(value) for value in query_indices):
            base_query_losses[query_index] = (
                self.pts_bbox_head.query_loss_for_original_query(
                    base_preds,
                    fused_targets,
                    base_output_query_indices,
                    fused_output_query_indices,
                    query_index,
                )
            )
        records = []
        best_preds = base_preds
        best_score = None
        best_query_index = None
        best_destination_route = None
        for query_index, destination_route in zip(query_indices, destination_routes):
            query_index = int(query_index)
            destination_route = int(destination_route)
            override = base_routes.clone()
            if query_index < 0 or query_index >= override.shape[1]:
                raise IndexError(f'query index out of range: {query_index}')
            if destination_route not in (0, 1, 2):
                raise ValueError(f'invalid destination route: {destination_route}')
            override[0, query_index] = destination_route
            candidate_preds = execute(override, expose_state=True)
            candidate_query_loss = (
                self.pts_bbox_head.query_loss_for_original_query(
                    candidate_preds,
                    fused_targets,
                    candidate_preds[0][0]['output_query_indices'],
                    fused_output_query_indices,
                    query_index,
                )
            )
            candidate_rematched = self.pts_bbox_head.return_query_losses(
                gt_bboxes_3d,
                gt_labels_3d,
                candidate_preds,
                fixed_targets=None,
            )['total'].sum(dim=1)
            query_advantage = (
                base_query_losses[query_index] - candidate_query_loss
            )
            group_advantage = base_rematched[0] - candidate_rematched[0]
            records.append(
                {
                    'query_index': query_index,
                    'base_route': int(base_routes[0, query_index].item()),
                    'destination_route': destination_route,
                    'strict_query_advantage': query_advantage,
                    'strict_group_advantage': group_advantage,
                    'strict_positive': (query_advantage > 0) & (group_advantage > 0),
                }
            )
            strict_positive = bool(
                torch.isfinite(query_advantage).item()
                and torch.isfinite(group_advantage).item()
                and (query_advantage > 0).item()
                and (group_advantage > 0).item()
            )
            candidate_score = min(
                float(query_advantage.item()), float(group_advantage.item())
            )
            if strict_positive and (best_score is None or candidate_score > best_score):
                best_preds = candidate_preds
                best_score = candidate_score
                best_query_index = query_index
                best_destination_route = destination_route
        output = {
            'base_routes': base_routes,
            'live_base_routes': live_base_routes,
            'router_features': base_task['router_features'],
            'reference_points': base_task['reference_points'],
            'records': records,
        }
        if return_oracle_predictions:
            base_bbox_list = self.pts_bbox_head.get_bboxes(
                base_preds, img_metas, rescale=False
            )
            oracle_bbox_list = self.pts_bbox_head.get_bboxes(
                best_preds, img_metas, rescale=False
            )
            output.update(
                {
                    'base_bbox_results': [
                        bbox3d2result(bboxes, scores, labels)
                        for bboxes, scores, labels in base_bbox_list
                    ],
                    'oracle_bbox_results': [
                        bbox3d2result(bboxes, scores, labels)
                        for bboxes, scores, labels in oracle_bbox_list
                    ],
                    'oracle_query_index': best_query_index,
                    'oracle_destination_route': best_destination_route,
                    'oracle_strict_advantage': best_score,
                }
            )
        return output

"""S4-only audit routing around the frozen S2 train corruptions."""

from __future__ import annotations

import json
import os
from pathlib import Path

from mmdet.datasets.builder import PIPELINES

try:
    from .stage019_s2_training_conditions import Stage019S2TrainingConditionAdapter
except ImportError:
    from stage019_s2_training_conditions import Stage019S2TrainingConditionAdapter


S4_AUDIT_ENV = "VISFUSE3D_STAGE019_S4_AUDIT_PATH"


@PIPELINES.register_module()
class Stage019S4TrainingConditionAdapter(Stage019S2TrainingConditionAdapter):
    """Reuse exact corruption math while routing every row to its scene triad."""

    def _audit(self, payload) -> None:
        destination = os.environ.get(S4_AUDIT_ENV)
        if not destination:
            raise RuntimeError(f"{S4_AUDIT_ENV} is not set for S4 train extraction")
        path = Path(destination).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        output = dict(payload)
        output["schema"] = "visfuse3d_stage019_s4_training_condition_audit_v1"
        output["protocol_profile"] = "stage019_s4_object_set_attribute_fusion_v1"
        with path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(output, sort_keys=True, allow_nan=False) + "\n")

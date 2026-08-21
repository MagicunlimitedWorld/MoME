# Stage019-S4 terminal closeout

- Run: `2026-08-20-mome-stage019-s4-object-set-attribute-fusion-v1`
- Terminal attempt: `attempt_007`
- Terminal status: `completed_negative_pilot_cp_afr_attribute_gain_gate`
- Executed model-source commit: `c9ea29006408a1e47ae2f984ee665ee317fbb1ac`
- Attempt-007 source relock SHA256: `9ADC987F9966DE589C939389BF0BF5E569FBA15ADCC3A9D3408D8E15EBE95333`

G0 score and attribute gates passed; G1 and G2 calibration gates also passed, selecting primary CP-AFR in `stacked` mode at fusion strength `0.5`. The primary-only pilot then stopped at the locked scientific gate. Its gate manifest SHA256 is `D98CC8BD673A22B9099237C77D70179F11DD4A2AB308A80ACE2B1949CE11C4DB`, and the scientific-failure manifest SHA256 is `4F0FE0B83030E5B4FB95C7A7935DE2F530921681EBE90938E32DE46C7642395B`.

The pilot fault macro improved over original MoME by `+0.008073788079821342` mAP and `+0.004657131506021539` NDS. However, no individual actionable fault met the preregistered CP-AFR-versus-score-baseline NDS increment of `+0.0025`; the failed check was `cp_afr_one_fault_nds_plus_0p0025_vs_score`. The gate returned the complete scientific-negative code (`rc=3`), not an engineering failure. Thresholds were not relaxed, and full validation was not run. Therefore this run provides no formal full-validation result and authorizes no method-success claim.

The terminal closeout manifest SHA256 is `2DF711ECEC7ADD853021E38673BC04D169E75EE70BBF1A1D86E9E07F24E88D17`. The immutable artifact was archived at:

`D:\VisFuse3D\VisFuse3D_static\artifacts_archive\experiment_result_archives\2026-08-20-stage019-s4-terminal-closeout\roots\2026-08-20-mome-stage019-s4-object-set-attribute-fusion-v1`

The archive batch manifest has SHA256 `0ACAD51A7F4A23085A853EE472719C12CA3BE27997A1E47F8E378526BA616126`, status `archived_verified`, 49,869 files, 32,709,955,580 bytes, zero reparse points, and inventory-tree SHA256 `E93C9F487E65392727DB6C4A8F19F74BBBB7C69B14DF34AE969696E5B6347C0F`.

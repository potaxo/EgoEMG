# 磁盘字段至模型输入核查

| 状态 | 项目 | 证据/实际行为 |
| --- | --- | --- |
| 已确认 | field | "emg_right_filtered_paper" |
| 已确认 | sample_rate_hz | 2000 |
| 已确认 | window_rows | 7790 |
| 已确认 | window_nominal_seconds | 3.895 |
| 已确认 | hand | "right" |
| 已确认 | raw_shape | "T,8" |
| 已确认 | layout_slots | 16 |
| 已确认 | channel_map | [10, 12, 0, 1, 2, 4, 5, 6] |
| 已确认 | mirror | "none in executed loader path; upstream manifest left_hand_strategy is recorded separately" |
| 已确认 | interpolation | false |
| 已确认 | order | "read float32 -> sparse 8-to-16 zero placement -> scalar normalization (std+1e-6) -> transpose C,T -> CUDA FP32" |
| 已确认 | requested_key | "egoemg__filtered_paper" |
| 已确认 | requested_entry | {"note": "placeholder — recompute via scripts/data/compute_per_dataset_norm_stats.py with --emg-fields filtered_paper"} |
| 已确认 | actual_key | "egoemg" |
| 已确认 | mean | 0.028774764388799667 |
| 已确认 | std | 94.73531341552734 |
| 已确认 | padding_after_normalization | -0.00030373852207412336 |
| 已确认 | constant_control | "raw zero live signal after same sparse placement and normalization: every slot equals padding_after_normalization" |
| 已确认 | legacy_zero_control | "all slots set to zero AFTER normalization, matching historical F2" |
| 已确认 | rgb_mean | [123.67500305175781, 116.27999877929688, 103.52999877929688] |
| 已确认 | rgb_std | [58.39500045776367, 57.12000274658203, 57.375] |
| 已确认 | rgb | "read-only precrop LMDB, uint8 RGB 256x256, transpose CHW, no divide by 255 before mean/std" |
| 已确认 | alignment | "center=start+3895; same global center image mapping and reference; output 22x1 directly center supervised" |
| 已确认 | angle_units | "generated fingers already radians; wrist stored degrees converted to radians; MAE first20 linear absolute radians -> degrees" |
| 已确认 | joint_order | ["thumb_cmc_fe", "thumb_cmc_aa", "thumb_mcp_fe", "thumb_ip_fe", "index_mcp_aa", "index_mcp_fe", "index_pip_fe", "index_dip_fe", "middle_mcp_aa", "middle_mcp_fe", "middle_pip_fe", "middle_dip_fe", "ring_mcp_aa", "ring_mcp_fe", "ring_pip_fe", "ring_dip_fe", "pinky_mcp_aa", "pinky_mcp_fe", "pinky_pip_fe", "pinky_dip_fe"] |
| 已确认 | strict_load | true |
| 已确认 | model_mode | "center_supervised" |
| 已确认 | encoder_frozen_eval | true |
| 未知 | training_statistics | "checkpoint records original training-machine path but no stats bytes/hash; legacy choice at original execution not provable from supplied checkpoint" |
| 未知 | checkpoint_participant_exposure | "not fully traced" |
| 未知 | upstream_filter | "filtered_paper name and supplied preprocessing code available; original complete processing execution not re-run" |
| 未知 | historical_training_logs | "logs/fusion/resnet_small_emgfusion_center absent locally" |
| 已确认 | 名义采样率与timestamp_us | 全录制跨度推算约2000Hz；逐行timestamp_us成批到达且非均匀；加载器不按timestamp重采样，时间块按名义行时间定义。详见input_support_supplement.json。 |
| 未知 | EMG物理单位 | not specified by supplied manifest; do not infer microvolts |
| 已确认 | checkpoint/current共享配置 | 规范化emg2pose→egoemg命名空间后，共享字段除视觉初始化文件路径外一致；完整发布fusion state严格覆盖初始化。原始执行源码数值等价未恢复。 |

## 证据来源分层

{
  "code": "egoemg/datasets/egoemg_memmap_dataset.py:_load_norm_stats/_convert_to_emg2pose_layout/_getitem_center_supervised_emg",
  "config": "config/experiment/fusion/fusion_rn18_s_center_16ch_wl7790.yaml",
  "checkpoint_hparams": "fusion_checkpoint_hparams.json",
  "training_records": "no original training statistics content recovered; current config is not execution provenance"
}

配置、代码、checkpoint hparams与原训练记录分别保存。[额外时间戳与配置核查](input_support_supplement.json)区分名义2kHz与非均匀到达时间；不据此重采样。strict load证明当前模型参数结构匹配，不能证明原训练stats文件内容一致。无已证实处理错误，保留发布归一化，未按MAE更换尺度。

# 单参与者校准：结果快照

实验日期：2026-10-05；整理日期：2026-10-21（Asia/Tokyo）。

## 报告与关键证据

- [完整报告](report.md)：wmh / episode_000020 / 右手手指20维的预览录制内离线校准。
- [七条件比较](primary_comparison.md) · [CSV](primary_comparison.csv)。
- [同池监督比较](matched_comparison.md) · [CSV](matched_comparison.csv)。
- [所有 Head seed](all_seeds.md) · [逐 seed 指标](per_seed_metrics.csv)。
- [细分指标](detailed_metrics.csv) · [teacher 质量](teacher_quality.csv) · [参考变化](reference_variation.csv)。
- [覆盖说明](coverage.md) · [覆盖 JSON](coverage.json) · [动作覆盖](gesture_coverage.csv)。
- [RGB 区间覆盖](rgb_interval_coverage.csv) · [RGB 动作覆盖](rgb_gesture_coverage.csv)。
- [逐关节图](per_joint.png) · [逐动作图](per_gesture.png) · [时序图](time_traces.png)。
- [原始设置与执行身份](settings.json) · [历史验证记录](verification.json) · [历史复算检查](verification_recalculation.json)。

这轮检验跨动作迁移，不等同于后续的同动作后次执行协议。当前 Head 未改善总体评估误差；不能从单参与者录制推断未见用户上的效果。完整解释边界见报告；后续判断优先参考 [10 月 14 日诊断](../2026-10-14_research_diagnostic/report.md)。

## 来源与复现

正式来源为本地 `myself/outputs/single_subject_calibration_suite/2026-10-05_wmh_right_suite_full_02/`。所选文件按原始字节复制，映射、大小和 SHA-256 见 [snapshot_manifest.json](snapshot_manifest.json)。原设置与历史验证中的旧路径保留；本地旧名映射位于 `myself/outputs/run_directory_renames.json`。

协议与命令见 [校准套件说明](../../docs/single_subject_calibration_suite.md)，入口为 [single_subject_calibration_suite.py](../../scripts/analysis/single_subject_calibration_suite.py)。复现应使用新的 run-dir，原始代码身份以设置中的哈希为准；导出时的 checkout commit 不是实验执行版本。

未收录逐窗口预测、manifest/特征/teacher 缓存、拟合参数、完整日志、crop 联系表及 Word/PDF 副本。完整预测级复算和恢复运行需要本地原始产物。这里的验证记录来自原实验，整理快照时没有重新运行实验或测试。

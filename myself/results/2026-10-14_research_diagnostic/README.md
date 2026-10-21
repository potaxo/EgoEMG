# 重复动作校准与融合诊断：结果快照

实验日期：2026-10-14；整理日期：2026-10-21（Asia/Tokyo）。

## 报告与关键证据

- [完整报告](report.md)：两位参与者的重复动作校准、一位参与者的共同样本模态比较及三 seed 结构试验。
- [重复动作逐类指标](repeated_gesture_class_summary.csv) · [wmh 汇总](evidence/wmh/summary.csv) · [zbk 汇总](evidence/zbk/summary.csv)。
- [共同样本逐动作指标](common_gesture_summary.csv) · [模态与消融指标](evidence/common/metrics.csv) · [覆盖汇总与原始执行身份](evidence/common/verification.json)。
- [聚合试验配对汇总](pilot_paired_summary.csv) · [逐动作指标](pilot_gesture_summary.csv) · [逐 seed 指标](evidence/pilot/metrics.csv)。
- [试验 EMG 依赖检查](evidence/pilot/emg_dependence.csv) · [训练曲线数据](evidence/pilot/training.csv)。
- [训练资源](evidence/pilot/resources.json) · [逐模型计算预算](evidence/pilot/isolated_budget.json)。
- [历史独立复算](independent_recalculation.json) · [历史验证记录](validation_checks.json) · [原始来源哈希](source_manifest.json)。
- 原始执行设置：[wmh](evidence/wmh/settings.json)、[zbk](evidence/zbk/settings.json)、[聚合试验](evidence/pilot/settings.json)。共同模态的执行哈希见上述 `verification.json`。

报告使用的四幅图随附于 `figures/`，可直接在 GitHub 报告中阅读。当前证据支持暂停本轮 Head 设置，优先检查融合是否使用 EMG；视觉引导聚合本轮未获得实际增益。结论限定于预览录制、右手手指20维及本轮训练预算。

## 正式来源

| 内容 | 原始本地路径 |
| --- | --- |
| 报告、复算表与图 | `myself/outputs/2026-10-14_research_diagnostic_report/` |
| wmh 重复动作 | `myself/outputs/2026-10-14_repeated_gesture_wmh_02/` |
| zbk 重复动作 | `myself/outputs/2026-10-14_repeated_gesture_zbk_02/` |
| 共同样本模态比较 | `myself/outputs/2026-10-14_common_modality_wmh_03/` |
| 视觉引导聚合 | `myself/outputs/2026-10-14_visual_guided_pool_wmh_01/` |

不混入早期 `*_01` 校准或失败的共同模态 `*_01`/`*_02` 数值。选定文件按原始字节复制；[snapshot_manifest.json](snapshot_manifest.json) 提供快照到原文件的映射与 SHA-256。`source_manifest.json` 是原报告的历史来源记录，其中部分文件继续只在本地保留。

## 复现与边界

运行命令、脚本入口与有效 run 说明见 [研究诊断工作流](../../docs/research_diagnostic_2026-10-14.md)。报告构建入口为 [build_research_diagnostic_report.py](../../scripts/analysis/build_research_diagnostic_report.py)，从本地完整产物和 D 盘数据构建原始报告；本次仅导出快照，没有重新运行该脚本或实验。

未收录逐窗口预测/特征、逐窗口 RGB 映射、训练参数、checkpoint、完整日志或数据输入。汇总文件可在网页核对，但完整预测级复算仍需本地产物。原始设置与验证记录中的执行哈希保留；导出时的 checkout commit 不代表实验执行版本。历史测试通过记录也不表示本次重新跑过测试。

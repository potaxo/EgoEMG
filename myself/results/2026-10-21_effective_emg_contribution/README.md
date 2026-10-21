# 2026-10-21 有效 EMG 贡献验证

正式训练10月21日完成；报告整理、均匀时间选窗追踪补充和最终导出为10月22日。

[中文报告](report.md) · [输入核查](input_audit.md) · [主表](main_metrics.csv) · [复算验证](validation.json)

有效运行：`myself/outputs/2026-10-21_wmh_right_effective_emg_full_01`；正式10epochs、seeds42/43/44。真实数据smoke另保留在outputs，本快照只收录正式运行精选证据。历史中心与协议经核验复用，所有本次数值为新执行。完整逐窗口预测、donor映射、权重、日志、脚本副本留在ignored outputs；没有输入或批量特征副本。

复现：[工作流说明](../../docs/effective_emg_contribution_2026-10-21.md)。`snapshot_manifest.json`逐项记录来源和SHA-256；settings区分执行版本与导出日期。所有快照文件可独立在GitHub阅读；预测级复算仍需本地outputs。当前未提交、未推送。

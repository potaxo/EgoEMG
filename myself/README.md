# 个人研究导航

基于 EgoEMG 发布资产和 Preview，研究个体校准与 EMG–视觉增量。

## 当前证据与下一步

- **2026-10-29 固定结构套件已完成并独立复算**：七家族×三条件×三 seed，加锁后复核 zbk/wsj，再将 R0 与选中的 R2 在 T_all 重训；合计81个逻辑单元（69实际优化、12解析锚点），有效 batch128 经用户授权。详见[英文报告](results/2026-10-29_emg_innovation/report.md)。
- **可解码性与视觉增量分开判断**：正确 EMG 的姿态/局部变化/视觉残差小探针未超越各自简单基线；R0→R1 没有一致总体收益，不能据此断言 EMG 没有任务信息。
- **局部正证据**：wsj 的 R3（参考差分＋姿态辅助）相对视觉锚点改善约0.300°，同模型错配平均恶化0.034°；但仅216中心、3执行、全为steady，大部分改善在错配后仍保留。wmh 的 R6 仅微降且错配未恶化；没有跨录制一致的正确 EMG 净收益。
- **动态监督**：额外动态辅助在 wmh/zbk 的 transition 收益方向相反，wsj 无此覆盖。T_val 选中 R2 不代表它优于视觉锚点；T_all 也未改善其 wmh 表现。
- **下一步**：优先在新录制复核 R3 的小部分正确 EMG 特异性，补足 transition 和执行覆盖，并继续追溯原训练归一化字节。暂不根据当前局部证据加入 gate、lag 或大编码器。

历史上，Head 个体校准在两人的后次执行上恶化；Affine 保留为对照。10月21日的小幅融合改善与本轮使用不同视觉锚点和训练协议，不能直接归因比较。旧报告中限定结构实验范围的表述仅属于相应历史轮次。

**范围**：预览包、右手手指20维；不代表完整数据集、跨参与者或因果实时性能。详见各报告。

## 从哪里开始读

| 内容 | 入口 |
| --- | --- |
| 最新证据：手指查询、参考差分、姿态/动态辅助与完整固定矩阵 | [2026-10-29 报告](results/2026-10-29_emg_innovation/report.md) · [精选快照](results/2026-10-29_emg_innovation/README.md) |
| 新套件执行、恢复、独立复算与导出 | [EMG innovation 工作流](docs/emg_innovation_suite.md) |
| 历史证据：有效 EMG 贡献、匹配训练与跨录制冻结复核 | [2026-10-21 报告](results/2026-10-21_effective_emg_contribution/report.md) |
| 本轮输入核查与复现命令 | [有效 EMG 工作流](docs/effective_emg_contribution_2026-10-21.md) |
| 历史证据：重复动作校准、共同样本融合诊断与结构试验 | [2026-10-14 报告](results/2026-10-14_research_diagnostic/report.md) |
| 早期证据：单参与者跨动作校准与 RGB teacher | [2026-10-05 报告](results/2026-10-05_single_subject_calibration/report.md) |
| 全部精选结果、证据来源及适用范围 | [结果索引](results/README.md) |
| 重复动作与融合实验的运行说明 | [研究诊断工作流](docs/research_diagnostic_2026-10-14.md) |
| 单参与者校准协议与运行说明 | [校准套件](docs/single_subject_calibration_suite.md) |
| 最小校准流程检查 | [Smoke 工作流](docs/preview_calibration_smoke.md) |
| 分析与报告构建代码 | [scripts/analysis/](scripts/analysis/) |
| 个人工作流测试 | [egoemg/tests/](egoemg/tests/) |
| 本地环境、数据存储与结果整理规则 | [AGENTS.md](../AGENTS.md) |

## 目录分工

- `docs/`：计划、协议与运行说明。
- `results/`：可提交 Git 的报告、关键 CSV/JSON 与图片。
- `outputs/`：本地预测、缓存、权重与日志，继续忽略。
- `scripts/`、`egoemg/tests/`、`config/`：个人脚本、测试与配置。

网页版讨论从本文件和对应报告开始；文件须先提交并推送。预测级复算与重训练仍需本地数据及完整产物。

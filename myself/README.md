# 个人研究导航

基于 EgoEMG 发布模型和预览数据，研究个体校准与 EMG–视觉融合。

## 研究方向与当前判断

1. **个体校准与视觉监督校准**：当前 Head 在两人的后次执行上均恶化，暂缓作为主线；Affine 保留为对照。
2. **融合有效性**：发布融合优于独立视觉模型，但 EMG 置零几乎不影响误差；视觉引导时间聚合未获稳定增益。

**本轮新增证据（2026-10-21）**：三 seed 匹配训练中，wmh 的对齐组比常量组改善 0.006275°，但错配输入未消除收益；跨录制 zbk 反而恶化 0.018039°。尚未验证对齐 EMG 的可重复特有贡献。发布模型的变化在池化之后明显减弱；legacy 归一化实际命中已确认，原训练统计字节仍未知。

**下一步**：优先追溯原训练统计资产，比较同一窗口的训练/当前输入；确认不一致后再修个人输入流程并原样重跑固定对照。已有评估池按开发证据解释，不开展自动结构或超参搜索。

**范围**：预览包、右手手指20维；不代表完整数据集、跨参与者或因果实时性能。详见各报告。

## 从哪里开始读

| 内容 | 入口 |
| --- | --- |
| 最新证据：有效 EMG 贡献、匹配训练与跨录制冻结复核 | [2026-10-21 报告](results/2026-10-21_effective_emg_contribution/report.md) |
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

# 个人研究导航

基于 EgoEMG 发布模型和预览数据，研究个体校准与 EMG–视觉融合。

## 研究方向与当前判断

1. **个体校准与视觉监督校准**：当前 Head 在两人的后次执行上均恶化，暂缓作为主线；Affine 保留为对照。
2. **融合有效性**：发布融合优于独立视觉模型，但 EMG 置零几乎不影响误差；视觉引导时间聚合未获稳定增益。

**下一步**：核对融合归一化、信号布局和训练资产，检查 EMG 是否被可靠使用。已观察样本作为开发集，新参与者或录制留作测试。

**范围**：预览包、右手手指20维；不代表完整数据集、跨参与者或因果实时性能。详见各报告。

## 从哪里开始读

| 内容 | 入口 |
| --- | --- |
| 最新证据：重复动作校准、共同样本融合诊断与结构试验 | [2026-10-14 报告](results/2026-10-14_research_diagnostic/report.md) |
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

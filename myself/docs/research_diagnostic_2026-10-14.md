# 重复动作校准与融合验证（2026-10-14）

可在 GitHub 阅读的[最终报告](../results/2026-10-14_research_diagnostic/report.md)
及[关键表格、图与来源说明](../results/2026-10-14_research_diagnostic/README.md)
位于 `myself/results/2026-10-14_research_diagnostic/`。
报告包含协议、逐动作比较、共同样本模态比较、三 seed 结构实验、
EMG 置零检查、计算预算与解释边界。原始报告与复算产物仍保留在本地
`myself/outputs/2026-10-14_research_diagnostic_report/`。

有效运行目录：

| 实验 | `myself/outputs/` 下的目录 |
| --- | --- |
| wmh 第一次→第二次同动作校准 | `2026-10-14_repeated_gesture_wmh_02` |
| zbk 独立复核 | `2026-10-14_repeated_gesture_zbk_02` |
| 共同样本模态与原融合消融 | `2026-10-14_common_modality_wmh_03` |
| 冻结编码器的视觉指导时间聚合 | `2026-10-14_visual_guided_pool_wmh_01` |

校准初版未显式关闭 cuDNN TF32，保留历史记录后以 `*_02` 重跑。
共同模态 `*_01` 在默认 crop 探测处失败；`*_02` 在严格数值等价验证处失败；
统一关闭 TF32 后的 `*_03` 才是有效结果。不要混用初版与最终数值。

主要结论：当前 Head 设置在两位参与者上均能拟合第一段，但恶化后次执行；
Affine 只在 zbk 上有明确局部收益。发布融合在共同样本上优于独立视觉模型，
但将 EMG 置零几乎不影响总体误差，必须区分视觉路径增强与 EMG 互补性。
视觉指导时间聚合试验的配对结果与资源记录以最终报告及原始 CSV 为准。

各脚本均从仓库根目录解析模型和资产，从 D 盘直接读取 dataset，
只在 `myself/outputs/` 下创建新 run-dir，不会覆盖已有运行：

```bash
cd /home/potaxo/research/egoemg/EgoEMG
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="$PWD/myself/.cache/matplotlib"
PY=/home/potaxo/miniconda3/envs/egoemg/bin/python
$PY myself/scripts/analysis/repeated_gesture_diagnostic.py --episode episode_000020 --run-dir myself/outputs/NEW_wmh_repeat
$PY myself/scripts/analysis/repeated_gesture_diagnostic.py --episode episode_000008 --run-dir myself/outputs/NEW_zbk_repeat
$PY myself/scripts/analysis/common_modality_evaluation.py --run-dir myself/outputs/NEW_common
$PY myself/scripts/analysis/visual_guided_pool_pilot.py --run-dir myself/outputs/NEW_pool_pilot --epochs 10
$PY myself/scripts/analysis/verify_fusion_pilot_and_budget.py --pilot myself/outputs/NEW_pool_pilot
```

报告构建器 `myself/scripts/analysis/build_research_diagnostic_report.py`
固定读取上表中的有效运行，重算保存的预测，生成 Markdown、CSV、JSON 和图片。

针对性测试：

```bash
PYTHONDONTWRITEBYTECODE=1 /home/potaxo/miniconda3/envs/egoemg/bin/python -m pytest \
  myself/egoemg/tests/test_visual_guided_pool_pilot.py \
  myself/egoemg/tests/test_repeated_gesture_diagnostic.py \
  myself/egoemg/tests/test_single_subject_calibration_suite.py \
  egoemg/tests/test_mid_fusion_forward_modes.py \
  egoemg/tests/test_mid_fusion_vision_lock.py \
  -q -o cache_dir="$PWD/myself/.pytest_cache" \
  --basetemp="$PWD/myself/.cache/research_verification_pytest"
```

本轮只评预览包右手手指 20 维。重复动作窗口虽独立分隔，同池相邻窗口仍重叠；
原预处理是全录制非因果滤波。checkpoint 对参与者的训练暴露未完全追溯。
这些结果不能作为跨参与者、双手44维或因果实时部署的最终基准。

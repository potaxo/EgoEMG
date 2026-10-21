# 有效 EMG 贡献验证工作流（2026-10-21）

本轮限定发布融合架构、固定编码器与局部融合容量。主要量为
`G = MAE(B0常量) - MAE(B1对齐)`；另比较B2错配训练与B1干预。
不重复Head校准，不自动开展结构/超参搜索。

结果入口：[中文报告](../results/2026-10-21_effective_emg_contribution/report.md)、
[精选结果说明](../results/2026-10-21_effective_emg_contribution/README.md)。
完整运行：`myself/outputs/2026-10-21_wmh_right_effective_emg_full_01/`。

## 复现

沿用当前工作树；所有输入直接读取D盘，所有新产物在myself。
下方新目录必须不存在。模型、统计、crop、episode、历史清单均显式传入；
不改写历史脚本中的固定wmh/SOURCE。

```bash
cd /home/potaxo/research/egoemg/EgoEMG
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="$PWD/myself/.cache/matplotlib"
PY=/home/potaxo/miniconda3/envs/egoemg/bin/python
$PY myself/scripts/analysis/effective_emg_contribution.py \
  --run-dir myself/outputs/NEW_effective_emg \
  --episode episode_000020 --review-episode episode_000008 \
  --memmap-dir /mnt/d/egoemg/data/dataset_egoemg_preview/data/memmap_data \
  --crops-dir /mnt/d/egoemg/data/dataset_egoemg_preview/data/pre-crop_webcam_videoframes \
  --checkpoint checkpoints/fusion_resnet18_emgfusion_center.ckpt \
  --vision-checkpoint checkpoints/vision_resnet18.ckpt \
  --norm-stats assets/per_dataset_norm_stats.json \
  --history myself/outputs/single_subject_calibration_suite/2026-10-05_wmh_right_suite_full_02 \
  --common-history myself/outputs/2026-10-14_common_modality_wmh_03 \
  --epochs 10 --seeds 42 43 44
```

首次先另选run-dir并追加`--smoke`：完成相同资格扫描，在真实数据上训练两批，
每录制评估32个中心；不是正式结果。smoke_01因zbk空训练池的统计序列化失败；
smoke_02修正后通过，均保留执行副本与日志。该修正属于个人流程健壮性，未改变发布输入。

历史manifest可用时核验completion marker中的源文件SHA、重读标签、按原协议
重建中心并报告集合差异；缺失时在新目录重建，绝不从其他数据替换。
无历史选择时不传`--history`；`--review-history`支持另一录制同协议manifest。
每个episode使用真实metadata边界与分区，不能把其他录制的manifest传入。
历史共同集只用于比较中心差异，不作为本次预测。

## 预测级独立复算

正式run的`complete.json`通过才可复算；脚本验证记录的全部执行产物哈希。
此阶段不加载模型、不读取原数据、不调用训练MAE函数，使用NumPy float64：

```bash
$PY myself/scripts/analysis/summarize_effective_emg.py \
  --run-dir myself/outputs/NEW_effective_emg
# 同时导出精选结果（目标目录必须不存在）：
$PY myself/scripts/analysis/summarize_effective_emg.py \
  --run-dir myself/outputs/NEW_effective_emg \
  --snapshot myself/results/NEW_effective_emg
```

在正式运行完成后，还可重跑只读补充核查与完整产物验证：

```bash
$PY myself/scripts/analysis/supplement_effective_emg_inputs.py --run-dir myself/outputs/NEW_effective_emg
$PY myself/scripts/analysis/verify_effective_emg.py --run-dir myself/outputs/NEW_effective_emg
# 导出后验证源文件/快照SHA、链接与ignore规则：
$PY myself/scripts/analysis/verify_effective_emg.py \
  --run-dir myself/outputs/NEW_effective_emg \
  --snapshot myself/results/NEW_effective_emg
```

补充核查区分名义2kHz与非均匀逐行时间戳，并逐项比对checkpoint与当前共享配置。
验证器独立检查全部训练donor双射、池隔离、epoch顺序、执行脚本副本及资产哈希；
从预测直接重算G，并将各分组按有效角度数加权还原总MAE。
本轮额外核查在正式训练期间执行，执行源码与指令单独保存，未改训练设置。

`analysis/`包含主表、按动作/原split/连续时间块/动作执行的汇总、配对量、图及中文报告。
正式预测在`episode_*/released.npz`、`vision.npz`、`trained_seed*.npz`；包含全体条件共享
的中心、参考、掩码、图像frame索引，以及预测弧度数组。

`T_all_selection.npz`与`E_selection.npz`保存中心和ID；`*_donors.csv/npz`对应F4的三个固定
映射。训练B2每epoch的donor下标保存在`train_donors_seed*_epoch*.npz`，以该录制
`T_all_selection.npz`的顺序解码；同epoch三组共用顺序保存在`order_seed*_epoch*.npz`。
F1每个recipient使用自身完整EMG窗口；F2/F3使用固定常量；F5反转自身完整窗口。
RGB和reference始终来自recipient，donor不提供图像或目标。

## 针对性检查

```bash
$PY -m pytest \
  myself/egoemg/tests/test_effective_emg_contribution.py \
  myself/egoemg/tests/test_visual_guided_pool_pilot.py \
  egoemg/tests/test_mid_fusion_forward_modes.py \
  egoemg/tests/test_mid_fusion_vision_lock.py \
  -q -o cache_dir="$PWD/myself/.pytest_cache" \
  --basetemp="$PWD/myself/.cache/effective_emg_pytest"
```

合成测试检查中心/完整支持、同split双射、非重叠、不可行时明确失败、随机复现、常量
填充、实际loader的legacy回退、冻结BN、训练复现、参数重载和独立MAE。
正式运行另检查全部真实样本标签一致性、发布forward等价、冻结state摘要不变、
重载预测一致性及训练/评估完整支持隔离。

## 解释边界

固定设置：Adam1e-4、weight_decay1e-4、batch32、10epochs、seeds42/43/44；
训练前20维MAE，最后epoch评估，FP32且两种TF32关闭。无新交互、门控或cross-attention。
B0/B1/B2共享冻结eval/no_grad的featurizer/decoder/vision backbone/head，分别训练
原池化/vision projection/fusion projection/residual head，初值和配对dropout相同。

历史12,000行支持控制中心选择；实际融合7,790行。资格在评分前固定。
30秒连续中心块在原split边界重启；块边界附近窗口仍可能重叠。
bootstrap使用块或动作执行，不将窗口数当独立重复；seed SD与块变动分开。
checkpoint参与者暴露、原训练stats字节/哈希尚未知。当前不按MAE挑选归一化。

运行失败时先读`failure.json`和`run.log`。保留失败目录，修复后使用新run-dir从上述命令
恢复；没有“部分训练当完成”或静默补足历史样本数的恢复逻辑。

## 交付前诊断修正

正式9模型训练与主预测于10月21日完成。最初路径追踪使用合格中心序列的均匀下标；
10月22日改为24个均匀名义时间目标的最近合格中心并去重，补做未更新发布模型的同组干预。
原追踪保留在正式run根目录；最终报告使用`uniform_time_trace/`，其执行源码和哈希独立保存。
当前主工作流已采用修正后的时间选点。该修正不改变任何训练、评估池或主指标。

原run的补充命令（新建子目录，不覆盖已有诊断）：

```bash
$PY myself/scripts/analysis/trace_effective_emg_uniform_time.py --run-dir myself/outputs/NEW_effective_emg
```

若已有`uniform_time_trace/`，直接复用其已保存证据；无需重复运行。
报告构建器优先使用此补充目录。新增测试检查时间选点不会误用序列下标。

#!/usr/bin/env python3
"""Independent float64 recomputation from saved predictions; no model or data access."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

sys.dont_write_bytecode=True
REPO=Path(__file__).resolve().parents[3]
os.environ.setdefault('MPLCONFIGDIR',str(REPO/'myself/.cache/matplotlib'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def write_json(path,obj):path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def csv_write(path,rows):
    if not rows:return
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def score(pred,ref,mask):
    # Deliberately separate from training's angular_mae helper.
    error=np.abs(np.asarray(pred[:,:20],dtype=np.float64)-np.asarray(ref[:,:20],dtype=np.float64))
    valid=np.asarray(mask[:,:20],bool)
    assert np.isfinite(error[valid]).all()
    return float(np.sum(error[valid],dtype=np.float64)/np.count_nonzero(valid)*180/np.pi)


def table(headers,rows):
    return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+''.join('| '+' | '.join(str(v) for v in row)+' |\n' for row in rows)


def load_predictions(run):
    data={};metrics=[];checks=[]
    primary=json.loads((run/'settings.json').read_text())['args']['episode']
    for epdir in sorted(run.glob('episode_*'),key=lambda p:(p.name!=primary,p.name)):
        if not (epdir/'released.npz').exists():continue
        rel=np.load(epdir/'released.npz');vis=np.load(epdir/'vision.npz')
        for k in ('centers','reference','mask','frame'):
            np.testing.assert_array_equal(rel[k],vis[k])
        records={'released':{k:rel[k] for k in rel.files if k.startswith('released_')},'vision':{'vision_normal':vis['vision']}}
        selections={'released':rel,'vision':vis}
        for path in sorted(epdir.glob('trained_seed*.npz')):
            seed=path.stem.removeprefix('trained_seed');arr=np.load(path)
            for k in ('centers','reference','mask','frame'):
                np.testing.assert_array_equal(rel[k],arr[k])
            records[seed]={k:arr[k] for k in arr.files if k.startswith(('B0_','B1_','B2_'))};selections[seed]=arr
        data[epdir.name]=(rel,records)
        for seed,preds in records.items():
            sel=selections[seed]
            anchor=preds.get('B1_normal',preds.get('released_normal',records['released']['released_normal']))
            base=score(preds['B0_constant'],sel['reference'],sel['mask']) if 'B0_constant' in preds else None
            for key,pred in preds.items():
                for grouping in ('all','split','gesture','block','execution'):
                    values=['all'] if grouping=='all' else np.unique(sel[grouping])
                    for value in values:
                        ix=np.ones(len(pred),bool) if grouping=='all' else sel[grouping]==value
                        mae=score(pred[ix],sel['reference'][ix],sel['mask'][ix])
                        b=score(preds['B0_constant'][ix],sel['reference'][ix],sel['mask'][ix]) if base is not None else None
                        metrics.append(dict(episode=epdir.name,participant=json.loads((epdir/'selection_audit.json').read_text())['participant'],
                            seed=seed,condition=key,grouping=grouping,label=str(value),centers=int(ix.sum()),valid_angles=int(sel['mask'][ix,:20].sum()),
                            mae_deg=mae,gain_vs_B0_deg=b-mae if b is not None else None,
                            gain_vs_B0_percent=100*(b-mae)/b if b is not None else None,
                            prediction_change_vs_B1_normal_or_released_deg=float(np.mean(np.abs(pred[ix,:20].astype(np.float64)-anchor[ix,:20]))*180/np.pi)))
            checks.append(dict(episode=epdir.name,seed=seed,centers_equal=True,reference_equal=True,mask_equal=True,frame_equal=True))
    return data,metrics,checks


def paired_statistics(data):
    summaries=[];blocks=[]
    for ep,(sel,records) in data.items():
        seeds=[s for s in records if s.isdigit()]
        for contrast,a,b in [('G_B0_minus_B1','B0_constant','B1_normal'),('B2_minus_B1','B2_mismatch_101','B1_normal'),('B1_wrong_minus_normal','B1_mismatch_101','B1_normal')]:
            overall=[]
            for seed in seeds:
                p=records[seed]
                overall.append(score(p[a],sel['reference'],sel['mask'])-score(p[b],sel['reference'],sel['mask']))
            for grouping in ('block','execution','gesture'):
                vals=[]
                for label in np.unique(sel[grouping]):
                    ix=sel[grouping]==label;gains=[]
                    for seed in seeds:
                        p=records[seed];gains.append(score(p[a][ix],sel['reference'][ix],sel['mask'][ix])-score(p[b][ix],sel['reference'][ix],sel['mask'][ix]))
                    mean=float(np.mean(gains));vals.append(mean)
                    blocks.append(dict(episode=ep,contrast=contrast,grouping=grouping,label=int(label),centers=int(ix.sum()),seed_mean_gain_deg=mean,
                        **{f'seed{s}':float(g) for s,g in zip(seeds,gains)}))
                values=np.asarray(vals);rng=np.random.default_rng(20261021)
                boot=np.mean(rng.choice(values,size=(10000,len(values)),replace=True),axis=1)
                summaries.append(dict(episode=ep,contrast=contrast,grouping=grouping,units=len(values),
                    window_weighted_seed_mean_deg=float(np.mean(overall)),seed_sd_deg=float(np.std(overall,ddof=1)) if len(overall)>1 else 0.,
                    equal_unit_mean_deg=float(values.mean()),unit_sd_deg=float(values.std(ddof=1)) if len(values)>1 else 0.,
                    descriptive_bootstrap_low_deg=float(np.quantile(boot,.025)),descriptive_bootstrap_high_deg=float(np.quantile(boot,.975)),
                    positive_units=int((values>0).sum())))
    return summaries,blocks


def plots(run,out,metrics,stats,blocks):
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    allrows=[r for r in metrics if r['grouping']=='all']
    eps=list(dict.fromkeys(r['episode'] for r in allrows))
    fig,axes=plt.subplots(1,len(eps),figsize=(11,4),squeeze=False)
    colors=['#9ca3af','#167D9A','#BB7751']
    for ax,ep in zip(axes[0],eps):
        seeds=sorted(set(r['seed'] for r in allrows if r['episode']==ep and r['seed'].isdigit()))
        for j,(condition,label) in enumerate([('B0_constant','B0 constant'),('B1_normal','B1 aligned'),('B2_mismatch_101','B2 mismatch')]):
            ys=[next(r['mae_deg'] for r in allrows if r['episode']==ep and r['seed']==seed and r['condition']==condition) for seed in seeds]
            ax.scatter(np.arange(len(seeds))+(j-1)*.12,ys,c=colors[j],label=label,s=40)
            ax.axhline(np.mean(ys),color=colors[j],alpha=.4,lw=1)
        ax.set_xticks(np.arange(len(seeds)),seeds);ax.set_xlabel('Training seed');ax.set_ylabel('Finger MAE (degrees)');ax.set_title(ep+' / fixed encoders')
        ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(out/'figures/matched_training.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,ep in zip(axes,eps):
        rows=[r for r in allrows if r['episode']==ep and r['seed']=='released' and r['condition']!='released_visual_branch']
        names=[r['condition'].removeprefix('released_') for r in rows]
        normal=next(r['mae_deg'] for r in rows if r['condition']=='released_normal')
        ax.bar(np.arange(len(rows)),[r['mae_deg']-normal for r in rows],color='#167D9A',alpha=.7,label='MAE change')
        ax.plot(np.arange(len(rows)),[r['prediction_change_vs_B1_normal_or_released_deg'] for r in rows],'o-',color='#BB7751',label='Prediction change')
        ax.set_xticks(np.arange(len(rows)),names,rotation=40,ha='right');ax.set_ylabel('Degrees');ax.set_title(ep+' / released weights');ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(out/'figures/released_interventions.png',dpi=180);plt.close(fig)
    trace_root=run/'uniform_time_trace' if (run/'uniform_time_trace/path_trace.json').exists() else run
    trace=json.loads((trace_root/'path_trace.json').read_text())['stages'];stages=['input','featurizer','decoder','temporal_pool','residual','output']
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for c in ['zero','constant','mismatch_101','reverse']:
        rows=[next(r for r in trace if r['stage']==s and r['condition']==c) for s in stages]
        axes[0].plot(stages,[max(r['relative_l1_change'],1e-12) for r in rows],'o-',label=c)
        axes[1].plot(stages,[max(r['mean_abs_change'],1e-12) for r in rows],'o-',label=c)
    for ax in axes:
        ax.set_yscale('log');ax.tick_params(axis='x',rotation=40);ax.legend(fontsize=8)
    axes[0].set_ylabel('Relative L1 change');axes[1].set_ylabel('Mean absolute change (stage units)')
    axes[1].set_title('Residual / output units: radians')
    fig.tight_layout();fig.savefig(out/'figures/emg_path.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for ax,group in zip(axes,['block','gesture']):
        rows=[r for r in blocks if r['episode']=='episode_000020' and r['grouping']==group and r['contrast']=='G_B0_minus_B1']
        rows=sorted(rows,key=lambda r:r['label'])
        vals=[r['seed_mean_gain_deg'] for r in rows]
        ax.bar(np.arange(len(rows)),vals,color=['#167D9A' if v>0 else '#BB7751' for v in vals]);ax.axhline(0,color='black',lw=.7)
        ticks=np.arange(0,len(rows),max(1,len(rows)//12));ax.set_xticks(ticks,[rows[i]['label'] for i in ticks],rotation=45)
        ax.set_xlabel('Continuous 30s block ID' if group=='block' else 'Gesture class ID');ax.set_ylabel('G = B0 MAE - B1 MAE (degrees)')
        ax.set_title(f'wmh / equal seed mean / {group}')
    fig.tight_layout();fig.savefig(out/'figures/group_gains.png',dpi=180);plt.close(fig)


def report(run,out,data,metrics,stats,blocks):
    allrows=[r for r in metrics if r['grouping']=='all']
    lookup=lambda ep,seed,c:next(r for r in allrows if r['episode']==ep and r['seed']==str(seed) and r['condition']==c)
    main=[];summary=[]
    for ep,(sel,records) in data.items():
        for condition in ('B0_constant','B1_normal','B2_mismatch_101'):
            rows=[r for r in allrows if r['episode']==ep and r['condition']==condition]
            vals=[r['mae_deg'] for r in rows]
            summary.append([ep,condition,len(sel['centers']),f'{np.mean(vals):.6f} ± {np.std(vals,ddof=1):.6f}',
                f'{np.mean([r["gain_vs_B0_deg"] for r in rows]):+.6f}',f'{np.mean([r["gain_vs_B0_percent"] for r in rows]):+.4f}%'])
        for seed in sorted(s for s in records if s.isdigit()):
            for c in ('B0_constant','B1_normal','B2_mismatch_101'):
                r=lookup(ep,seed,c)
                main.append([r['participant'],ep,c,seed,r['centers'],r['valid_angles'],f'{r["mae_deg"]:.6f}',f'{r["gain_vs_B0_deg"]:+.6f}',f'{r["gain_vs_B0_percent"]:+.4f}%',f'{r["prediction_change_vs_B1_normal_or_released_deg"]:.6f}'])
    text='# 有效 EMG 贡献验证：发布权重与匹配局部训练\n\n正式训练日期：2026-10-21；报告整理与均匀时间诊断补充：2026-10-22（Asia/Tokyo）。右手、Preview、20 个手指角度；所有分数单位为度，越低越好。\n\n'
    text+='![匹配训练：逐 seed](figures/matched_training.png)\n\n'
    text+=table(['录制','训练/评估条件','中心数','MAE 均值 ± seed样本SD','相对B0收益 G（°）','相对B0收益'],summary)
    text+='\n'+table(['人','录制','训练/评估','seed','中心','有效角度','MAE','相对B0收益','相对B0收益%','预测变化对B1正常'],main)
    text+='\n## 结论与下一轮最小实验\n\n'
    for ep,(sel,records) in data.items():
        s=next(r for r in stats if r['episode']==ep and r['contrast']=='G_B0_minus_B1' and r['grouping']=='block')
        directions=[lookup(ep,seed,'B1_normal')['gain_vs_B0_deg'] for seed in records if seed.isdigit()]
        text+=f'{ep}：G 均值 {s["window_weighted_seed_mean_deg"]:+.6f}°，seed SD {s["seed_sd_deg"]:.6f}°；{sum(v>0 for v in directions)}/{len(directions)} seed 为正。连续时间块等权收益 {s["equal_unit_mean_deg"]:+.6f}°，描述性 bootstrap 95%区间 [{s["descriptive_bootstrap_low_deg"]:+.6f}, {s["descriptive_bootstrap_high_deg"]:+.6f}]°；正收益块 {s["positive_units"]}/{s["units"]}。\n\n'
    for ep,(sel,records) in data.items():
        seeds=[s for s in records if s.isdigit()]
        b0=float(np.mean([lookup(ep,s,'B0_constant')['mae_deg'] for s in seeds]))
        b1=float(np.mean([lookup(ep,s,'B1_normal')['mae_deg'] for s in seeds]))
        wrong=float(np.mean([lookup(ep,s,'B1_mismatch_'+str(m))['mae_deg'] for s in seeds for m in (101,102,103)]))
        b2=float(np.mean([lookup(ep,s,'B2_mismatch_101')['mae_deg'] for s in seeds]))
        text+=f'{ep}：B2−B1均值 {b2-b1:+.6f}°；把同一B1的输入换成三个错配映射后，平均MAE只改变 {wrong-b1:+.6f}°，相对B0的收益量为 {b0-wrong:+.6f}°。训练组之间的MAE差与“正确对齐输入当下提供的信息”应分开看。\n\n'
    text+='**本轮未获得可重复的“正确对齐EMG带来特有收益”的充分证据。** wmh三个seed的B1均略优于B0，但B1与B2排序不稳定，且对同一B1换错配输入并不消除其相对B0的优势。跨录制结果和分块变动进一步限制了这种微小训练组差异的解释。这个结论针对本轮局部训练设置，不是断言EMG没有预测信息。\n\n'
    text+='只有 B1 同时稳定优于 B0/B2、且错配使其失去收益，才支持有效对齐信息贡献。不能把发布融合优于独立视觉模型当作已控制容量的 EMG 收益。结果只适用于给定权重、冻结编码器和本轮预算。\n\n'
    text+='**建议优先追溯并验证输入归一化的训练资产。** 当前执行确认 filtered_paper 使用 legacy raw 标量，但缺少原训练统计文件字节，尚不能判定发布输入应改成其他尺度。下一轮最小实验：取得原训练 stats 文件及版本/日志，先作同一真实窗口的训练/当前张量逐项比对；若确认不一致，再修复个人加载路径并原样重跑 B0/B1/B2。没有证据前不按评估 MAE 选尺度，不自动开展交互结构或超参搜索。\n\n'
    text+='## 发布模型：固定权重配对干预\n\n![发布模型干预](figures/released_interventions.png)\n\n'
    frozen=[]
    for r in allrows:
        if r['seed'] in ('released','vision'):
            normal=lookup(r['episode'],'released','released_normal')['mae_deg']
            frozen.append([r['episode'],r['condition'],r['centers'],f'{r["mae_deg"]:.6f}',f'{r["mae_deg"]-normal:+.6f}',f'{r["prediction_change_vs_B1_normal_or_released_deg"]:.6f}'])
    text+=table(['录制','条件','中心数','MAE','相对发布正常 ΔMAE','预测平均绝对变化'],frozen)
    text+='\nF1=normal；F2=归一化后全部槽位置零；F3=原始有效通道常量0经同样布局/归一化；F4=三个预先固定双射（101/102/103）；F5=时间反转。F4 保留完整窗口结构与通道关系、每个 donor 恰好一次，完全保留本池经验边际；同人/同手/同录制/同池/同原split，完整支持不重叠。不同执行优先，不依据预测误差选映射。时间反转保留统计及姿态信息，仅为辅助诊断。独立视觉checkpoint和内部视觉分支的训练历史、表征不同，仅作背景。\n\n'
    text+='wmh完整窗口错配的预测变化约0.047°，MAE近乎不变；zbk预测变化约0.43°，明显大于wmh，而MAE变化仍小。因而弱“误差敏感性”在两录制中出现，但不能据此说zbk模型不使用EMG；个体预测变化会在总体MAE中抵消。zbk常量输入还改善约0.115°，显示该输入条件下EMG并未带来净误差收益。\n\n'
    if (run/'donor_summary.csv').exists():
        donorrows=list(csv.DictReader((run/'donor_summary.csv').open()))
        text+='固定映射不同执行/类别覆盖（完整窗口支持均不重叠，最小中心距离≥7790行）：\n\n'+table(['录制','池','映射seed','中心数','不同执行数','不同类别数'],[[r['episode'],r['pool'],r['mapping_seed'],r['centers'],r['different_execution'],r['different_class']] for r in donorrows if r['pool']=='E'])+'\n'
    text+='## B1 输入干预与分组\n\n'
    intervention=[]
    for ep,(sel,records) in data.items():
        for c in ('B1_normal','B1_constant','B1_mismatch_101','B1_mismatch_102','B1_mismatch_103','B1_reverse'):
            rows=[r for r in allrows if r['episode']==ep and r['condition']==c]
            intervention.append([ep,c,f'{np.mean([r["mae_deg"] for r in rows]):.6f}',f'{np.mean([r["prediction_change_vs_B1_normal_or_released_deg"] for r in rows]):.6f}'])
    text+=table(['录制','B1评估输入','三seed均值MAE','预测变化对B1正常'],intervention)
    text+='\n![收益按时间块与动作分布](figures/group_gains.png)\n\n'
    statrows=[[r['episode'],r['contrast'],r['grouping'],r['units'],f'{r["equal_unit_mean_deg"]:+.6f}',f'{r["unit_sd_deg"]:.6f}',f'[{r["descriptive_bootstrap_low_deg"]:+.6f}, {r["descriptive_bootstrap_high_deg"]:+.6f}]'] for r in stats if r['grouping'] in ('block','execution')]
    text+=table(['录制','配对量','统计单位','单位数','等权均值','数据单位SD','描述性95%区间'],statrows)
    text+='\n时间块按每段原 split 内30秒名义行时间定义；动作执行按连续（gesture class, active）定义，不因split切换人为新增。相邻窗口仍重叠，跨块边界可能共享少量输入，因此区间只描述这批录制的块变动，不是独立参与者置信区间。bootstrap先在每块内对seed取均值，再重采样块10000次；seed SD单独报告。按动作、原split、执行和时间块的全部逐seed汇总见 [group_metrics.csv](group_metrics.csv)，逐块配对量见 [paired_units.csv](paired_units.csv)。未把重叠窗口当作独立重复。未开展遮挡/信号质量机制归因。\n\n'
    text+='## 输入与 EMG 路径\n\n![EMG输入至输出的变化](figures/emg_path.png)\n\n详见 [输入核查表](input_audit.md)、[阶段追踪](path_trace.csv) 和 [通道分布](input_channels.csv)。阶段绝对差与相对L1/L2差均保留，残差/输出单位为rad，其他阶段为各自特征单位；不能横向把不同阶段绝对幅值当作同一物理量。选窗仅由训练中心时刻决定：24个均匀名义时间目标，各取最近合格训练中心，去重后实际窗口数见path_trace.json。此诊断补充在训练完成后以未更新发布权重执行，不使用预测误差；初版按合格序列下标均匀抽样的追踪留在完整run根目录，未覆盖。\n\n'
    trace_root=run/'uniform_time_trace' if (run/'uniform_time_trace/path_trace.json').exists() else run
    trace=json.loads((trace_root/'path_trace.json').read_text())['stages']
    pathrows=[r for r in trace if r['condition']=='constant']
    text+=table(['常量干预阶段','基准meanabs','干预meanabs','绝对mean变化','相对L1变化'],[[r['stage'],f"{next(t['mean_abs'] for t in trace if t['condition']=='normal' and t['stage']==r['stage']):.8f}",f"{r['mean_abs']:.8f}",f"{r['mean_abs_change']:.8f}",f"{r['relative_l1_change']:.6f}"] for r in pathrows])+'\n'
    text+='该批次常量干预在池化表征上仍有显著相对变化，而输出变化很小；弱依赖不等于输入未到达模型。变化在池化之后的融合投影与残差head中被大幅压低，支持优先排查已有融合路径对EMG的使用方式。这里未分离两层各自作用，也未证明归一化是唯一原因。\n\n'
    text+='采样率按名义2kHz计算窗口长度；全录制timestamp跨度推算wmh约2000.012Hz、zbk约2000.015Hz，但逐行timestamp_us并非均匀采样时间（中位相邻差1μs，存在相同时间戳与约8.5ms跳变）。加载器使用原行序与中心，不按该时间戳重采样；上游packet到达时间与采样时刻关系仍未完全追溯，不能据此擅自修索引。\n\n'
    text+='局部常量干预若依然留下非零 decoder 特征，不代表残留有效EMG信息：编码器偏置/归一化/位置编码可生成常量特征。残差路径同时接收视觉，残差大也不证明EMG贡献。当前审计未确认单位、索引或布局错误；没有执行依据评估分数挑选的尺度“修复”。legacy缩放可能压低特征差异只是待验证机制，尚不能据此确定病因。\n\n'
    text+='## 样本与训练协议\n\n'
    for ep in data:
        a=json.loads((run/ep/'selection_audit.json').read_text())
        text+=f'{ep}（{a["participant"]}）：候选 {a["candidates"]}，资格通过 {a["eligible"]}；T={a["counts"]["T_all"]}，E={a["counts"]["E"]}。排除计数：`{json.dumps(a["exclusions"],ensure_ascii=False)}`。'
        if a.get('history_used'):text+=' 历史manifest哈希验证通过，标签逐项重读核对，中心按原12,000行/stride200协议复核重建；中心差异见selection_audit。'
        else:text+=' 未复用历史中心，按原12,000行/stride200完整split支持协议重建到本次目录。'
        text+='\n\n'
    text+='所有正式比较共用相同E中心、RGB、参考及掩码；资格在评分前固定，检查标签、全窗口有限性和crop解码。训练与评估完整7790行支持隔离，且沿用历史12000行安全边界。RGB要求映射非负、stale=false、|delta_ms|≤50、uint8 crop可解码和预处理有限。训练窗口资格与评估独立检查。\n\n'
    text+='B0/B1/B2分别复制发布时间池、vision projection、fusion projection及residual head；共享冻结eval/no_grad的featurizer、decoder、视觉backbone及视觉head。每seed三组初值相同、数据顺序相同、配对批次dropout随机数相同。Adam lr=1e-4、weight_decay=1e-4、batch32、10epochs、FP32，matmul/cuDNN TF32关闭。前20维MAE loss；仅评估最后epoch。B2每epoch由seed固定训练池内错配双射，评估使用固定101映射；训练映射不使用评估池。zbk上只冻结复核，不训练或调参；wmh模型直接用于zbk同协议E池。\n\n'
    check=json.loads((run/'training_checks.json').read_text());complete=json.loads((run/'complete.json').read_text())
    text+=f'参数每组 {check["resources"][0]["parameters_per_branch"]:,}。三seed合计训练 {sum(r["training_seconds"] for r in check["resources"]):.1f}s；峰值allocated {max(r["peak_allocated_bytes"] for r in check["resources"])/2**20:.1f}MiB，reserved {max(r["peak_reserved_bytes"] for r in check["resources"])/2**20:.1f}MiB。含资格扫描/多干预评估的完整运行 {complete["wall_seconds"]:.1f}s。输入直接读取D盘，没有复制memmap、crop、视频或批量特征。\n\n'
    text+='## 验证、来源与限制\n\n'
    text+='本次实际执行：真实输入smoke、wmh与zbk发布融合F1–F5、独立视觉背景、wmh三seed×B0/B1/B2训练、每个B1的正常/常量/三个错配/反转、wmh模型对zbk的固定复核；从本次保存预测独立float64重算全部MAE和分组。复用历史仅限wmh中心选择与背景协议，没有复用历史预测作为本次实验数值。\n\n'
    text+='运行检查：原PoolingBranch与发布forward逐批一致；冻结模型完整state_dict（含BN buffers）训练前后SHA相同；每组保存参数重新加载预测最大差为0rad；所有比较的中心/图像/参考/掩码逐项一致。针对性pytest、复算和快照校验见 [validation.json](validation.json)。完整命令与环境见 [settings.json](settings.json)，所有输入/脚本哈希保留。\n\n'
    text+='已有wmh评估池属于开发证据；checkpoint参与者训练暴露尚未完全追溯；zbk只是不同录制上的冻结复核，不能称未见参与者测试。预处理含非因果滤波，窗口重叠；不是官方benchmark，不外推至全部数据、双手44维或在线部署。没有重跑原始数据预处理，也没有恢复缺失的原训练stats内容。当前结果不证明EMG在其他表征/训练预算下没有信息。\n\n'
    text+=f'完整本地运行：`{run.relative_to(REPO)}`。执行HEAD `{json.loads((run/"settings.json").read_text())["head"]}`。本次为未提交、未推送的本地工作；结果导出日期与执行身份分别记录在manifest。\n'
    (out/'report.md').write_text(text)


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--snapshot',type=Path)
    a=p.parse_args();run=a.run_dir.resolve();out=run/'analysis';out.mkdir(exist_ok=True);(out/'figures').mkdir(exist_ok=True)
    complete=json.loads((run/'complete.json').read_text());assert not complete['smoke']
    for name,h in complete['files'].items():assert digest(run/name)==h,(name,'execution artifact changed')
    data,metrics,checks=load_predictions(run);stats,blocks=paired_statistics(data)
    csv_write(out/'main_metrics.csv',[r for r in metrics if r['grouping']=='all']);csv_write(out/'group_metrics.csv',metrics)
    csv_write(out/'paired_summary.csv',stats);csv_write(out/'paired_units.csv',blocks)
    plots(run,out,metrics,stats,blocks)
    for name in ['input_audit.json','path_trace.json','path_trace.csv','input_channels.csv','settings.json','training_checks.json','training_history.csv','resolved_model.yaml','resolved_dataset.json','fusion_checkpoint_hparams.json']:
        shutil.copyfile(run/name,out/name)
    if (run/'uniform_time_trace').exists():
        for name in ('path_trace.json','path_trace.csv','input_channels.csv'):
            shutil.copyfile(run/'uniform_time_trace'/name,out/name)
        shutil.copyfile(run/'uniform_time_trace/execution.json',out/'trace_execution.json')
    audit=json.loads((run/'input_audit.json').read_text());rows=[]
    for status,section in [('已确认','confirmed'),('已修复','fixed'),('未知','unknown')]:
        values=audit[section]
        if isinstance(values,dict):
            for k,v in values.items():rows.append([status,k,json.dumps(v,ensure_ascii=False)])
    if (run/'input_support_supplement.json').exists():
        supplement=json.loads((run/'input_support_supplement.json').read_text())
        rows.append(['已确认','名义采样率与timestamp_us','全录制跨度推算约2000Hz；逐行timestamp_us成批到达且非均匀；加载器不按timestamp重采样，时间块按名义行时间定义。详见input_support_supplement.json。'])
        rows.append(['未知','EMG物理单位',supplement['physical_emg_unit']])
        rows.append(['已确认','checkpoint/current共享配置','规范化emg2pose→egoemg命名空间后，共享字段除视觉初始化文件路径外一致；完整发布fusion state严格覆盖初始化。原始执行源码数值等价未恢复。'])
    (out/'input_audit.md').write_text('# 磁盘字段至模型输入核查\n\n'+table(['状态','项目','证据/实际行为'],rows)+'\n## 证据来源分层\n\n'+json.dumps(audit['evidence_separated'],ensure_ascii=False,indent=2)+'\n\n配置、代码、checkpoint hparams与原训练记录分别保存。[额外时间戳与配置核查](input_support_supplement.json)区分名义2kHz与非均匀到达时间；不据此重采样。strict load证明当前模型参数结构匹配，不能证明原训练stats文件内容一致。无已证实处理错误，保留发布归一化，未按MAE更换尺度。\n')
    validation=dict(independent_prediction_recomputation=True,float64=True,all_execution_hashes_verified=True,paired_input_checks=checks,
        bootstrap_units='30s continuous center blocks or action execution; 10000 draws; seed mean before resampling',
        seed_variability_separate=True,source_predictions_sha256={str(p.relative_to(REPO)):digest(p) for p in run.glob('episode_*/*.npz')},
        completed_prediction_archives=len(checks),pytest='see test_results_final.txt for final 15-test execution; test_results.txt preserves initial 14-test execution',limitations=['not independent subjects','overlapping windows may span block boundaries','original training stats bytes absent'])
    for optional in ('test_results.txt','test_results_final.txt','donor_summary.csv','artifact_verification.json','input_support_supplement.json'):
        if (run/optional).exists():shutil.copyfile(run/optional,out/optional)
    write_json(out/'validation.json',validation)
    write_json(out/'analysis_execution.json',dict(command=sys.argv,script_sha256=digest(Path(__file__)),script=str(Path(__file__).resolve().relative_to(REPO)),time_tokyo=datetime.now(ZoneInfo('Asia/Tokyo')).isoformat(),method='independent NumPy float64; no inference or data inputs'))
    shutil.copyfile(__file__,run/'executed_summarizer.py')
    report(run,out,data,metrics,stats,blocks)
    if a.snapshot:
        target=a.snapshot.resolve();assert target.is_relative_to(REPO/'myself/results');target.mkdir(exist_ok=False)
        exported=[]
        for src in sorted(out.rglob('*')):
            if src.is_file():
                dest=target/src.relative_to(out);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dest)
                exported.append(dict(file=str(dest.relative_to(target)),source=str(src.relative_to(REPO)),sha256=digest(src)))
        # Small selection/protocol audits contain no per-window inputs or predictions.
        for ep in data:
            src=run/ep/'selection_audit.json';dest=target/f'{ep}_selection_audit.json';shutil.copyfile(src,dest)
            exported.append(dict(file=dest.name,source=str(src.relative_to(REPO)),sha256=digest(src)))
        manifest=dict(export_date=datetime.now(ZoneInfo('Asia/Tokyo')).isoformat(),checkout_commit=__import__('subprocess').check_output(['git','rev-parse','HEAD'],text=True).strip(),
            execution_head=json.loads((run/'settings.json').read_text())['head'],execution_source=str(run.relative_to(REPO)),
            note='export date/checkout are not execution identity; settings includes actual scripts and assets hashes',files=exported)
        write_json(target/'snapshot_manifest.json',manifest)
        (target/'README.md').write_text('# 2026-10-21 有效 EMG 贡献验证\n\n[中文报告](report.md) · [输入核查](input_audit.md) · [主表](main_metrics.csv) · [复算验证](validation.json)\n\n有效运行：`'+str(run.relative_to(REPO))+'`；正式10epochs、seeds42/43/44。真实数据smoke另保留在outputs，本快照只收录正式运行精选证据。历史中心与协议经核验复用，所有本次数值为新执行。完整逐窗口预测、donor映射、权重、日志、脚本副本留在ignored outputs；没有输入或批量特征副本。\n\n复现：[工作流说明](../../docs/effective_emg_contribution_2026-10-21.md)。`snapshot_manifest.json`逐项记录来源和SHA-256；settings区分执行版本与导出日期。所有快照文件可独立在GitHub阅读；预测级复算仍需本地outputs。当前未提交、未推送。\n')
    print(json.dumps([r for r in stats if r['grouping']=='block'],ensure_ascii=False,indent=2))

if __name__=='__main__':main()

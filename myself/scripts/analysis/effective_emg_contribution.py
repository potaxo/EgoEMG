#!/usr/bin/env python3
"""Matched EMG contribution experiment; stream designated inputs without staging them."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset

from egoemg.datasets.egoemg_memmap_dataset import EgoEmgMemmapDataset
from myself.scripts.analysis.common_modality_evaluation import configuration
from myself.scripts.analysis.preview_calibration_smoke import open_field, runs, sha256
from myself.scripts.analysis.run_directories import create_run_directory
from myself.scripts.analysis.single_subject_calibration_suite import (
    coverage, csv_write, enumerate_windows, json_write, npz_write, rgb_reason, strict_model,
)
from myself.scripts.analysis.visual_guided_pool_pilot import PoolingBranch, frozen_digest

WINDOW = 7790
HALF = WINDOW // 2
MAPPING_SEEDS = (101, 102, 103)


def configure_precision():
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def make_base(args, episode):
    base = EgoEmgMemmapDataset(
        memmap_dir=args.memmap_dir, window_length=WINDOW, stride=200,
        allowed_episode_ids=[episode], allowed_splits=['train', 'user', 'gesture', 'both'],
        modalities=['emg', 'joint_angles', 'labels'], target_hand='right',
        emg_field_preference='filtered_paper', emg_layout='emg2pose_interpolate16',
        emg2pose_channel_indices=[10, 12, 0, 1, 2, 4, 5, 6], channel_interpolate=False,
        norm_mode='per-dataset', norm_stats_path=str(args.norm_stats), center_target_only=True,
    )
    base.per_episode_crops_dir = str(args.crops_dir)
    base._pec_pid, base._pec_lmdb_cache, base._pec_lmdb_max_open = os.getpid(), {}, 1
    return base


def support_ok(center, origin, stop, split, bad_prefix, valid, reference):
    a, b = int(center) - HALF, int(center) + HALF
    if a < origin or b > stop:
        return 'episode_support'
    sid = split[center]
    if np.any(split[a:b] != sid):
        return 'split_support'
    if bad_prefix[b-origin] != bad_prefix[a-origin]:
        return 'nonfinite_emg_support'
    if not valid[center]:
        return 'invalid_label'
    if not np.isfinite(reference[center]).all():
        return 'nonfinite_label'
    return 'eligible'


def donor_mapping(centers, splits, executions, seed, window=WINDOW):
    """A seeded bijection within each split, with nonoverlapping complete supports.

    Try a different execution first. If that perfect matching is impossible, allow
    same execution but retain nonoverlap. Never use predictions or target values.
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import maximum_bipartite_matching
    rng = np.random.default_rng(seed)
    result = np.full(len(centers), -1, np.int64)
    for sid in np.unique(splits):
        ix = np.flatnonzero(splits == sid)
        c, ex = centers[ix], executions[ix]
        allowed = np.abs(c[:, None] - c[None, :]) >= window
        for different in (True, False):
            ok = allowed & (ex[:, None] != ex[None, :]) if different else allowed
            order = rng.permutation(len(ix))
            # A sparse exact matching retains every donor once: exact empirical marginal.
            match = maximum_bipartite_matching(csr_matrix(ok[:, order]), perm_type='column')
            if (match >= 0).all():
                result[ix] = ix[order[match]]
                break
        if (result[ix] < 0).any():
            raise ValueError(f'No nonoverlapping donor bijection for split={sid}, n={len(ix)}')
    assert np.array_equal(np.sort(result), np.arange(len(result)))
    assert np.all(splits == splits[result])
    assert np.all(np.abs(centers - centers[result]) >= window)
    return result


def constant_input(x, value):
    # Physical zero in all eight live channels, then exactly the published layout/norm.
    return torch.full_like(x, value)


class Centers(Dataset):
    def __init__(self, base, episode, selection):
        self.base, self.episode, self.s = base, episode, selection
        self.ep = base._episode_id.index(episode)

    def __len__(self):
        return len(self.s['centers'])

    def emg(self, i):
        c = int(self.s['centers'][i])
        sample = self.base._getitem_center_supervised_emg(self.ep, c-HALF, c+HALF)
        np.testing.assert_allclose(sample['joint_angles'][:, 0], self.s['reference'][i], atol=1e-6)
        assert bool(sample['label_valid_mask'][0]) and np.isfinite(sample['emg']).all()
        return sample['emg']

    def __getitem__(self, i):
        rgb = self.base._read_episode_crops(self.episode, np.array([self.s['frame'][i]]), 'R')[0]
        img = (rgb.transpose(2, 0, 1).astype(np.float32)-self.base.vision_mean[:, None, None])/self.base.vision_std[:, None, None]
        assert np.isfinite(img).all()
        return self.emg(i), img, self.s['reference'][i, :20], i


def prepare(args, episode, history, run):
    out = run / episode
    out.mkdir()
    base = make_base(args, episode)
    ep = base._episode_id.index(episode)
    participant = base._episode_subject[ep]
    origin, stop = int(base._episode_start_idx[ep]), int(base._episode_end_idx[ep])
    mf = json.loads((args.memmap_dir/'manifest.json').read_text())
    field = lambda k: open_field(args.memmap_dir, mf, k)
    assert np.all(field('episode_index')[origin:stop] == ep)
    assert np.unique(field('subject_id')[origin:stop]).size == 1
    split = field('frame_split_id')
    valid = field('generated_label_valid')[:, 1]
    ref20 = field('generated_joint_angles_right')
    raw = field('emg_right_filtered_paper')
    bad = np.r_[0, np.cumsum(~np.isfinite(raw[origin:stop]).all(1), dtype=np.int64)]
    md = np.load(args.memmap_dir/'metadata.npz')
    names = {int(i): v.decode() for i,v in zip(md['splits_split_id'], md['splits_split'])}
    rebuilt_starts, rebuilt_pools, intervals = enumerate_windows(
        split[origin:stop], valid[origin:stop], raw[origin:stop], ref20[origin:stop], names, 12000, 200)
    rebuilt_centers = rebuilt_starts + origin + 6000
    source_info = {'protocol': 'historical 12000-row complete split support, stride 200; fusion support 7790',
                   'history_requested': str(history) if history else None, 'history_used': False}
    if history and (history/'manifest.npz').is_file():
        marker = json.loads((history/'manifest.complete.json').read_text())
        for name,digest in marker['files'].items():
            assert sha256(history/name) == digest, (name, 'historical hash mismatch')
        old = np.load(history/'manifest.npz')
        centers, pools = old['centers'], old['pools']
        np.testing.assert_allclose(ref20[centers], old['reference'][:, :20], atol=1e-6)
        np.testing.assert_array_equal(old['mask'][:, :20], np.isfinite(ref20[centers]))
        assert np.all(centers >= origin) and np.all(centers < stop)
        source_info.update(history_used=True, history_sha256=sha256(history/'manifest.npz'),
                           historical_identity=marker['identity'],
                           only_historical_centers=np.setdiff1d(centers, rebuilt_centers).tolist(),
                           only_rebuilt_centers=np.setdiff1d(rebuilt_centers, centers).tolist())
    else:
        centers, pools = rebuilt_centers, rebuilt_pools
        source_info.update(rebuilt=True, center_difference='no historical selection available')
    reference = np.column_stack([ref20[centers], np.deg2rad(field('mocap_right_wrist_pitch')[centers]),
                                 np.deg2rad(field('mocap_right_wrist_yaw')[centers])]).astype(np.float32)
    frame = np.asarray(field('image_head_frame_index')[centers])
    stale, delta = field('image_head_stale')[centers], field('image_head_delta_ms')[centers]
    reasons = []
    for i,c in enumerate(centers):
        reason = support_ok(c, origin, stop, split, bad, valid, ref20)
        # Revalidate the longer historical support as well (includes isolation guard).
        if reason == 'eligible' and (c-6000 < origin or c+6000 > stop or np.any(split[c-6000:c+6000] != split[c])):
            reason = 'historical_support'
        if reason == 'eligible':
            reason = rgb_reason(int(frame[i]), bool(stale[i]), float(delta[i]), 50.)
        if reason == 'eligible':
            try:
                rgb = base._read_episode_crops(episode, np.array([frame[i]]), 'R')
                if rgb is None:
                    reason = 'missing_crop'
                elif rgb.shape != (1,256,256,3) or rgb.dtype != np.uint8:
                    reason = 'crop_shape_dtype'
                else:
                    img=(rgb[0].transpose(2,0,1).astype(np.float32)-base.vision_mean[:,None,None])/base.vision_std[:,None,None]
                    if not np.isfinite(img).all(): reason='nonfinite_rgb_preprocess'
            except (OSError, ValueError, TypeError) as exc:
                reason = 'decode_'+type(exc).__name__
        reasons.append(reason)
        if i % 1500 == 0: logging.info('%s eligibility %d/%d',episode,i,len(centers))
    reasons = np.asarray(reasons)
    csv_write(out/'eligibility.csv', [dict(center=int(c), pool=str(p), frame=int(f), reason=str(r)) for c,p,f,r in zip(centers,pools,frame,reasons)])
    keep = reasons == 'eligible'
    # Execution IDs follow continuous (gesture, active), independently of original split.
    gesture_all, active_all = field('label_gesture_class'), field('label_gesture_active')
    g, ac = np.asarray(gesture_all[origin:stop]), np.asarray(active_all[origin:stop])
    execution = np.r_[0, np.cumsum((g[1:] != g[:-1]) | (ac[1:] != ac[:-1]))]
    # Continuous 30-second center blocks, separated at every split boundary.
    block = np.empty(stop-origin, np.int64)
    offset=0
    for a,b,sid in runs(split[origin:stop]):
        block[a:b] = offset + np.arange(b-a)//60000
        offset += (b-a+59999)//60000
    selection = dict(centers=centers[keep], pools=pools[keep], reference=reference[keep],
        mask=np.isfinite(reference[keep]), frame=frame[keep], split=np.asarray(split[centers[keep]]),
        gesture=np.asarray(gesture_all[centers[keep]]), execution=execution[centers[keep]-origin],
        block=block[centers[keep]-origin], ids=np.asarray([f'{episode}:R:{c}' for c in centers[keep]]))
    selection['mask'][:,20:] &= field('mocap_right_wrist_angles_valid')[selection['centers'],None]
    sets={}
    for pool in ('T_all','E'):
        s={k:v[selection['pools']==pool] for k,v in selection.items()}
        npz_write(out/f'{pool}_selection.npz',**s)
        maps={str(seed):donor_mapping(s['centers'],s['split'],s['execution'],seed) for seed in MAPPING_SEEDS}
        npz_write(out/f'{pool}_donors.npz',**maps)
        csv_write(out/f'{pool}_donors.csv', [dict(mapping_seed=seed, recipient=str(s['ids'][i]), donor=str(s['ids'][maps[str(seed)][i]]),
            recipient_center=int(c), donor_center=int(s['centers'][maps[str(seed)][i]]), split=int(s['split'][i]),
            different_execution=bool(s['execution'][i]!=s['execution'][maps[str(seed)][i]]),
            different_class=bool(s['gesture'][i]!=s['gesture'][maps[str(seed)][i]]))
            for seed in MAPPING_SEEDS for i,c in enumerate(s['centers'])])
        sets[pool]=(Centers(base,episode,s),maps)
    tr,ev=sets['T_all'][0].s['centers'],sets['E'][0].s['centers']
    assert not (np.abs(tr[:,None]-ev[None,:]) < WINDOW).any(), 'Train/eval complete support overlap'
    source_info.update(participant=participant, episode=episode, episode_bounds=[origin,stop],
        candidates=len(centers), eligible=int(keep.sum()), exclusions={str(r):int((reasons==r).sum()) for r in np.unique(reasons)},
        counts={p:len(v[0]) for p,v in sets.items()},
        coverage={p:coverage(v[0].s['centers']-HALF,WINDOW,200) for p,v in sets.items()},
        original_intervals=intervals, complete_support_train_eval_isolated=True,
        block_definition='30s nominal row-time center bins split at original split boundaries; windows may overlap block borders',
        donor_definition='same participant/episode/right hand/pool/original split, seeded bijection, complete supports disjoint; different continuous center execution preferred',
        donor_different_execution={p:{seed:float(np.mean(ds.s['execution']!=ds.s['execution'][maps[seed]])) if len(ds) else None for seed in maps} for p,(ds,maps) in sets.items()})
    if episode == args.episode and args.common_history and (args.common_history/'predictions.npz').is_file():
        old=np.load(args.common_history/'predictions.npz')
        source_info['common_history_difference']={'only_old':np.setdiff1d(old['centers'],ev).tolist(), 'only_new':np.setdiff1d(ev,old['centers']).tolist(), 'sha256':sha256(args.common_history/'predictions.npz')}
    json_write(out/'selection_audit.json',source_info)
    logging.info('%s fixed selection %s',episode,source_info['counts'])
    return sets, source_info


def encode(model,x,img):
    with torch.no_grad():
        emg=model.decoder(model.featurizer(x))
        vision=model._extract_vision_features(img)
        pose=model.head_vision(vision)
    return emg,vision,pose


def emg_variant(ds,x,indices,condition,maps,padding):
    if condition == 'normal': return x
    if condition == 'zero': return torch.zeros_like(x)
    if condition == 'constant': return constant_input(x,padding)
    if condition == 'reverse': return x.flip(-1)
    key=condition.removeprefix('mismatch_')
    donor=maps[key][np.asarray(indices)]
    return torch.from_numpy(np.stack([ds.emg(int(i)) for i in donor])).to(x.device)


def evaluate(model,branches,ds,maps,padding,out,seed=None,batch_size=32):
    predictions={}
    if branches is None:
        requests=[('released',c,None) for c in ('normal','zero','constant','mismatch_101','mismatch_102','mismatch_103','reverse')]
    else:
        requests=[('B0','constant',branches['B0']),('B2','mismatch_101',branches['B2'])]+[
            ('B1',c,branches['B1']) for c in ('normal','constant','mismatch_101','mismatch_102','mismatch_103','reverse')]
    released_branch=PoolingBranch(model).cuda().eval() if branches is None else None
    accum={f'{b}_{c}':[] for b,c,_ in requests}
    for branch in (branches or {}).values(): branch.eval()
    with torch.inference_mode():
        for x,img,y,indices in DataLoader(ds,batch_size=batch_size,shuffle=False,num_workers=0):
            x,img=x.cuda(),img.cuda()
            vf=model._extract_vision_features(img); pose=model.head_vision(vf)
            for b,c,branch in requests:
                signal=emg_variant(ds,x,indices,c,maps,padding)
                decoded=model.decoder(model.featurizer(signal))
                pred=(branch if branch is not None else released_branch)(decoded,vf,pose)
                if branches is None and c=='normal':
                    torch.testing.assert_close(pred,model({'emg':signal,'vision_img':img})[...,0],atol=1e-6,rtol=1e-5)
                accum[f'{b}_{c}'].append(pred.cpu().numpy())
            if branches is None:
                predictions.setdefault('released_visual_branch',[]).append(pose.cpu().numpy())
    predictions={k:np.concatenate(v) for k,v in predictions.items()}
    predictions.update({k:np.concatenate(v) for k,v in accum.items()})
    npz_write(out,**ds.s,**predictions)
    logging.info('Saved %s',out)
    return predictions


def time_uniform_indices(centers, count=24):
    targets=np.linspace(centers[0],centers[-1],min(count,len(centers)))
    return np.unique([int(np.argmin(np.abs(centers-t))) for t in targets])


def path_trace(model,ds,maps,padding,out):
    ids=time_uniform_indices(ds.s['centers'])
    batches=[ds[int(i)] for i in ids]
    x=torch.from_numpy(np.stack([v[0] for v in batches])).cuda()
    img=torch.from_numpy(np.stack([v[1] for v in batches])).cuda()
    branch=PoolingBranch(model).cuda().eval()
    def stages(signal):
        f=model.featurizer(signal); d=model.decoder(f); pooled=branch.pool(d)
        vf=model._extract_vision_features(img); pose=model.head_vision(vf)
        residual=branch.head(branch.fusion_proj(torch.cat([pooled,branch.vision_proj(vf)],-1)[...,None]))[...,0]
        return dict(input=signal,featurizer=f,decoder=d,temporal_pool=pooled,residual=residual,output=pose+residual)
    rows=[]; distributions=[]
    with torch.inference_mode():
        normal=stages(x)
        for c in ('normal','zero','constant','mismatch_101','mismatch_102','mismatch_103','reverse'):
            values=stages(emg_variant(ds,x,ids,c,maps,padding))
            for stage,v in values.items():
                ref=normal[stage]; diff=v-ref
                rows.append(dict(condition=c,stage=stage,shape=list(v.shape),mean_abs=float(v.abs().mean()),rms=float(v.square().mean().sqrt()),
                    mean_abs_change=float(diff.abs().mean()),relative_l1_change=float(diff.abs().sum()/ref.abs().sum().clamp_min(1e-15)),
                    relative_l2_change=float(diff.norm()/ref.norm().clamp_min(1e-15)),max_abs_change=float(diff.abs().max())))
        for slot in range(16):
            v=x[:,slot].cpu().numpy()
            distributions.append(dict(slot=slot,live=slot in ds.base._emg2pose_channel_indices,mean=float(v.mean()),std=float(v.std()),
                minimum=float(v.min()),p01=float(np.quantile(v,.01)),median=float(np.median(v)),p99=float(np.quantile(v,.99)),maximum=float(v.max())))
    json_write(out/'path_trace.json',dict(input_tensor_sha256=hashlib.sha256(x.cpu().numpy().tobytes()).hexdigest(), rgb_tensor_sha256=hashlib.sha256(img.cpu().numpy().tobytes()).hexdigest(),selected_ids=ds.s['ids'][ids].tolist(),selection='nearest eligible training centers to 24 equally spaced nominal row-time targets; duplicates removed, no predictions used',stages=rows,channels=distributions))
    csv_write(out/'path_trace.csv',[{**r,'shape':str(r['shape'])} for r in rows])
    csv_write(out/'input_channels.csv',distributions)


def train(args,model,ds,maps,padding,run,other_sets):
    before=frozen_digest(model); history=[]; checks=[]; resources=[]
    conditions=('B0','B1','B2')
    for seed in args.seeds:
        torch.manual_seed(seed)
        branches={k:PoolingBranch(model).cuda() for k in conditions}
        initial=frozen_digest(branches['B0'])
        assert all(frozen_digest(b)==initial for b in branches.values())
        opts={k:torch.optim.Adam(b.parameters(),lr=1e-4,weight_decay=1e-4) for k,b in branches.items()}
        generator=torch.Generator().manual_seed(seed)
        loader=DataLoader(ds,batch_size=32,shuffle=True,generator=generator,num_workers=0)
        torch.cuda.reset_peak_memory_stats(); start=time.perf_counter()
        for epoch in range(args.epochs):
            mapping_seed=seed*10000+epoch
            mapping=donor_mapping(ds.s['centers'],ds.s['split'],ds.s['execution'],mapping_seed)
            npz_write(run/f'train_donors_seed{seed}_epoch{epoch+1}.npz',recipient=np.arange(len(ds)),donor=mapping,seed=np.array(mapping_seed))
            sums={k:0. for k in conditions}; orders=[]
            for batch_index,(x,img,y,indices) in enumerate(loader):
                if args.smoke and batch_index>=2: break
                orders.extend(indices.tolist())
                x,img,y=x.cuda(),img.cuda(),y.cuda()
                normal=encode(model,x,img)
                with torch.no_grad():
                    const=model.decoder(model.featurizer(constant_input(x,padding)))
                    wrong=emg_variant(ds,x,indices,'mismatch_train',{'train':mapping},padding)
                    mismatch=model.decoder(model.featurizer(wrong))
                features={'B0':(const,*normal[1:]),'B1':normal,'B2':(mismatch,*normal[1:])}
                if epoch==0 and batch_index==0:
                    with torch.no_grad():
                        b=branches['B1'].eval()(*normal)
                        raw=model({'emg':x,'vision_img':img})[...,0]
                        torch.testing.assert_close(b,raw,atol=1e-6,rtol=1e-5)
                        checks.append(dict(seed=seed,initial_released_equivalence_max=float((b-raw).abs().max())))
                for k,branch in branches.items():
                    branch.train()
                    torch.cuda.manual_seed(seed*1000000+epoch*10000+batch_index)
                    opts[k].zero_grad(set_to_none=True)
                    loss=(branch(*features[k])[:,:20]-y).abs().mean()
                    assert torch.isfinite(loss)
                    loss.backward();opts[k].step()
                    sums[k]+=float(loss.detach())*len(y)
                assert not model.training and not any(m.training for m in model.modules())
            npz_write(run/f'order_seed{seed}_epoch{epoch+1}.npz',indices=np.asarray(orders))
            row=dict(seed=seed,epoch=epoch+1,**{k:float(np.rad2deg(v/len(orders))) for k,v in sums.items()})
            history.append(row);csv_write(run/'training_history.csv',history)
            logging.info('seed=%d epoch=%d MAE B0=%.5f B1=%.5f B2=%.5f elapsed=%.1fs',seed,epoch+1,row['B0'],row['B1'],row['B2'],time.perf_counter()-start)
        torch.cuda.synchronize()
        resources.append(dict(seed=seed,training_seconds=time.perf_counter()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),parameters_per_branch=sum(p.numel() for p in branches['B0'].parameters())))
        reloaded={}
        for k,b in branches.items():
            b.eval();torch.save(b.state_dict(),run/f'{k}_seed{seed}.pt')
            reloaded[k]=PoolingBranch(model).cuda().eval()
            reloaded[k].load_state_dict(torch.load(run/f'{k}_seed{seed}.pt',weights_only=True),strict=True)
        x,img,_,_=next(iter(DataLoader(ds,batch_size=8)))
        with torch.inference_mode():
            f=encode(model,x.cuda(),img.cuda())
            for k,b in branches.items():
                error=float((b(*f)-reloaded[k](*f)).abs().max())
                assert error==0
                checks.append(dict(seed=seed,condition=k,reload_prediction_max_rad=error))
        for ep,(evds,evmaps) in other_sets.items():
            evaluate(model,reloaded,evds,evmaps,padding,run/ep/f'trained_seed{seed}.npz',seed)
        assert frozen_digest(model)==before,'Frozen parameters or buffers changed'
        json_write(run/'training_checks.json',dict(frozen_digest_before=before,frozen_digest_after=frozen_digest(model),checks=checks,resources=resources))
    return checks


def audit(args,model,hparams,base,run):
    stats=json.loads(args.norm_stats.read_text());padding=-float(base._emg_mean)/(float(base._emg_std)+1e-6)
    audit=dict(confirmed=dict(field='emg_right_filtered_paper',sample_rate_hz=2000,window_rows=WINDOW,window_nominal_seconds=WINDOW/2000,
        hand='right',raw_shape='T,8',layout_slots=16,channel_map=[10,12,0,1,2,4,5,6],
        mirror='none in executed loader path; upstream manifest left_hand_strategy is recorded separately',interpolation=False,
        order='read float32 -> sparse 8-to-16 zero placement -> scalar normalization (std+1e-6) -> transpose C,T -> CUDA FP32',
        requested_key='egoemg__filtered_paper',requested_entry=stats.get('egoemg__filtered_paper'),actual_key='egoemg',
        mean=float(base._emg_mean),std=float(base._emg_std),padding_after_normalization=padding,
        constant_control='raw zero live signal after same sparse placement and normalization: every slot equals padding_after_normalization',
        legacy_zero_control='all slots set to zero AFTER normalization, matching historical F2',
        rgb_mean=base.vision_mean.tolist(),rgb_std=base.vision_std.tolist(),rgb='read-only precrop LMDB, uint8 RGB 256x256, transpose CHW, no divide by 255 before mean/std',
        alignment='center=start+3895; same global center image mapping and reference; output 22x1 directly center supervised',
        angle_units='generated fingers already radians; wrist stored degrees converted to radians; MAE first20 linear absolute radians -> degrees',
        joint_order=json.loads((args.memmap_dir/'manifest.json').read_text())['generated_joint_angles_semantics'],
        strict_load=True,model_mode=model.fusion_mode,encoder_frozen_eval=True),
        fixed=[],unknown=dict(training_statistics='checkpoint records original training-machine path but no stats bytes/hash; legacy choice at original execution not provable from supplied checkpoint',
        checkpoint_participant_exposure='not fully traced',upstream_filter='filtered_paper name and supplied preprocessing code available; original complete processing execution not re-run',
        historical_training_logs='logs/fusion/resnet_small_emgfusion_center absent locally'),
        evidence_separated=dict(code='egoemg/datasets/egoemg_memmap_dataset.py:_load_norm_stats/_convert_to_emg2pose_layout/_getitem_center_supervised_emg',
            config='config/experiment/fusion/fusion_rn18_s_center_16ch_wl7790.yaml',checkpoint_hparams='fusion_checkpoint_hparams.json',
            training_records='no original training statistics content recovered; current config is not execution provenance'),
        normalization_decision='preserve published local asset; do not select scale by evaluation MAE',
        checkpoint_architecture='16ch TDS, 256-dim decoder 3 layers/4 heads, 22 outputs, center_supervised; strict state load verifies shapes; old emg2pose import names migrated by current config')
    json_write(run/'input_audit.json',audit)
    json_write(run/'fusion_checkpoint_hparams.json',hparams)
    return padding


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path)
    p.add_argument('--episode',default='episode_000020')
    p.add_argument('--review-episode',default='episode_000008')
    p.add_argument('--memmap-dir',type=Path,required=True)
    p.add_argument('--crops-dir',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--vision-checkpoint',type=Path,required=True)
    p.add_argument('--norm-stats',type=Path,required=True)
    p.add_argument('--history',type=Path)
    p.add_argument('--common-history',type=Path)
    p.add_argument('--review-history',type=Path)
    p.add_argument('--epochs',type=int,default=10)
    p.add_argument('--seeds',type=int,nargs='+',default=[42,43,44])
    p.add_argument('--smoke',action='store_true')
    args=p.parse_args();configure_precision()
    run=args.run_dir.resolve() if args.run_dir else create_run_directory(REPO/'myself/outputs','wmh','right','effective_emg')
    if not run.is_relative_to(REPO/'myself/outputs'): raise ValueError(run)
    if args.run_dir: run.mkdir(parents=True,exist_ok=False)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.FileHandler(run/'run.log'),logging.StreamHandler()])
    begin=time.perf_counter()
    try:
        paths=[Path(__file__),args.checkpoint,args.vision_checkpoint,args.norm_stats,args.memmap_dir/'manifest.json',args.memmap_dir/'metadata.npz']
        paths+=list((REPO/'myself/scripts/analysis').glob('*.py'))+list((REPO/'egoemg/models').rglob('*.py'))+[REPO/'egoemg/datasets/egoemg_memmap_dataset.py']
        settings=dict(args={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
            command=sys.argv,head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
            branch=subprocess.check_output(['git','branch','--show-current'],cwd=REPO,text=True).strip(),
            worktree=subprocess.check_output(['git','status','--short'],cwd=REPO,text=True),
            precision='FP32 deterministic; matmul TF32 false, cuDNN TF32 false',gpu=torch.cuda.get_device_name(),torch=torch.__version__,
            python=sys.version,optimizer=dict(name='Adam',lr=1e-4,weight_decay=1e-4,batch=32),
            selection='fixed last epoch; no scale/model/hyperparameter selection by MAE',
            block_statistics='continuous 30s center blocks within original split; descriptive, not independent participants',
            input_sha256={str(x):sha256(x) for x in paths},start_time_tokyo=__import__('datetime').datetime.now(__import__('zoneinfo').ZoneInfo('Asia/Tokyo')).isoformat())
        json_write(run/'settings.json',settings)
        src=run/'executed_sources';src.mkdir()
        for x in paths:
            if x.suffix=='.py' and x.is_relative_to(REPO):
                dest=src/x.relative_to(REPO);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(x,dest)
        cfg=configuration('fusion/fusion_rn18_s_center_16ch_wl7790')
        cfg.module.vision_pretrained_checkpoint=str(args.vision_checkpoint.resolve())
        model,hparams=strict_model(cfg.module,args.checkpoint)
        OmegaConf.save(cfg.module,run/'resolved_model.yaml',resolve=True)
        json_write(run/'resolved_dataset.json',dict(memmap_dir=str(args.memmap_dir.resolve()),crops_dir=str(args.crops_dir.resolve()),episodes=[args.episode,args.review_episode],window_length=WINDOW,stride=200,historical_window_length=12000,target_hand='right',emg_field='filtered_paper',layout='emg2pose_interpolate16',channel_indices=[10,12,0,1,2,4,5,6],channel_interpolate=False,norm_mode='per-dataset',norm_stats=str(args.norm_stats.resolve()),center_target_only=True,augmentation=None))
        primary,info=prepare(args,args.episode,args.history,run)
        review,review_info=prepare(args,args.review_episode,args.review_history,run)
        padding=audit(args,model,hparams,primary['T_all'][0].base,run)
        datasets={args.episode:primary['E'],args.review_episode:review['E']}
        path_trace(model,primary['T_all'][0],primary['T_all'][1],padding,run)
        if args.smoke:
            args.epochs=1;args.seeds=[42]
            # Process two real batches, keep only 32 evaluation centers per recording.
            for ep,(ds,maps) in list(datasets.items()):
                small={k:v[:32] for k,v in ds.s.items()}
                # donors reference the complete fixed evaluation pool; retain full dataset for reads.
                class Small(Centers):
                    pass
                tiny=Small(ds.base,ep,small)
                original_emg=ds.emg
                tiny.emg=lambda i, original_emg=original_emg: original_emg(i)
                datasets[ep]=(tiny,maps)
        for ep,(ds,maps) in datasets.items():
            evaluate(model,None,ds,maps,padding,run/ep/'released.npz')
        cfgv=configuration('fusion/vision_resnet18');cfgv.module.pretrained=False
        vision,vh=strict_model(cfgv.module,args.vision_checkpoint)
        for ep,(ds,maps) in datasets.items():
            acc=[]
            with torch.inference_mode():
                for x,img,y,i in DataLoader(ds,batch_size=32): acc.append(vision({'vision_img':img.cuda()})[...,0].cpu().numpy())
            npz_write(run/ep/'vision.npz',**ds.s,vision=np.concatenate(acc))
        del vision
        train(args,model,primary['T_all'][0],primary['T_all'][1],padding,run,datasets)
        json_write(run/'complete.json',dict(status='smoke_complete' if args.smoke else 'complete',wall_seconds=time.perf_counter()-begin,
            files={str(x.relative_to(run)):sha256(x) for x in run.rglob('*') if x.is_file() and x.suffix in ('.npz','.pt','.csv','.json','.yaml')},smoke=args.smoke))
    except Exception as exc:
        json_write(run/'failure.json',dict(type=type(exc).__name__,message=str(exc),wall_seconds=time.perf_counter()-begin))
        logging.exception('Run failed; preserve partial evidence at %s',run)
        raise

if __name__=='__main__': main()

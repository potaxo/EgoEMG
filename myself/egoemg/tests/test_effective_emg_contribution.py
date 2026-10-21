"""Protocol tests use synthetic inputs, never score-selected real windows."""
import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from myself.scripts.analysis.effective_emg_contribution import (
    constant_input, donor_mapping, support_ok,
)
from myself.scripts.analysis.visual_guided_pool_pilot import PoolingBranch, frozen_digest
from egoemg.models.modules._pooling import TemporalAttentionPool


def test_donors_are_reproducible_disjoint_bijections_and_split_isolated():
    centers=np.arange(120)*400
    split=np.repeat([0,3],60);execution=centers//6000
    a=donor_mapping(centers,split,execution,101)
    np.testing.assert_array_equal(a,donor_mapping(centers,split,execution,101))
    assert not np.array_equal(a,donor_mapping(centers,split,execution,102))
    np.testing.assert_array_equal(np.sort(a),np.arange(120))
    assert np.all(np.abs(centers-centers[a])>=7790)
    assert np.all(split==split[a])


def test_impossible_donors_fail_explicitly():
    with pytest.raises(ValueError,match='No nonoverlapping'):
        donor_mapping(np.arange(4),np.zeros(4),np.arange(4),101)


def test_full_support_boundaries_and_center_label():
    n=30000;split=np.zeros(n,int);split[15000:]=1
    valid=np.ones(n,bool);ref=np.ones((n,20));prefix=np.zeros(n+1,int)
    assert support_ok(6000,0,n,split,prefix,valid,ref)=='eligible'
    assert support_ok(3894,0,n,split,prefix,valid,ref)=='episode_support'
    assert support_ok(14000,0,n,split,prefix,valid,ref)=='split_support'
    valid[6000]=False
    assert support_ok(6000,0,n,split,prefix,valid,ref)=='invalid_label'
    valid[6000]=True;ref[6000,3]=np.nan
    assert support_ok(6000,0,n,split,prefix,valid,ref)=='nonfinite_label'
    ref[6000,3]=1;prefix[5000:]=1
    assert support_ok(6000,0,n,split,prefix,valid,ref)=='nonfinite_emg_support'


def test_constant_preserves_normalized_padding_and_removes_signal():
    x=torch.randn(3,16,30);value=-.028774764388799667/(94.73531341552734+1e-6)
    y=constant_input(x,value)
    assert torch.all(y==value) and y.shape==x.shape and y.dtype==x.dtype
    assert value!=0


def test_frozen_bn_and_save_reload_and_paired_training(tmp_path):
    torch.manual_seed(42)
    backbone=nn.Sequential(nn.Linear(8,8),nn.BatchNorm1d(8)).eval().requires_grad_(False)
    before=frozen_digest(backbone)
    source=SimpleNamespace(temporal_attn=TemporalAttentionPool(8),vision_proj=nn.Linear(512,4),
                           fusion_proj=nn.Conv1d(12,8,1),head=nn.Sequential(nn.Dropout(.1),nn.Conv1d(8,22,1)))
    def fit():
        b=PoolingBranch(source);opt=torch.optim.Adam(b.parameters(),lr=1e-4,weight_decay=1e-4)
        for i in range(2):
            with torch.no_grad(): f=backbone(torch.ones(3,8))[:,:,None].expand(-1,-1,7)
            torch.manual_seed(42000+i)
            pred=b(f,torch.ones(3,512),torch.zeros(3,22))
            opt.zero_grad();pred.abs().mean().backward();opt.step()
        return b.eval()
    a,b=fit(),fit()
    assert frozen_digest(a)==frozen_digest(b)
    assert frozen_digest(backbone)==before
    torch.save(a.state_dict(),tmp_path/'branch.pt')
    c=PoolingBranch(source).eval();c.load_state_dict(torch.load(tmp_path/'branch.pt',weights_only=True))
    inputs=(torch.ones(3,8,7),torch.ones(3,512),torch.zeros(3,22))
    assert torch.equal(a(*inputs),c(*inputs))


def test_real_loader_legacy_fallback_for_placeholder_and_sparse_slots(tmp_path):
    import json
    from egoemg.datasets.egoemg_memmap_dataset import EgoEmgMemmapDataset
    path=tmp_path/'stats.json'
    path.write_text(json.dumps({'egoemg__filtered_paper':{'note':'placeholder'},'egoemg':{'mean':.03,'std':94.7}}))
    base=EgoEmgMemmapDataset.__new__(EgoEmgMemmapDataset)
    base.norm_mode='per-dataset';base.norm_stats_path=str(path);base.dataset_name='egoemg'
    base.emg_field_preference='filtered_paper';base.emg_layout='emg2pose_interpolate16';base.target_hand='right'
    base._emg2pose_channel_indices=np.array([10,12,0,1,2,4,5,6])
    base._load_norm_stats()
    assert base._emg_mean==.03 and base._emg_std==94.7
    raw=np.tile(np.arange(8,dtype=np.float32),(3,1))
    placed,mask=base._convert_to_emg2pose_layout(raw,interpolate_missing=False)
    np.testing.assert_array_equal(placed[:,base._emg2pose_channel_indices],raw)
    np.testing.assert_array_equal(placed[:,~mask],0)
    normalized=(placed-base._emg_mean)/(base._emg_std+1e-6)
    np.testing.assert_allclose(normalized[:,~mask],-.03/(94.7+1e-6))


def test_center_wrapper_checks_reference_and_exact_global_support():
    from myself.scripts.analysis.effective_emg_contribution import Centers
    class Fake:
        _episode_id=['episode_000020']
        def _getitem_center_supervised_emg(self,ep,a,b):
            assert ep==0 and (a,b)==(6000-3895,6000+3895)
            return {'emg':np.ones((16,7790),np.float32),'joint_angles':np.arange(22,dtype=np.float32)[:,None],'label_valid_mask':[True]}
    s={'centers':np.array([6000]),'reference':np.arange(22,dtype=np.float32)[None]}
    assert Centers(Fake(),'episode_000020',s).emg(0).shape==(16,7790)
    s['reference'][0,3]+=1
    with pytest.raises(AssertionError):Centers(Fake(),'episode_000020',s).emg(0)


def test_independent_recompute_masks_and_group_weighting():
    from myself.scripts.analysis.summarize_effective_emg import score
    ref=np.zeros((3,22));pred=np.ones((3,22));mask=np.ones((3,22),bool)
    pred[2,:20]=3;mask[2,:10]=False
    expected=(40+30)/50*180/np.pi
    assert abs(score(pred,ref,mask)-expected)<1e-12
    pred[:,20:]=999
    assert abs(score(pred,ref,mask)-expected)<1e-12


def test_trace_selection_is_uniform_in_time_not_eligible_index():
    from myself.scripts.analysis.effective_emg_contribution import time_uniform_indices
    centers=np.array([0,1,2,3,4,90,95,100])
    # Targets [0, 50, 100] choose [0, 90, 100], rather than evenly spaced indices.
    np.testing.assert_array_equal(time_uniform_indices(centers,3),[0,5,7])

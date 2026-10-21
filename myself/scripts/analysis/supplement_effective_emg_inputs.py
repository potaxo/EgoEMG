#!/usr/bin/env python3
"""Read-only input timing and checkpoint/config consistency supplement."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import sys
sys.dont_write_bytecode=True
REPO=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(REPO))
import numpy as np
from omegaconf import OmegaConf
from myself.scripts.analysis.preview_calibration_smoke import open_field,sha256


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);a=p.parse_args();run=a.run_dir.resolve()
    settings=json.loads((run/'settings.json').read_text());root=Path(settings['args']['memmap_dir'])
    mf=json.loads((root/'manifest.json').read_text());md=np.load(root/'metadata.npz');timestamp=open_field(root,mf,'timestamp_us')
    timing=[]
    for ep,lo,hi in zip(md['episode_id'],md['episode_start_idx'],md['episode_end_idx']):
        name=ep.decode()
        if name not in [settings['args']['episode'],settings['args']['review_episode']]:continue
        delta=np.diff(timestamp[lo:hi]);span=float((timestamp[hi-1]-timestamp[lo])/1e6)
        timing.append(dict(episode=name,rows=int(hi-lo),span_seconds=span,span_implied_hz=float((hi-lo-1)/span),
            delta_us_quantiles={str(q):float(np.quantile(delta,q)) for q in (0,.01,.5,.9,.99,1)},nonpositive_deltas=int((delta<=0).sum()),
            loader_timing='uses row order and fixed center; no resampling from timestamp_us'))
    current=OmegaConf.to_container(OmegaConf.load(run/'resolved_model.yaml'))
    hp=json.loads((run/'fusion_checkpoint_hparams.json').read_text())['module_conf']
    def flatten(obj,prefix=''):
        out={}
        for k,v in obj.items():
            key=prefix+k
            if isinstance(v,dict):out.update(flatten(v,key+'.'))
            else:out[key]=v.replace('emg2pose.','egoemg.') if isinstance(v,str) else v
        return out
    old,new=flatten(hp),flatten(current)
    # Nested featurizer lists differ only in migrated module namespace.
    normalize=lambda v:json.dumps(v,sort_keys=True).replace('emg2pose.','egoemg.')
    shared={k:normalize(old[k])==normalize(new[k]) for k in old.keys()&new.keys()}
    differences={k:{'checkpoint':old[k],'current':new[k]} for k,equal in shared.items() if not equal}
    assert set(differences)<= {'vision_pretrained_checkpoint'},differences
    evidence=dict(timing=timing,nominal_sample_rate_hz=2000,
        timing_interpretation='episode duration agrees with nominal 2000Hz, but per-row timestamps arrive in bursts and are nonuniform; block durations are nominal row-time, not independent evenly timestamped observations',
        module_shared_keys_equal_after_namespace_migration=shared,module_shared_key_differences=differences,
        checkpoint_only_keys={k:v for k,v in old.items() if k not in new},current_only_keys={k:v for k,v in new.items() if k not in old},
        namespace='checkpoint emg2pose -> current egoemg; shape-strict loading, not proof of original-source numerical identity',
        physical_emg_unit='not specified by supplied manifest; do not infer microvolts',
        field_specs={k:mf['fields'][k] for k in ['emg_right_filtered_paper','generated_joint_angles_right','generated_label_valid','timestamp_us','image_head_frame_index','image_head_delta_ms','image_head_stale']},
        manifest_left_hand_strategy=mf.get('left_hand_strategy'),right_hand_mirroring='none in actual right-hand loader conversion',
        input_sha256={str(root/'manifest.json'):sha256(root/'manifest.json'),str(root/'metadata.npz'):sha256(root/'metadata.npz')},
        script_sha256=sha256(Path(__file__)),command=sys.argv)
    (run/'input_support_supplement.json').write_text(json.dumps(evidence,indent=2)+'\n')
    shutil.copyfile(__file__,run/'executed_input_supplement.py')
    print(json.dumps({'timing':timing,'differences':differences},indent=2))

if __name__=='__main__':main()

#!/usr/bin/env python3
"""Supplement the preserved index-spaced trace with actual uniform row-time targets."""
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
from myself.scripts.analysis.common_modality_evaluation import configuration
from myself.scripts.analysis.effective_emg_contribution import Centers,configure_precision,make_base,path_trace
from myself.scripts.analysis.preview_calibration_smoke import sha256
from myself.scripts.analysis.single_subject_calibration_suite import strict_model


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);a=p.parse_args();run=a.run_dir.resolve()
    settings=json.loads((run/'settings.json').read_text())
    args=argparse.Namespace(**settings['args'])
    for k in ['memmap_dir','crops_dir','checkpoint','vision_checkpoint','norm_stats']:setattr(args,k,Path(getattr(args,k)).resolve())
    configure_precision();base=make_base(args,args.episode)
    selected=np.load(run/args.episode/'T_all_selection.npz');ds=Centers(base,args.episode,{k:selected[k] for k in selected.files})
    maps=np.load(run/args.episode/'T_all_donors.npz');cfg=configuration('fusion/fusion_rn18_s_center_16ch_wl7790')
    cfg.module.vision_pretrained_checkpoint=str(args.vision_checkpoint);model,_=strict_model(cfg.module,args.checkpoint)
    out=run/'uniform_time_trace';out.mkdir(exist_ok=False)
    padding=-base._emg_mean/(base._emg_std+1e-6)
    path_trace(model,ds,maps,padding,out)
    sources=[Path(__file__),REPO/'myself/scripts/analysis/effective_emg_contribution.py']
    for path in sources:shutil.copyfile(path,out/path.name)
    (out/'execution.json').write_text(json.dumps(dict(command=sys.argv,source_sha256={str(p.relative_to(REPO)):sha256(p) for p in sources},
        note='diagnostic-only supplement: original full run trace and every training/prediction artifact preserved; no model updates'),indent=2)+'\n')
    print('Saved',out)

if __name__=='__main__':main()

#!/usr/bin/env python3
"""Artifact validation independent of inference; writes only a new verification receipt."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

sys.dont_write_bytecode=True
import numpy as np

REPO=Path(__file__).resolve().parents[3]


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def verify_run(run):
    complete=json.loads((run/'complete.json').read_text());assert complete['status']=='complete'
    settings=json.loads((run/'settings.json').read_text());source_copies=0;asset_hashes=0
    for name,h in settings['input_sha256'].items():
        source=Path(name)
        if source.suffix=='.py' and source.is_relative_to(REPO):
            assert sha(run/'executed_sources'/source.relative_to(REPO))==h,name
            source_copies+=1
        elif source.suffix!='.py':
            assert sha(source)==h,name
            asset_hashes+=1
    analysis=json.loads((run/'analysis/analysis_execution.json').read_text())
    assert sha(run/'executed_summarizer.py')==analysis['script_sha256']
    checked=0
    for path,digest in complete['files'].items():assert sha(run/path)==digest;checked+=1
    donor_counts={};delta_recompute=[];donor_profiles=[]
    for ep in run.glob('episode_*'):
        selections={pool:np.load(ep/f'{pool}_selection.npz') for pool in ('T_all','E')}
        for pool,s in selections.items():
            maps=np.load(ep/f'{pool}_donors.npz')
            for seed in maps.files:
                d=maps[seed];n=len(s['centers'])
                assert np.array_equal(np.sort(d),np.arange(n))
                assert np.all(s['split']==s['split'][d])
                assert np.all(np.abs(s['centers']-s['centers'][d])>=7790)
                donor_counts[f'{ep.name}/{pool}/{seed}']=n
                donor_profiles.append(dict(episode=ep.name,pool=pool,mapping_seed=seed,centers=n,different_execution=int(np.sum(s['execution']!=s['execution'][d])),different_class=int(np.sum(s['gesture']!=s['gesture'][d])),minimum_center_distance_rows=int(np.min(np.abs(s['centers']-s['centers'][d]))) if n else None))
        train=selections['T_all'];ev=selections['E']
        assert not (np.abs(train['centers'][:,None]-ev['centers'][None,:])<7790).any()
        for path in sorted(ep.glob('trained_seed*.npz')):
            d=np.load(path)
            for key in ('centers','reference','mask','frame'):np.testing.assert_array_equal(d[key],ev[key])
            b0=d['B0_constant'][:,:20].astype(np.float64);b1=d['B1_normal'][:,:20].astype(np.float64)
            y=d['reference'][:,:20].astype(np.float64);mask=d['mask'][:,:20]
            mae0=float(np.abs(b0-y)[mask].mean()*180/np.pi);mae1=float(np.abs(b1-y)[mask].mean()*180/np.pi)
            # Reconcile independently with CSV emitted by the summarizer.
            rows=list(csv.DictReader((run/'analysis/main_metrics.csv').open()))
            seed=path.stem.removeprefix('trained_seed')
            row=next(r for r in rows if r['episode']==ep.name and r['seed']==seed and r['condition']=='B1_normal')
            assert abs(float(row['mae_deg'])-mae1)<1e-12
            assert abs(float(row['gain_vs_B0_deg'])-(mae0-mae1))<1e-12
            for group in ('split','gesture','block','execution'):
                grouped=list(csv.DictReader((run/'analysis/group_metrics.csv').open()))
                selected=[r for r in grouped if r['episode']==ep.name and r['seed']==seed and r['condition']=='B1_normal' and r['grouping']==group]
                total=sum(int(r['valid_angles']) for r in selected)
                reconciled=sum(int(r['valid_angles'])*float(r['mae_deg']) for r in selected)/total
                assert abs(reconciled-mae1)<1e-10
            delta_recompute.append(dict(episode=ep.name,seed=int(seed),G_deg=mae0-mae1))
    primary=next(ep for ep in run.glob('episode_*') if len(np.load(ep/'T_all_selection.npz')['centers']))
    s=np.load(primary/'T_all_selection.npz');n=len(s['centers'])
    training_maps=0
    for path in run.glob('train_donors_seed*_epoch*.npz'):
        d=np.load(path)['donor'];assert np.array_equal(np.sort(d),np.arange(n))
        assert np.all(s['split']==s['split'][d]);assert np.all(np.abs(s['centers']-s['centers'][d])>=7790)
        training_maps+=1
    orders=list(run.glob('order_seed*_epoch*.npz'))
    for path in orders:assert np.array_equal(np.sort(np.load(path)['indices']),np.arange(n))
    assert training_maps==30 and len(orders)==30
    with (run/'donor_summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(donor_profiles[0]));w.writeheader();w.writerows(donor_profiles)
    receipt=dict(script_sha256=sha(Path(__file__)),command=sys.argv,executed_source_copies_verified=source_copies,unchanged_assets_verified=asset_hashes,execution_files_hash_verified=checked,donor_counts=donor_counts,training_maps_verified=training_maps,
        matched_epoch_orders_verified=len(orders),independent_float64_G=delta_recompute,
        grouped_weighted_reconciliation=True,complete_support_pool_isolation=True)
    return receipt


def verify_snapshot(snapshot):
    m=json.loads((snapshot/'snapshot_manifest.json').read_text());checked=[]
    for row in m['files']:
        src,dest=REPO/row['source'],snapshot/row['file']
        assert sha(src)==sha(dest)==row['sha256'];checked.append(row['file'])
    broken=[];links=0
    for path in snapshot.rglob('*.md'):
        for dest in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',path.read_text()):
            if re.match(r'\w+://',dest) or dest.startswith('#'):continue
            target=(path.parent/dest.split('#')[0]).resolve();links+=1
            if not target.exists():broken.append(str(target))
    assert not broken,broken
    for name in checked:
        assert subprocess.run(['git','check-ignore','-q',str(snapshot/name)],cwd=REPO).returncode==1,name
    assert subprocess.run(['git','check-ignore','-q',str(REPO/m['execution_source']/'run.log')],cwd=REPO).returncode==0
    return dict(snapshot_hash_files=len(checked),local_links_checked=links,all_snapshot_files_trackable=True,full_run_ignored=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--snapshot',type=Path)
    args=p.parse_args();r=verify_run(args.run_dir.resolve())
    if args.snapshot:r['snapshot']=verify_snapshot(args.snapshot.resolve())
    import shutil
    shutil.copyfile(__file__,args.run_dir/'executed_artifact_verifier.py')
    path=args.run_dir/'artifact_verification.json'
    path.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))

if __name__=='__main__':main()

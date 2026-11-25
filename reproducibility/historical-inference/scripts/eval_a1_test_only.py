"""Fail-closed A1 audit and paired evaluation; never trains or selects checkpoints.

Run from repository root. --run requires an externally verified provenance JSON
mapping each checkpoint relative path to {sha256, membership_verified: true}.
Verification must cover both stages and historical split contents, not filenames.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import platform
import re
import sqlite3
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def chip(name):
    return re.sub(r'_(S1|Label|S2IndexLabel)(Hand|Weak)$', '', Path(name).stem).casefold()


def rows(p):
    with Path(p).open(newline='') as f:
        result = [tuple(x) for x in csv.reader(f) if x]
    for x in result:
        if len(x) != 2 or chip(x[0]) != chip(x[1]):
            raise ValueError(f'Invalid paired identifier in {p}: {x}')
    ids = [chip(x[0]) for x in result]
    if len(ids) != len(set(ids)):
        raise ValueError(f'Duplicate canonical IDs in {p}')
    return result


def overlap(test, membership):
    return sorted({chip(x[0]) for x in test} & {chip(x[0]) for x in membership})


def audit(out):
    official = {s: rows(ROOT / f'data/splits/official/{s}.csv') for s in ('train','val','test')}
    events = sorted({r[0].split('_')[0] for s in official.values() for r in s})
    results = []
    for event in events:
        test = sorted(r for r in official['test'] if r[0].startswith(event + '_'))
        target = out / 'manifests' / event / 'test.csv'
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('w', newline='') as f:
            csv.writer(f).writerows(test)
        record = dict(event=event, held_out_group=event, country_mapping='unverified event prefix; Mekong is a basin',
                      n_official_test=len(test), manifest=str(target.relative_to(ROOT)),
                      manifest_sha256=digest(target), n_evaluated=0, delta_formula='LOCO IoU - generalist IoU',
                      generalist_iou=None, loco_iou=None, delta=None, checkpoints=[], overlaps={}, missing_samples=[])
        for arm, split in [('generalist','data/splits/official'),('loco',f'data/splits/loco/{event}/train9')]:
            record[arm + '_unique_labeled_train_current'] = len(rows(ROOT / split / 'train.csv'))
            for s in ('train','val'):
                record['overlaps'][arm+'_'+s] = overlap(test, rows(ROOT / split / (s+'.csv')))
        for arm, split in [('generalist_weak','data/splits/weak'),('loco_weak',f'data/splits/loco/{event}/weak9')]:
            for s in ('train','val'):
                record['overlaps'][arm+'_'+s] = overlap(test, rows(ROOT / split / (s+'.csv')))
        for r in test:
            for sub, name in zip(('S1Hand','LabelHand'),r):
                p = ROOT / 'data/processed/sen1floods11' / sub / name
                if not p.is_file(): record['missing_samples'].append(str(p))
        candidates = [('generalist',ROOT/'models/weak_pretrain_finetune/finetune/best.ckpt')]
        candidates += [('loco',p) for p in sorted((ROOT/f'models/loco/{event}').glob('finetune*/best*.ckpt'))]
        for arm,p in candidates:
            record['checkpoints'].append(dict(arm=arm,path=str(p.relative_to(ROOT)),exists=p.exists(),sha256=digest(p) if p.exists() else None))
        record['status'] = 'blocked: historical checkpoint membership and event/country mapping not verified'
        results.append(record)
    # Read-only run metadata: logged paths are evidence, not immutable membership.
    db = sqlite3.connect(f'file:{ROOT / "mlflow.db"}?mode=ro',uri=True)
    logged=[]
    for rid,exp in db.execute("select r.run_uuid,e.name from runs r join experiments e on r.experiment_id=e.experiment_id where e.name like '%loco%' or e.name = 'flood-water-seg/weak_pretrain_finetune'"):
        params=dict(db.execute('select key,value from params where run_uuid=?',(rid,)))
        logged.append(dict(run_id=rid,experiment=exp,params=params))
    db.close()
    (out/'logged_runs.json').write_text(json.dumps(logged,indent=2))
    meta=dict(python=platform.python_version(),commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
              working_tree=subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True),
              official_hashes={s:digest(ROOT/f'data/splits/official/{s}.csv') for s in official},
              split_file_hashes={str(p.relative_to(ROOT)):digest(p) for p in sorted((ROOT/'data/splits').rglob('*.csv')) if '/control/' not in str(p)},
              official_counts={s:len(v) for s,v in official.items()},
              train_fraction_of_all=len(official['train'])/sum(map(len,official.values())),
              note='Current membership audit only; no historical split hashes embedded in checkpoints. Bolivia is separate from official test.',results=results)
    (out/'audit.json').write_text(json.dumps(meta,indent=2))
    return meta


def run_pair(record, provenance, smoke=False, diagnostic=False):
    import torch
    import yaml
    def cfg_raw():
        return yaml.safe_load((ROOT / "config/experiments/weak_finetune.yaml").read_text())
    import sys
    sys.path.insert(0,str(ROOT/'src'))
    from data.datamodule import FloodDataModule
    from inference.evaluate import evaluate
    from utils.config import Config
    from training.lightning_module import FloodModel
    def load_model(path):
        raw=cfg_raw(); raw['model']['pretrained']=False
        model=FloodModel(Config.model_validate(raw))
        state=torch.load(path,map_location='cpu',weights_only=False)
        model.load_state_dict(state['state_dict'],strict=True)
        return model
    selected=provenance.get('events',{}).get(record['event'],{})
    if diagnostic:
        selected={arm: next((c['path'] for c in record['checkpoints'] if c['arm']==arm and c['path'].endswith('/finetune/best.ckpt')),None) for arm in ('generalist','loco')}
    if not diagnostic and selected and not selected.get('held_out_country'):
        raise ValueError('Verified event-to-country mapping required')
    record['held_out_country']=selected.get('held_out_country','unverified')
    paths=[selected.get(arm) for arm in ('generalist','loco')]
    if not all(paths): raise ValueError('Explicit checkpoint mapping required for '+record['event'])
    historical={}
    record["historical_membership_audit"]=historical
    for arm,p in zip(('generalist','loco'),paths):
        attestation=provenance.get('checkpoints',{}).get(p,{})
        if not diagnostic and (not attestation.get('membership_verified') or attestation.get('sha256') != digest(ROOT/p)):
            raise ValueError('Historical membership not verified: '+p)
        if not diagnostic:
            expected={(stage,role) for stage in ('pretrain','finetune') for role in ('train','val')}
            splits=attestation.get('splits',[])
            if {(x['stage'],x['role']) for x in splits} != expected or len(splits)!=4:
                raise ValueError('Four historical stage/role manifests required: '+p)
            historical[arm]={}
            test_rows=rows(ROOT/record['manifest'])
            for item in splits:
                source=Path(item['path'])
                if not source.is_absolute(): source=ROOT/source
                if digest(source)!=item['sha256']: raise ValueError('Historical manifest hash mismatch: '+str(source))
                members=rows(source)
                collision=overlap(test_rows,members)
                historical[arm][item['stage']+'_'+item['role']]=dict(n_unique=len(members),overlap_ids=collision,sha256=item['sha256'])
                if collision: raise ValueError('Historical '+item['role']+' overlap: '+str(collision))
                if arm=='loco' and any(x[0].split('_')[0]==record['event'] for x in members):
                    raise ValueError('Held-out event present in historical LOCO membership')
            if attestation.get('evaluation_config_sha256') != digest(ROOT/'config/experiments/weak_finetune.yaml'):
                raise ValueError('Verified inference/preprocessing config hash required: '+p)
            if not attestation.get('selection_calibration_verified_no_test'):
                raise ValueError('Selection/calibration provenance required: '+p)
        record['historical_membership_audit']=historical
    if record['missing_samples'] or not record['n_official_test']:
        raise ValueError('Overlap/missing/empty official test: '+record['event'])
    raw=cfg_raw(); raw['data'].update(split_dir=str(ROOT/record['manifest']).rsplit('/',1)[0],num_workers=0)
    dm=FloodDataModule(Config.model_validate(raw)); dm.setup('test')
    # Eager readability/label audit of every official sample, without dropping any.
    valid_pixels=0
    empty_valid_chips=[]
    for i in range(len(dm.test_dataset)):
        _, label, mask, _=dm.test_dataset[i]
        valid_pixels+=int(mask.sum())
        if not mask.any(): empty_valid_chips.append(str(dm.test_dataset.image_paths[i]))
        if not set(label[mask].unique().tolist()) <= {0,1}:
            raise ValueError('Unexpected labels')
    if not valid_pixels: raise ValueError('Official event subset has no valid pixels')
    if smoke:
        from torch.utils.data import DataLoader, Subset
        loader=DataLoader(Subset(dm.test_dataset,[0]),batch_size=1)
    else: loader=dm.test_dataloader()
    scores=[]
    for p in paths:
        model=load_model(str(ROOT/p))
        scores.append(evaluate(model,loader,'cuda' if torch.cuda.is_available() else 'cpu',tta=False))
        del model
    return dict(generalist=scores[0],loco=scores[1],delta=scores[1]['iou']-scores[0]['iou'],n_evaluated=1 if smoke else len(dm.test_dataset),valid_pixels_full_manifest=valid_pixels,zero_valid_pixel_chips=empty_valid_chips)



def write_summary(meta, out):
    inventory_path=out/'sample_inventory.json'
    inventory=json.loads(inventory_path.read_text())['samples'] if inventory_path.exists() else None
    columns=['event','held_out_country','n_official_test','n_evaluated','generalist_iou','generalist_f1','loco_iou','loco_f1','delta','generalist_train_current','loco_train_current','historical_provenance','zero_valid_pixel_chips','status']
    columns += [arm+'_'+role+'_overlap_current' for arm in ('generalist','loco','generalist_weak','loco_weak') for role in ('train','val')]
    with (out/'summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=columns); writer.writeheader()
        for record in meta['results']:
            row={k:record.get(k) for k in columns[:9]}
            row.update(delta=record.get('delta'),generalist_train_current=record['generalist_unique_labeled_train_current'],loco_train_current=record['loco_unique_labeled_train_current'],historical_provenance='verified by supplied manifests' if record['status']=='complete' else 'unresolved',zero_valid_pixel_chips=sum(x['split']=='test' and x['image'].startswith(record['event']+'_') and x.get('valid_pixels')==0 for x in inventory) if inventory is not None else None,status=record['status'])
            if record['status']=='complete':
                for arm in ('generalist','loco'): row[arm+'_f1']=record['evaluation'][arm]['f1']
            for arm in ('generalist','loco','generalist_weak','loco_weak'):
                for role in ('train','val'): row[arm+'_'+role+'_overlap_current']=len(record['overlaps'][arm+'_'+role])
            writer.writerow(row)


def main():
    p=argparse.ArgumentParser(); p.add_argument('--out',default='data/reference/a1_test_only'); p.add_argument('--run',action='store_true'); p.add_argument('--provenance'); p.add_argument('--events',nargs='+'); p.add_argument('--smoke',action='store_true'); p.add_argument('--diagnostic',action='store_true',help='Provisional smoke inference only; bypass historical attestation'); a=p.parse_args()
    if a.diagnostic and not a.smoke: p.error('--diagnostic requires --smoke')
    out=ROOT/a.out; out.mkdir(parents=True,exist_ok=True); meta=audit(out)
    if a.events and set(a.events)-{x['event'] for x in meta['results']}: p.error('Unknown event group in --events')
    if a.run:
        provenance=json.loads(Path(a.provenance).read_text()) if a.provenance else {}
        for r in meta['results']:
            if a.events and r['event'] not in a.events: continue
            try:
                r['evaluation']=run_pair(r,provenance,a.smoke,a.diagnostic)
                if not a.smoke:
                    r.update(n_evaluated=r['evaluation']['n_evaluated'],generalist_iou=r['evaluation']['generalist']['iou'],loco_iou=r['evaluation']['loco']['iou'],delta=r['evaluation']['delta'])
                r['status']='diagnostic smoke; provenance unresolved' if a.diagnostic else ('smoke only' if a.smoke else 'complete')
            except Exception as e: r['status']='blocked: '+str(e)
        (out/('smoke.json' if a.smoke else 'results.json')).write_text(json.dumps(meta,indent=2))
    if not a.smoke: write_summary(meta,out)
    print(json.dumps([{k:r[k] for k in ('event','n_official_test','status')} for r in meta['results']],indent=2))
    if a.run and any(x['status'].startswith('blocked') for x in meta['results'] if not a.events or x['event'] in a.events): return 2
    return 0

if __name__=='__main__': raise SystemExit(main())

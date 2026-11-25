"""Frozen, explicitly R-strong partial reconstruction of A1; no training.

freeze inspects hashes, run/validation evidence and committed manifests before
any new inference. run consumes that immutable selection and saves per-chip
confusion counts. Historical initialization and spatial exposure remain unknown.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
import sqlite3
import subprocess
import sys
from collections import Counter
from pathlib import Path

from eval_a1_test_only import ROOT, chip, digest, rows, overlap
OUT = ROOT / 'data/reference/a1_reconstruction'
SELECTED = [
    ('Somalia',42,'48584935','models/loco/Somalia/finetune/best.ckpt'),
    ('Spain',42,'d5b7c3b1','models/loco/Spain/finetune/best.ckpt'),
    ('Sri-Lanka',42,'faabe0d1','models/loco/Sri-Lanka/finetune/best.ckpt'),
    ('USA',42,'46bcbbc8','models/loco/USA/finetune/best-v1.ckpt'),
    ('Pakistan',43,'212b10bb','models/loco/Pakistan/finetune_seed43/best.ckpt'),
    ('Pakistan',44,'345e9433','models/loco/Pakistan/finetune_seed44/best.ckpt'),
    ('Mekong',43,'152b5e4f','models/loco/Mekong/finetune_seed43/best.ckpt'),
    ('Mekong',44,'5b38d529','models/loco/Mekong/finetune_seed44/best.ckpt'),
    ('Nigeria',43,'e58513e8','mlruns/21/e58513e88c384f2f84d914444d535282/artifacts/best/best.ckpt'),
]


def git_bytes(commit, path):
    return subprocess.check_output(['git','show',f'{commit}:{path}'],cwd=ROOT)


def dump(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2)+'\n')


def cache_audit():
    candidates=[]
    for base, dirs, files in os.walk(ROOT):
        dirs[:]=[d for d in dirs if d not in {'.git','.venv','node_modules','.dvc','processed','raw','jrc_gsw','dem','a1_reconstruction'}]
        for name in files:
            p=Path(base)/name
            if p.suffix.lower() in {'.npz','.npy','.parquet','.csv','.json','.tif','.tiff','.png','.db','.ckpt'}:
                candidates.append(dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size))
    schemas={}
    for x in candidates:
        p=ROOT/x['path']
        if p.suffix=='.csv':
            try:
                with p.open() as f: schemas[x['path']]=next(csv.reader(f),[])
            except (UnicodeError,OSError): pass
    db=sqlite3.connect(f'file:{ROOT/"mlflow.db"}?mode=ro',uri=True)
    metric_keys=[x[0] for x in db.execute('select distinct key from metrics')]; db.close()
    return dict(candidates=candidates,csv_first_rows=schemas,mlflow_metric_keys=metric_keys,
                decision='No reusable complete test-chip predictions or per-chip confusion counts with run/checkpoint and inference provenance found. CSVs are split lists or aggregate metrics; TIFF/PNG outputs are deployment/reference products, not original LOCO chip predictions. No cache reused.',
                archive='loco_results.tar.gz inspected: two complete pretrain copies then truncated fine-tune member; no prediction/count members before truncation. No artifact deleted or extracted over existing files.')


def freeze():
    import torch
    OUT.mkdir(parents=True,exist_ok=True)
    target=OUT/'selection_frozen.json'
    if target.exists(): raise FileExistsError('Selection already frozen; do not overwrite it')
    dump(OUT/'cache_recovery_audit.json',cache_audit())
    db=sqlite3.connect(f'file:{ROOT/"mlflow.db"}?mode=ro',uri=True)
    supplied={line.split(maxsplit=1)[1].strip():line.split()[0] for line in (ROOT/'data/reference/checkpoint_sha256.txt').read_text().splitlines() if line.strip()}
    official={k:rows(ROOT/f'data/splits/official/{k}.csv') for k in ('train','val','test')}
    events=sorted({x[0].split('_')[0] for x in official['test']})
    manifests={}
    for event in events:
        entries=sorted(x for x in official['test'] if x[0].startswith(event+'_'))
        path=OUT/'manifests'/event/'test.csv';path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('w',newline='') as f:csv.writer(f).writerows(entries)
        manifests[event]=dict(path=str(path.relative_to(ROOT)),sha256=digest(path),n=len(entries),held_out_country=event if event!='Mekong' else 'Mekong basin event (not a single country)')
    def checkpoint(event,seed,prefix,path):
        rid,status=db.execute('select run_uuid,status from runs where run_uuid like ?',(prefix+'%',)).fetchone()
        if status!='FINISHED':raise ValueError('Uncompleted selected run')
        params=dict(db.execute('select key,value from params where run_uuid=?',(rid,)))
        tags=dict(db.execute('select key,value from tags where run_uuid=?',(rid,)))
        sha=digest(ROOT/path)
        if path in supplied and supplied[path]!=sha:raise ValueError('Supplied SHA mismatch')
        state=torch.load(ROOT/path,map_location='cpu',weights_only=False)
        callbacks=[v for v in state['callbacks'].values() if isinstance(v,dict) and 'best_model_score' in v]
        callback=callbacks[0]
        maximum=db.execute("select max(value) from metrics where run_uuid=? and key='val_iou'",(rid,)).fetchone()[0]
        if abs(float(callback['best_model_score'])-maximum)>1e-5: raise ValueError('Validation selection mismatch '+rid)
        for key,value in {'data/vv_clip':'(-25.0, 0.0)','data/vh_clip':'(-35.0, 0.0)','data/ratio_clip':'(0.0, 18.0)','inference/threshold':'0.5'}.items():
            if params.get(key)!=value:raise ValueError('Config mismatch '+key)
        snapshots=[]
        for stage,base,commit in [('finetune',f'data/splits/loco/{event}/train9' if event else 'data/splits/official','f5c3432' if event else 'd42023b'),('pretrain',f'data/splits/loco/{event}/weak9' if event else 'data/splits/weak','f5c3432' if event else '1d2199b')]:
            for role in ('train','val'):
                source=base+'/'+role+'.csv';data=git_bytes(commit,source)
                if data!=(ROOT/source).read_bytes():raise ValueError('Current/committed manifest mismatch '+source)
                dest=OUT/'provenance_manifests'/commit/source;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
                members=rows(dest)
                test=official['test'] if not event else rows(ROOT/manifests[event]['path'])
                collisions=overlap(test,members)
                if collisions:raise ValueError('Test overlap '+source)
                if event and any(x[0].startswith(event+'_') for x in members):raise ValueError('Heldout member '+source)
                snapshots.append(dict(stage=stage,role=role,path=str(dest.relative_to(ROOT)),git_reference=commit+':'+source,sha256=digest(dest),unique_chips=len(members),test_overlap=collisions,bytes_evidence='D',historical_usage_evidence='R-strong'))
        result=dict(event=event or 'generalist',seed=seed,path=path,sha256=sha,run_id=rid,run_status=status,run_mapping_evidence='D for stored run/artifact metadata; R-strong for model-directory hash/callback match',membership_evidence='R-strong; not fully proven',params=params,tags=tags,epoch=int(state['epoch']),global_step=int(state['global_step']),best_validation_score=float(callback['best_model_score']),checkpoint_best_model_path=callback.get('best_model_path'),manifests=snapshots,initialization_evidence='R pipeline reconstruction; init weights not embedded',unresolved=['historical process-read membership','spatial/derived-sample overlap','pretraining initialization identity'])
        if event=='USA':result['initialization_evidence']='U: completed fine-tune may have loaded attempt-1 pretrain best.ckpt; completed retry pretrain best-v1 is not established as its initialization'
        del state
        return result
    generalist=checkpoint(None,42,'c0ec384f','models/weak_pretrain_finetune/finetune/best.ckpt')
    selected=[checkpoint(*x) for x in SELECTED]
    inventory=json.loads((ROOT/'data/reference/a1_test_only/sample_inventory.json').read_text())
    data=dict(label='partial reconstruction, R-strong membership; not the complete original experiment',frozen_at_utc=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),selection_rule='Existing completed runs; maximum validation IoU checkpoint. USA completed retry best-v1; interrupted attempt retained historically. No new test scores used for selection.',generalist=generalist,selected=selected,manifests=manifests,policy=dict(preprocessing='VV [-25,0], VH [-35,0], VV-VH [0,18], clip/scale',threshold='sigmoid > 0.5',tta=False,postprocessing='none',valid_mask='label != -1 AND all raw SAR bands finite',aggregation='pooled flood TP/FP/FN; no mean per-chip scores',empty_chip='covered, all counts zero, individual IoU/F1 null'),all_ignored_test=[x['image'] for x in inventory['samples'] if x['split']=='test' and not x['valid_pixels']],config_sha256=digest(ROOT/'config/experiments/weak_finetune.yaml'),git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),working_tree=subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True),source_hashes={p:digest(ROOT/p) for p in ['scripts/reconstruct_a1.py','src/inference/evaluate.py','src/data/preprocessing.py','src/data/dataset.py']})
    with target.open('x') as f:json.dump(data,f,indent=2)
    (OUT/'selection_frozen.sha256').write_text(digest(target)+'\n')
    db.close()
    print('Frozen before inference:',len(selected),'LOCO runs;',len(set(x['event'] for x in selected)),'events')


def metrics(counts):
    totals={k:sum(x[k] for x in counts) for k in ('tp','fp','fn','tn','valid_pixels')}
    totals['iou']=totals['tp']/(totals['tp']+totals['fp']+totals['fn']) if totals['tp']+totals['fp']+totals['fn'] else (0.0 if totals['valid_pixels'] else None)
    den=2*totals['tp']+totals['fp']+totals['fn']
    totals['f1']=2*totals['tp']/den if den else (0.0 if totals['valid_pixels'] else None)
    return totals


def run(smoke=False):
    import torch,yaml
    sys.path.insert(0,str(ROOT/'src'))
    from data.datamodule import FloodDataModule
    from training.lightning_module import FloodModel
    from utils.config import Config
    from inference.evaluate import evaluate
    frozen=OUT/'selection_frozen.json'
    if digest(frozen)!=(OUT/'selection_frozen.sha256').read_text().strip():raise ValueError('Frozen selection changed')
    plan=json.loads(frozen.read_text())
    if digest(ROOT/'config/experiments/weak_finetune.yaml')!=plan['config_sha256']:raise ValueError('Config changed after freeze')
    torch.set_num_threads(4)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    def score(entry,event):
        path=ROOT/entry['path']
        if digest(path)!=entry['sha256']:raise ValueError('Checkpoint changed')
        manifest=plan['manifests'][event]
        if digest(ROOT/manifest['path'])!=manifest['sha256']:raise ValueError('Manifest changed')
        raw=yaml.safe_load((ROOT/'config/experiments/weak_finetune.yaml').read_text());raw['model']['pretrained']=False
        raw['data'].update(split_dir=str((ROOT/manifest['path']).parent),num_workers=0)
        cfg=Config.model_validate(raw);dm=FloodDataModule(cfg);dm.setup('test')
        model=FloodModel(cfg);state=torch.load(path,map_location='cpu',weights_only=False);model.load_state_dict(state['state_dict'],strict=True);del state
        model.eval().to(device);result=[]
        with torch.no_grad():
            for images,labels,masks,paths in dm.test_dataloader():
                probability=torch.sigmoid(model(images.to(device))).cpu().squeeze(1)
                for prob,label,mask,name in zip(probability,labels,masks,paths):
                    if not set(label[mask].unique().tolist())<={0,1}:raise ValueError('Unknown label')
                    prediction=prob>0.5;truth=label==1
                    row=dict(chip_id=chip(name),event=event,valid_pixels=int(mask.sum()),tp=int((prediction & truth & mask).sum()),fp=int((prediction & ~truth & mask).sum()),fn=int((~prediction & truth & mask).sum()),tn=int((~prediction & ~truth & mask).sum()),image_sha256=digest(name),label_sha256=digest(ROOT/'data/processed/sen1floods11/LabelHand'/Path(name).name.replace('_S1Hand','_LabelHand')),mask_sha256=hashlib.sha256(mask.numpy().tobytes()).hexdigest(),checkpoint_sha256=entry['sha256'],run_id=entry['run_id'],status='scored' if mask.any() else 'all_ignored')
                    row.update(iou=metrics([row])['iou'],f1=metrics([row])['f1']);result.append(row)
                if smoke:break
        if not smoke:
            expected={chip(x[0]) for x in rows(ROOT/manifest['path'])}
            if {x['chip_id'] for x in result}!=expected or len(result)!=manifest['n']:raise ValueError('Incomplete coverage')
            reference=evaluate(model,dm.test_dataloader(),device,tta=False)
            for metric in ('iou','f1'):
                if abs(reference[metric]-metrics(result)[metric])>1e-6:raise ValueError('Metric aggregation mismatch')
        del model
        return result
    generalist={};individual=[];coverage=[]
    for event,manifest in plan['manifests'].items():
        g=score(plan['generalist'],event);generalist[event]=g
        save_counts('generalist_'+event,g,smoke)
        selected=[x for x in plan['selected'] if x['event']==event]
        coverage.append(dict(event=event,n_official_test=manifest['n'],available_seeds=[x['seed'] for x in selected],n_available=len(selected),missing_seeds=sorted({42,43,44}-{x['seed'] for x in selected}),status='partial reconstruction' if selected else 'missing checkpoint / no usable cached counts'))
        for entry in selected:
            loco=score(entry,event);save_counts(event+'_s'+str(entry['seed']),loco,smoke)
            if [(x['chip_id'],x['mask_sha256']) for x in g]!=[(x['chip_id'],x['mask_sha256']) for x in loco]:raise ValueError('Unpaired membership or valid pixels')
            gm,lm=metrics(g),metrics(loco)
            individual.append(dict(event=event,seed=entry['seed'],n_chips=len(loco),n_valid_chips=sum(x['valid_pixels']>0 for x in loco),n_all_ignored=sum(x['valid_pixels']==0 for x in loco),generalist_iou=gm['iou'],generalist_f1=gm['f1'],loco_iou=lm['iou'],loco_f1=lm['f1'],delta=lm['iou']-gm['iou'],checkpoint_sha256=entry['sha256'],run_id=entry['run_id'],evidence_class='R-strong partial reconstruction',initialization=entry['initialization_evidence']))
            print(individual[-1],flush=True)
        if smoke:break
    dump(OUT/('smoke_reconstruction.json' if smoke else 'results.json'),dict(selection_sha256=digest(frozen),individual=individual,coverage=coverage,generalist={e:metrics(g) for e,g in generalist.items()},no_overall_ten_event_mean='Missing events retained explicitly; no complete-experiment mean computed.'))
    if not smoke:
        for name,data in [('individual_seeds.csv',individual),('coverage.csv',coverage)]:
            with (OUT/name).open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)


def save_counts(name,counts,smoke):
    path=OUT/('smoke_counts' if smoke else 'counts')/(name+'.csv');path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(counts[0]));w.writeheader();w.writerows(counts)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['freeze','run']);parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    if args.action=='freeze':freeze()
    else:run(args.smoke)

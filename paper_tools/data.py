"""Read completed scalar runs and explicitly retained instability trajectories."""
import csv
from pathlib import Path
import numpy as np
import pandas as pd
from . import style
from .common import PANEL_SWEEPS, entries, read_json, record_input, run_dir, validate_run_config


def load_paper_data(results, checksums):
    results=Path(results)
    provenance=[]
    def read_run(record, group):
        validate_run_config(results,record)
        rd=run_dir(results,record);state=read_json(rd/'status.json');cfg=record['config']
        if state['status']!='completed':raise ValueError(f'Incomplete {group} run: {rd}')
        filename='metrics.json' if record['backend']=='control' else 'eval.csv'
        path=rd/filename;record_input(path,results,checksums)
        if filename=='eval.csv':
            with path.open() as f:rows=list(csv.DictReader(f))
        else:
            rows=[dict(step=v['step'],split=s,accuracy=v[s+'_accuracy'],error=v[s+'_error'])
                  for v in read_json(path) for s in ('train','test')]
        for split in ('train','test'):
            x,y=style.series(rows,split,'accuracy');style.validate_series(x,y,cfg['max_steps'])
            if not ((y>=0)&(y<=1)).all():raise ValueError(f'Invalid accuracy in {rd}')
        provenance.append(dict(group=group,dataset=cfg['dataset'],method=cfg['optimizer'],seed=cfg['seed'],
                               loss=record['loss'],path=str(Path('runs')/record['run_id']),
                               final_step=cfg['max_steps'],backend=record['backend']))
        return cfg,state,rows
    base={};configs={};controls={};tp={};tpstates={}
    for rec in entries(results,'fig2_optimizer_comparison'):
        cfg,state,rows=read_run(rec,'fig2');key=(cfg['dataset'],cfg['optimizer'],cfg['seed'])
        if key in base:raise ValueError(f'Duplicate Fig.2 trial {key}')
        base[key]=rows;configs[key]=cfg
    for rec in entries(results,'mse_ce'):
        cfg,state,rows=read_run(rec,'mse_ce');key=(cfg['dataset'],cfg['optimizer'],rec['loss'],cfg['seed'])
        if key in controls:raise ValueError(f'Duplicate objective trial {key}')
        controls[key]=rows
    for rec in entries(results,'tp_dtp'):
        cfg,state,rows=read_run(rec,'tp_dtp');key=(cfg['dataset'],cfg['optimizer'],cfg['seed'])
        if key in tp:raise ValueError(f'Duplicate TP/DTP trial {key}')
        tp[key]=rows;tpstates[key]=state
    if (len(base),len(controls),len(tp))!=(180,240,60):
        raise ValueError('The complete paper requires 180 Fig.2, 240 objective and 60 TP/DTP runs')
    for dataset in style.DATASETS:
        for seed in range(10):
            initial=[style.series(base[dataset,m,seed],'test','accuracy')[1][0] for m in style.METHODS]
            if max(initial)!=min(initial):raise ValueError('Fig.2 initialization accuracy mismatch')
        for method in style.GD:
            for seed in range(10):
                a=style.series(controls[dataset,method,'mse',seed],'test','accuracy')
                b=style.series(controls[dataset,method,'ce',seed],'test','accuracy')
                if not np.array_equal(a[0],b[0]) or a[1][0]!=b[1][0]:
                    raise ValueError('Objective pairing/grid mismatch')
    return dict(base=base,configs=configs,controls=controls,tp=tp,tpstates=tpstates,
                provenance=provenance,ce_status='paper scalar individual-run backend; logits argmax')


def panel_runs(results, panel, completed_only=True, checksums=None):
    results=Path(results);rows=[]
    for rec in entries(results,PANEL_SWEEPS[panel]):
        validate_run_config(results,rec)
        rd=run_dir(results,rec);state=read_json(rd/'status.json')
        if state['status'] not in ('completed','failed'):raise ValueError(f'Unfinished panel run: {rd}')
        if completed_only and state['status']!='completed':continue
        rows.append({**rec['config'],'run_id':rec['run_id'],'run_dir':str(rd),'status':state['status']})
        if checksums is not None:
            for filename in ('eval.csv','er_layers.csv'):
                path=rd/filename
                if path.exists():record_input(path,results,checksums)
    return pd.DataFrame(rows)


def long_eval(results, panel, completed_only=True, checksums=None):
    parts=[]
    for _,meta in panel_runs(results,panel,completed_only,checksums).iterrows():
        path=Path(meta.run_dir)/'eval.csv'
        if not path.exists():
            if meta.status=='completed':raise FileNotFoundError(path)
            continue
        ev=pd.read_csv(path);ev['error']=1-ev['accuracy']
        for column in ('run_id','dataset','optimizer','seed','batch_size','lr','er_mix','ridge_lambda',
                       'er_solver','secant_clip','secant_small_series','target_value'):
            if column in meta.index:ev[column]=meta[column]
        ev['run_status']=meta.status;parts.append(ev)
    if not parts:raise ValueError(f'No trajectory records for {panel}')
    return pd.concat(parts,ignore_index=True).drop_duplicates(['run_id','step','split'])

"""Regression checks against the adopted scalar paper baseline."""
import argparse
from collections import Counter
import csv
import gzip
import json
import math
from pathlib import Path

RUNTIME={'device','data_root','output_root','run_name','num_workers','log_every','save_model','eval_every','eval_every_epochs','epochs'}
ER_FIELDS={'er_solver','ridge_lambda','ridge_relative','ridge_strategy','solve_dtype','er_mix','secant_clip','secant_small','secant_small_series','forward_recompute','cholesky_retries','diag_every','diag_every_epochs','log_er_layers','diag_schedule'}


def key(config, loss):
    cfg={k:v for k,v in config.items() if k not in RUNTIME}
    if cfg['optimizer']=='er':cfg['optimizer']='er1'
    er=cfg['optimizer'] in ('er0','er1','pcg32_10','tp','dtp')
    for field in (['lr','momentum','weight_decay'] if er else ER_FIELDS):cfg.pop(field,None)
    if cfg['optimizer']!='momentum':cfg.pop('momentum',None)
    if er:loss='original'
    return json.dumps([cfg,loss],sort_keys=True)


def read_reference(path):
    path=Path(path)
    opener=gzip.open if path.suffix=='.gz' else open
    with opener(path,'rt',encoding='utf-8') as stream:return json.load(stream)


def load_fresh(output):
    found={}
    for rd in sorted((Path(output)/'runs').glob('*')):
        if not (rd/'status.json').exists():continue
        state=json.loads((rd/'status.json').read_text())
        if (rd/'record.json').exists():
            record=json.loads((rd/'record.json').read_text())
            if record.get('smoke'):continue
            config=record['config'];loss=record['loss'];backend=record.get('backend','control')
            path=rd/'metrics.json';raw=json.loads(path.read_text()) if path.exists() else []
            series=[dict(step=r['step'],accuracy=r['test_accuracy']) for r in raw]
        elif (rd/'config.json').exists():
            config=json.loads((rd/'config.json').read_text());loss='mse';backend='original';path=rd/'eval.csv'
            if path.exists():
                with path.open() as stream:series=[r for r in csv.DictReader(stream) if r['split']=='test']
            else:series=[]
        else:continue
        failure=json.loads((rd/'failure.json').read_text()) if (rd/'failure.json').exists() else {}
        values=[float(r['accuracy']) for r in series]
        valid=bool(values) and all(math.isfinite(v) and 0<=v<=1 for v in values)
        actual=dict(run_id=rd.name,backend=backend,status=state['status'],
                    failure_type=state.get('exception_type',failure.get('exception_type')),
                    final_step=int(float(series[-1]['step'])) if series else None,
                    final_accuracy=values[-1] if values else None,
                    best_accuracy=max(values) if valid else None,valid_metrics=valid)
        found.setdefault(key(config,loss),[]).append(actual)
    return found


def compare(reference, output, tolerance_pp=0.5):
    if not math.isfinite(tolerance_pp) or tolerance_pp<0:raise ValueError('Tolerance must be finite and nonnegative')
    metadata=reference.get('metadata',{}) if isinstance(reference,dict) else {}
    records=reference['records'] if isinstance(reference,dict) else reference
    if not isinstance(records,list) or not records:raise ValueError('Reference must contain at least one trial')
    fresh=load_fresh(output);rows=[]
    for ref in records:
        cfg=ref['config'];candidates=fresh.get(key(cfg,ref['loss']),[])
        if 'backend' in ref:candidates=[x for x in candidates if x['backend']==ref['backend']]
        row=dict(group=ref['group'],dataset=cfg['dataset'],optimizer=cfg['optimizer'],seed=cfg['seed'],loss=ref['loss'],expected_status=ref['status'],status='missing')
        if 'backend' in ref:row['backend']=ref['backend']
        if 'reference_run_id' in ref:
            exact=[x for x in candidates if x['run_id']==ref['reference_run_id']]
            if exact:candidates=exact
            elif len(candidates)>1:
                rows.append({**row,'status':'ambiguous_match','candidate_runs':[x['run_id'] for x in candidates]});continue
        if not candidates:rows.append(row);continue
        for actual in candidates:
            result={**row,'run_id':actual['run_id'],'actual_status':actual['status']}
            if actual['status'] in ('running','interrupted'):result['status']='pending'
            elif actual['status']!=ref['status']:result['status']='status_mismatch'
            elif ref['status']=='failed':
                result['status']='expected_failure' if not ref.get('failure_type') or actual['failure_type']==ref['failure_type'] else 'failure_type_mismatch'
            elif not actual['valid_metrics']:result['status']='invalid_metrics'
            elif actual['final_step']!=ref['final_step']:result['status']='budget_mismatch'
            else:
                for metric in ('final_accuracy','best_accuracy'):result[metric+'_delta_pp']=100*(actual[metric]-ref[metric])
                result['status']='within_tolerance' if all(abs(result[m+'_delta_pp'])<=tolerance_pp for m in ('final_accuracy','best_accuracy')) else 'numerical_mismatch'
            rows.append(result)
    return dict(reference=metadata,tolerance_pp=tolerance_pp,reference_trials=len(records),
                counts=dict(Counter(r['status'] for r in rows)),
                complete=not any(r['status'] in ('missing','pending','ambiguous_match') for r in rows),
                passed=all(r['status'] in ('within_tolerance','expected_failure') for r in rows),comparisons=rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',type=Path,default=Path(__file__).resolve().parent/'validation/paper_reference.json.gz')
    parser.add_argument('--output',type=Path,default=Path('results'))
    parser.add_argument('--tolerance-pp',type=float,default=.5)
    args=parser.parse_args()
    if not math.isfinite(args.tolerance_pp) or args.tolerance_pp<0:parser.error('tolerance must be finite and nonnegative')
    report=compare(read_reference(args.reference),args.output,args.tolerance_pp)
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='comparisons'},indent=2))
    raise SystemExit(0 if report['passed'] else 1)


if __name__=='__main__':main()

"""Compare fresh runs against paper endpoint references; missing runs never pass."""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

RUNTIME={'device','data_root','output_root','run_name','num_workers','log_every','save_model','eval_every','eval_every_epochs','epochs'}
ER_FIELDS={'er_solver','ridge_lambda','ridge_relative','ridge_strategy','solve_dtype','er_mix','secant_clip','secant_small','secant_small_series','forward_recompute','cholesky_retries','diag_every','diag_every_epochs','log_er_layers','diag_schedule'}


def key(cfg,loss):
    cfg={k:v for k,v in cfg.items() if k not in RUNTIME}
    if cfg['optimizer']=='er':cfg['optimizer']='er1'
    er=cfg['optimizer'] in ('er0','er1','pcg32_10','tp','dtp')
    for field in (['lr','momentum','weight_decay'] if er else ER_FIELDS):cfg.pop(field,None)
    if cfg['optimizer']!='momentum':cfg.pop('momentum',None)
    if er:loss='original'
    return json.dumps([cfg,loss],sort_keys=True)


def load_fresh(output):
    found={}
    for rd in sorted((output/'runs').glob('*')):
        if not (rd/'status.json').exists():continue
        state=json.loads((rd/'status.json').read_text())['status']
        if (rd/'record.json').exists():
            rec=json.loads((rd/'record.json').read_text())
            if rec.get('smoke'):continue
            cfg=rec['config'];loss=rec['loss'];p=rd/'metrics.json'
            raw=json.loads(p.read_text()) if p.exists() else []
            series=[dict(step=r['step'],accuracy=r['test_accuracy']) for r in raw]
        elif (rd/'config.json').exists():
            cfg=json.loads((rd/'config.json').read_text());loss='mse';p=rd/'eval.csv'
            with p.open() if p.exists() else open('/dev/null') as f:series=[r for r in csv.DictReader(f) if r['split']=='test']
        else:continue
        data=dict(run_id=rd.name,status=state,final_step=int(float(series[-1]['step'])) if series else None,final_accuracy=float(series[-1]['accuracy']) if series else None,best_accuracy=max(float(r['accuracy']) for r in series) if series else None)
        k=key(cfg,loss)
        # An identical MSE config may be produced by either execution core.
        found.setdefault(k,[]).append(data)
    return found


def compare(reference,output,tolerance_pp=0.5):
    fresh=load_fresh(output);rows=[]
    for ref in reference:
        cfg=ref['config'];candidates=fresh.get(key(cfg,ref['loss']),[])
        row=dict(group=ref['group'],dataset=cfg['dataset'],optimizer=cfg['optimizer'],seed=cfg['seed'],loss=ref['loss'],expected_status=ref['status'],status='missing')
        if candidates:
            # Report every backend; do not select the one closest to reference.
            for actual in candidates:
                result={**row,'run_id':actual['run_id'],'actual_status':actual['status']}
                if actual['status'] in ('running','interrupted'):result['status']='pending'
                elif actual['status']!=ref['status']:result['status']='status_mismatch'
                elif ref['status']=='failed':result['status']='expected_failure'
                elif actual['final_step']!=ref['final_step']:result['status']='budget_mismatch'
                else:
                    for metric in ['final_accuracy','best_accuracy']:
                        result[metric+'_delta_pp']=100*(actual[metric]-ref[metric])
                    result['status']='within_tolerance' if all(abs(result[m+'_delta_pp'])<=tolerance_pp for m in ['final_accuracy','best_accuracy']) else 'numerical_mismatch'
                rows.append(result)
        else:rows.append(row)
    counts=dict(Counter(r['status'] for r in rows))
    report=dict(tolerance_pp=tolerance_pp,counts=counts,complete=not any(r['status'] in ('missing','pending') for r in rows),passed=all(r['status'] in ('within_tolerance','expected_failure') for r in rows),comparisons=rows)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',type=Path,default=Path('validation/paper_reference.json'))
    p.add_argument('--output',type=Path,default=Path('results'))
    p.add_argument('--tolerance-pp',type=float,default=0.5)
    a=p.parse_args()
    if a.tolerance_pp<0:p.error('tolerance must be nonnegative')
    report=compare(json.loads(a.reference.read_text()),a.output,a.tolerance_pp)
    a.output.mkdir(parents=True,exist_ok=True)
    (a.output/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='comparisons'},indent=2))
    raise SystemExit(0 if report['passed'] else 1)


if __name__=='__main__':main()

"""Recompute the manuscript's active numerical tables from scalar run records."""
import json
import statistics
import math
from collections import defaultdict
from pathlib import Path
from . import style
from .common import entries, read_json, record_input, run_dir, validate_run_config
from .memory import _estimated_working_memory_bytes


def moments(values):
    if not values:return dict(mean=None,se=None,n=0)
    return dict(mean=statistics.mean(values),se=statistics.stdev(values)/math.sqrt(len(values)) if len(values)>1 else 0.,n=len(values))


def write_tables(data, results, output, checksums):
    results,output=Path(results),Path(output)
    summary=style.analyze(data,output)
    convergence=summary['acceleration']
    # Match the current main.tex column definitions: no speedup columns.
    lines=[r'\begin{tabular}{llr@{\,$\pm$\,}lr@{\,$\pm$\,}lc}',r'\toprule',
           r'Dataset & Method & \multicolumn{2}{c}{$T_{50}$} & \multicolumn{2}{c}{$T_{75}$} & Best accuracy (\%) \\',r'\midrule']
    for dataset in style.DATASETS:
        if dataset!=style.DATASETS[0]:lines.append(r'\midrule')
        group=[r for r in convergence if r['dataset']==dataset]
        for i,r in enumerate(group):
            label=style.LABELS[dataset] if i==len(group)//2 else ''
            t75=r'\multicolumn{2}{c}{Not reached}' if r['t75_mean'] is None else f"{r['t75_mean']:.1f} & {r['t75_se']:.1f}"
            lines.append(f"{label} & {style.LABELS[r['method']]} & {r['t50_mean']:.1f} & {r['t50_se']:.1f} & {t75} & {r['best_mean_pct']:.2f} "+r'$\pm$ '+f"{r['best_se_pct']:.2f}"+r' \\')
    (output/'acceleration_table.tex').write_text('\n'.join(lines+[r'\bottomrule',r'\end{tabular}'])+'\n')
    stable=defaultdict(list)
    for rec in entries(results,'table_stability_ablation'):
        validate_run_config(results,rec)
        cfg=rec['config'];rd=run_dir(results,rec);state=read_json(rd/'status.json')
        record_input(rd/'status.json',results,checksums)
        if state['status'] not in ('completed','failed'):raise ValueError(f'Unfinished stability trial {rd}')
        error=None
        if state['status']=='completed':
            path=rd/'summary.json';record_input(path,results,checksums);error=1-read_json(path)['final_test_accuracy']
        stable[cfg['dataset'],cfg['batch_size'],cfg['er_solver']].append(error)
    stability=[]
    for (dataset,batch,solver),values in stable.items():
        complete=[x for x in values if x is not None]
        if len(values)!=100:raise ValueError('Each stability cell requires 100 seeds')
        stability.append(dict(dataset=dataset,batch_size=batch,solver=solver,failures=len(values)-len(complete),total=len(values),**moments(complete)))
    lookup={(r['dataset'],r['batch_size'],r['solver']):r for r in stability}
    lines=[r'\begin{tabular}{llrcccc}',r'\toprule',r'Dataset & Regime & $n$ & pinv failures & pinv test error & Ridge failures & Ridge test error \\',r'\midrule']
    def error_text(row):return '---' if row['mean'] is None else f"${row['mean']:.4f} "+r'\pm '+f"{row['se']:.5f}$"
    for dataset in style.DATASETS:
        if dataset!=style.DATASETS[0]:lines.append(r'\midrule')
        for batch,regime in ([(4096,'Overdetermined'),(1024,'Underdetermined')] if dataset=='cifar10' else [(2048,'Overdetermined'),(512,'Underdetermined')]):
            pinv=lookup[dataset,batch,'pinv'];ridge=lookup[dataset,batch,'ridge']
            lines.append(f"{style.LABELS[dataset]} & {regime} & {batch} & {pinv['failures']}/100 & {error_text(pinv)} & {ridge['failures']}/100 & {error_text(ridge)}"+r' \\')
    (output/'stability_table.tex').write_text('\n'.join(lines+[r'\bottomrule',r'\end{tabular}'])+'\n')
    timings=defaultdict(list);configs={}
    for rec in entries(results,'fig2_optimizer_comparison'):
        cfg=rec['config'];method=cfg['optimizer']
        if method not in ('er1','sgd','adam'):continue
        path=run_dir(results,rec)/'summary.json';record_input(path,results,checksums)
        raw=read_json(path)
        if raw['status']!='completed':raise ValueError(f'Incomplete timing run {path}')
        timings[cfg['dataset'],method].append(raw['mean_wall_ms_per_step']);configs[cfg['dataset'],method]=cfg
    costs=[]
    for dataset in style.DATASETS:
        sgd_mean=statistics.mean(timings[dataset,'sgd'])
        for method in ('er1','sgd','adam'):
            values=timings[dataset,method]
            if len(values)!=10:raise ValueError('Each timing cell requires ten seeds')
            cfg=configs[dataset,method];memory=_estimated_working_memory_bytes(dataset,method,cfg['batch_size'],cfg['hidden1'],cfg['hidden2'],cfg['num_classes'])
            sgd_memory=_estimated_working_memory_bytes(dataset,'sgd',cfg['batch_size'],cfg['hidden1'],cfg['hidden2'],cfg['num_classes'])
            m=moments(values);costs.append(dict(dataset=dataset,method=method,**m,time_relative=m['mean']/sgd_mean,memory_mib=memory/1024**2,memory_relative=memory/sgd_memory))
    lines=[r'\begin{tabular}{llrrrr}',r'\toprule',r'Dataset & Method & Time / iteration (ms) & Time $\times$ SGD & Est. memory (MiB) & Memory $\times$ SGD \\',r'\midrule']
    for i,r in enumerate(costs):
        if i and i%3==0:lines.append(r'\midrule')
        lines.append(f"{style.LABELS[r['dataset']]} & {style.LABELS[r['method']]} & {r['mean']:.3f} "+r'$\pm$ '+f"{r['se']:.3f} & {r['time_relative']:.2f} & {r['memory_mib']:.1f} & {r['memory_relative']:.2f}"+r' \\')
    (output/'computational_cost_table.tex').write_text('\n'.join(lines+[r'\bottomrule',r'\end{tabular}'])+'\n')
    style.write(output/'stability_aggregated.csv',stability);style.write(output/'computational_cost_aggregated.csv',costs)
    tables=dict(acceleration=convergence,stability=stability,computational_cost=costs)
    (output/'paper_tables.json').write_text(json.dumps(tables,indent=2,allow_nan=False)+'\n')
    return summary

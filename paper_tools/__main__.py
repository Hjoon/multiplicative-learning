"""Notebook-free paper reproduction and checkpoint-free local preservation."""
import argparse
import json
from pathlib import Path
from .archive import create_archive


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['render','tables','archive'])
    parser.add_argument('--results',type=Path,default=Path('results'))
    parser.add_argument('--output',type=Path,default=Path('results/paper'))
    parser.add_argument('--destination',type=Path)
    args=parser.parse_args()
    if args.command=='archive':
        if args.destination is None:parser.error('archive requires --destination')
        print(json.dumps(create_archive(args.results,args.destination),indent=2));return
    import matplotlib
    matplotlib.use('Agg')
    from .data import load_paper_data
    from .tables import write_tables
    checksums={};args.output.mkdir(parents=True,exist_ok=True)
    data=load_paper_data(args.results,checksums)
    write_tables(data,args.results,args.output,checksums)
    names=[]
    if args.command=='render':
        from . import style
        from .diagnostics import plot_sensitivity,plot_stability
        style.plot_all(data,args.output)
        plot_sensitivity(args.results,args.output,checksums)
        plot_stability(args.results,args.output,checksums)
        names=['Fig2','Fig3','FigA1','FigA2','FigA3','FigA4']
    (args.output/'input_checksums.json').write_text(json.dumps(checksums,indent=2)+'\n')
    (args.output/'render_manifest.json').write_text(json.dumps(dict(numerical_backend='scalar-individual',
        figures=names,figure_roles=dict(FigA3='layerwise numerical stability',FigA4='TP versus DTP'),
        base_runs=180,control_runs=240,tp_dtp_runs=60,input_files=len(checksums),
        conceptual_Fig1='not regenerated; retain the manuscript diagram'),indent=2)+'\n')
    print(f'Paper {args.command} completed: {args.output}')


if __name__=='__main__':main()

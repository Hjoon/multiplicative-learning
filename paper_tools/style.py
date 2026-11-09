"""Numerical summaries and publication style for the adopted scalar paper."""
import csv
import json
import math
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy import stats

DATASETS=('mnist','fashionmnist','cifar10')
METHODS=('er0','er1','sgd','momentum','rmsprop','adam')
GD=('sgd','momentum','rmsprop','adam')
COLORS={'er0':'#A65628','er1':'#E41A1C','sgd':'#377EB8','momentum':'#FF7F00','rmsprop':'#4DAF4A','adam':'#984EA3'}
LABELS={'er0':'ER-0','er1':'ER-1','sgd':'SGD','momentum':'SGD+Momentum','rmsprop':'RMSProp','adam':'Adam',
        'mnist':'MNIST','fashionmnist':'Fashion-MNIST','cifar10':'CIFAR-10 RGB'}

def csvrows(path):
    with Path(path).open() as f:return list(csv.DictReader(f))

def write(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)

def series(rows,split,metric):
    q=[r for r in rows if r['split']==split];return np.array([int(float(r['step'])) for r in q]),np.array([float(r[metric]) for r in q])

def validate_series(x,y,final):
    assert len(x)==len(y) and x[0]==0 and x[-1]==final and np.all(np.diff(x)>0)
    assert np.isfinite(y).all()


def first_crossing(steps,errors,target):
    found=np.flatnonzero(np.asarray(errors)<=target)
    return int(steps[found[0]]) if len(found) else None

def ratio(adam,method):
    return adam/method if adam is not None and method is not None and method>0 else None

def mediqr(values):
    x=np.asarray([v for v in values if v is not None],float)
    if not len(x):return dict(n=0,median=None,q25=None,q75=None,mean=None,se=None)
    return dict(mean=float(x.mean()),se=float(x.std(ddof=1)/np.sqrt(len(x))) if len(x)>1 else 0.,n=len(x),median=float(np.median(x)),q25=float(np.quantile(x,.25)),q75=float(np.quantile(x,.75)))

def holm(ps):
    ps=np.asarray(ps,float);order=np.argsort(ps);out=np.empty(len(ps));running=0
    for i,k in enumerate(order):running=max(running,(len(ps)-i)*ps[k]);out[k]=min(1.,running)
    return out

def paired(values):
    v=np.asarray(values,float);n=len(v);mean=float(v.mean());sd=float(v.std(ddof=1));se=sd/np.sqrt(n)
    T=mean/se if se>0 else (0. if mean==0 else math.copysign(math.inf,mean))
    p=float(2*stats.t.sf(abs(T),n-1));ci=stats.t.ppf(.975,n-1)*se
    return dict(n=n,df=n-1,mean_difference_pp=100*mean,sd_difference_pp=100*sd,se_difference_pp=100*se,
                standardized_t=T,p_value=p,ci_low_pp=100*(mean-ci),ci_high_pp=100*(mean+ci))

def texnum(x,fmt='.2f'):
    return '--' if x is None else format(x,fmt)

def intervaltext(row,prefix):
    if row[prefix+'_n']==0:return 'Not reached'
    value=row[prefix+'_mean'];se=row[prefix+'_se']
    if prefix.startswith('s'):
        text=r'$\times '+f'{value:.3g}'+(r' \pm '+f'{se:.3g}' if row['method']!='adam' else '')+'$'
    else:text=f'{value:.1f}'+r' $\pm$ '+f'{se:.1f}'
    return text+(f" ({row[prefix+'_n']}/10)" if row[prefix+'_n']<10 else '')

def acceleration_tex(rows,main):
    selected=[r for r in rows if not main or r['method'] in ('er1','sgd','adam')]
    lines=[r'\begin{tabular}{llrrrrr}',r'\toprule',r'Dataset & Method & $T_{50}$ & $S_{50}$ & $T_{75}$ & $S_{75}$ & Best test (\%) \\',r'\midrule']
    for d in DATASETS:
        group=[r for r in selected if r['dataset']==d]
        if d!=DATASETS[0]:lines.append(r'\midrule')
        for i,r in enumerate(group):
            label=LABELS[d] if i==len(group)//2 else ''
            lines.append(' & '.join([label,LABELS[r['method']],intervaltext(r,'t50'),intervaltext(r,'s50'),intervaltext(r,'t75'),intervaltext(r,'s75'),f"{r['best_mean_pct']:.2f} $\\pm$ {r['best_se_pct']:.2f}"])+r' \\')
    return '\n'.join(lines+[r'\bottomrule',r'\end{tabular}'])+'\n'

def analyze(data,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);raw=[];agg=[];base=data['base']
    for d in DATASETS:
        for seed in range(10):
            xa,aa=series(base[d,'adam',seed],'test','accuracy');e0=1-aa[0];eb=1-aa.max()
            for m in METHODS:
                x,a=series(base[d,m,seed],'test','accuracy');row=dict(dataset=d,method=m,seed=seed,initial_adam_error=e0,best_adam_error=eb,
                  best_test_accuracy=float(a.max()),best_test_step=int(x[np.argmax(a)]),final_test_accuracy=float(a[-1]),final_step=int(x[-1]))
                for q,name in [(.5,'50'),(.75,'75')]:
                    target=1-q*(1-eb);tm=first_crossing(x,1-a,target);ta=first_crossing(xa,1-aa,target)
                    row.update({f'target_error_{name}':target,f't{name}':tm,f'adam_t{name}':ta,f'speedup{name}':ratio(ta,tm),f'reached{name}':tm is not None,f'zero_crossing{name}':tm==0})
                raw.append(row)
    for d in DATASETS:
        for m in METHODS:
            rr=[r for r in raw if (r['dataset'],r['method'])==(d,m)];r=dict(dataset=d,method=m,total_seeds=10)
            for name in ('50','75'):
                for key,field in [('t','t'),('s','speedup')]:
                    st=mediqr([v[field+name] for v in rr]);r.update({key+name+'_'+k:v for k,v in st.items()})
                r['not_reached'+name]=sum(v['t'+name] is None for v in rr)
            for key,field in [('best','best_test_accuracy'),('final','final_test_accuracy')]:
                vals=np.array([v[field] for v in rr]);r[key+'_mean_pct']=100*vals.mean();r[key+'_sd_pct']=100*vals.std(ddof=1);r[key+'_se_pct']=r[key+'_sd_pct']/np.sqrt(len(vals))
            for name in ('50','75'):
                valid=[v for v in rr if v['t'+name] is not None and v['t'+name]>0 and v['adam_t'+name] is not None]
                if valid:
                    a=np.array([v['adam_t'+name] for v in valid],float);t=np.array([v['t'+name] for v in valid],float)
                    ratio_mean=float(a.mean()/t.mean());influence=(a-ratio_mean*t)/t.mean()
                    r['s'+name+'_mean']=ratio_mean;r['s'+name+'_se']=float(influence.std(ddof=1)/np.sqrt(len(valid))) if len(valid)>1 else 0.
            agg.append(r)
    comparisons=[];seed_differences=[]
    for d in DATASETS:
        for m in GD:
            vals={'best':[],'final':[]}
            for seed in range(10):
                x,a=series(data['controls'][d,m,'mse',seed],'test','accuracy');y,b=series(data['controls'][d,m,'ce',seed],'test','accuracy');assert np.array_equal(x,y)
                row=dict(dataset=d,method=m,seed=seed,best_mse=float(a.max()),best_ce=float(b.max()),final_mse=float(a[-1]),final_ce=float(b[-1]),final_step=int(x[-1]))
                for kind in vals:
                    diff=row[kind+'_ce']-row[kind+'_mse'];vals[kind].append(diff);row[kind+'_difference_pp']=diff*100
                seed_differences.append(row)
            for kind,v in vals.items():
                r=dict(dataset=d,method=m,comparison=kind,evaluation_status=data.get('ce_status','unknown'),**paired(v));comparisons.append(r)
    for kind in ('best','final'):
        rr=[r for r in comparisons if r['comparison']==kind]
        for r,p in zip(rr,holm([r['p_value'] for r in rr])):r['holm_p_value']=float(p)
    tpdiffs=[]
    for d in DATASETS:
        diffs=[]
        for seed in range(10):
            x,a=series(data['tp'][d,'tp',seed],'test','accuracy');y,b=series(data['tp'][d,'dtp',seed],'test','accuracy')
            if x[-1]==y[-1]:diffs.append(b[-1]-a[-1])
        tpdiffs.append(dict(dataset=d,**paired(diffs),tp_failed=sum(data['tpstates'][d,'tp',s]['status']=='failed' for s in range(10)),dtp_failed=sum(data['tpstates'][d,'dtp',s]['status']=='failed' for s in range(10))))
    write(out/'acceleration_per_seed.csv',raw);write(out/'acceleration_aggregated.csv',agg)
    write(out/'mse_ce_per_seed.csv',seed_differences);write(out/'mse_ce_paired.csv',comparisons);write(out/'tp_dtp_final.csv',tpdiffs)
    write(out/'data_provenance.csv',data['provenance'])
    (out/'acceleration_main.tex').write_text(acceleration_tex(agg,True));(out/'acceleration_full.tex').write_text(acceleration_tex(agg,False))
    lines=[r'\begin{tabular}{llrr}',r'\toprule',r'Dataset & Optimizer & $\Delta \pm$ 95\% CI (pp) & Confidence \\',r'\midrule']
    for d in DATASETS:
        if d!=DATASETS[0]:lines.append(r'\midrule')
        group=[r for r in comparisons if r['comparison']=='best' and r['dataset']==d]
        for i,r in enumerate(group):
            stars='***' if r['holm_p_value']<.001 else '**' if r['holm_p_value']<.01 else '*' if r['holm_p_value']<.05 else '--'
            half=(r['ci_high_pp']-r['ci_low_pp'])/2
            lines.append(f"{LABELS[d] if i==len(group)//2 else ''} & {LABELS[r['method']]} & {r['mean_difference_pp']:+.3f} "+r'$\pm$ '+f"{half:.3f} & {stars}"+r' \\')
    (out/'mse_ce_paired.tex').write_text('\n'.join(lines+[r'\bottomrule',r'\end{tabular}'])+'\n')
    result=dict(acceleration=agg,mse_ce=comparisons,tp_dtp=tpdiffs,validation={'fig2_runs':180,'mse_ce_runs':240,'tp_dtp_runs':60,'initial_accuracy_max_spread':0,'paired_seeds':10,'ce_input':'pre-tanh logits; no explicit softmax','ce_accuracy_status':data.get('ce_status','unknown')})
    (out/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    (out/'methodology.md').write_text('''# Paper analysis definitions

For each dataset/seed, thresholds are 1 - q times Adam best observed test accuracy, q=.5 and .75 only. Crossing is the first RAW observed iteration
with method error <= threshold. No interpolation/smoothing/binning is used in
crossings. Sparse logging yields first-observed crossing, not exact first crossing.
Seed-wise speedup is Adam crossing / method crossing, ratio of mean crossing times, with paired delta-method SE. Unreached seeds remain missing, with reached/10 displayed; step-zero
crossings yield undefined division ratios and are separately recorded. Original
six-method paired seed initial test accuracies are exactly identical.

Acceleration best accuracy is mean +/- SE across seed-wise maxima. Final accuracy
is retained separately at each method's fixed last iteration. Method horizons
are intentionally different, as in the verified Fig.2 cases. These thresholds
are retrospective because Adam best test performance uses the whole trajectory;
they do not define a deployment stopping rule or measure wall-clock efficiency.

MSE/CE primary exploratory endpoint: difference of each seed's best observed test
accuracy, CE-MSE. Paired T=mean(D)/(SD(D)/sqrt(10)), df=9; two-sided t p and t 95%
CI. Holm adjustment is applied to the family of 12 best-test comparisons. A second,
separate 12-comparison family reports fixed-final endpoint robustness. Differences
and CI are in percentage points. Paired t assumptions and the small seed count
limit interpretation; nonsignificance does not establish equivalence. Best-test
selection is exploratory and susceptible to selection bias. MSE/CE budgets and
initialization match within each pair, as do raw evaluation grids.

Display only: retain raw points <=2000; thereafter per-seed 0.1-decade temporal
bins, geometric centers, exact terminal point separately retained. Across-seed
mean +/- SE is then computed. Log y masks zeros rather than substituting fake
errors. All colors and dataset/method membership are manifest-derived/validated.
''')
    return result

def binned(rows,split,metric,transform=lambda y:y):
    x,y=series(rows,split,metric);y=transform(y);early=x<=2000;xx=list(x[early]);yy=list(y[early])
    late=x>2000
    if late.any():
        last=x[-1];interior=late&(x!=last);bins=np.floor((np.log10(x[interior])-np.log10(2000))*10).astype(int)
        for b in np.unique(bins):xx.append(2000*10**((b+.5)/10));yy.append(y[interior][bins==b].mean())
        xx.append(last);yy.append(y[-1])
    order=np.argsort(xx);return np.asarray(xx)[order],np.asarray(yy)[order]

def draw(ax,runs,split,metric,color,transform=lambda y:y,ls='-'):
    pp=[binned(r,split,metric,transform) for r in runs];x=pp[0][0]
    assert all(np.array_equal(x,p[0]) for p in pp)
    ys=np.stack([p[1] for p in pp]);mean=ys.mean(0);se=ys.std(0,ddof=1)/np.sqrt(len(ys))
    ax.plot(x,mean,color=color,ls=ls,lw=1.6);ax.fill_between(x,mean-se,mean+se,color=color,alpha=.14,lw=0)

# Manuscript STIX body font, 10 pt; fixed physical axes-box aspect ratio.
FIGSIZE=(6.843,6.5)
FONT_PT=10
PLOT_ORDER=('er1','sgd','adam','er0','momentum','rmsprop')
COLORS.update(er1='#D62728',sgd='#1F77B4',adam='#FF7F0E',er0='#2CA02C',momentum='#9467BD',rmsprop='#8C564B')

def publication_style():
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'STIXGeneral','mathtext.fontset':'stix','font.size':FONT_PT,
      'axes.labelsize':FONT_PT,'axes.titlesize':FONT_PT,'xtick.labelsize':FONT_PT,'ytick.labelsize':FONT_PT,
      'legend.fontsize':FONT_PT,'legend.title_fontsize':FONT_PT,'figure.titlesize':FONT_PT,
      'pdf.fonttype':42,'ps.fonttype':42,'figure.dpi':100,'axes.spines.top':False,'axes.spines.right':False})

def xstyle(ax,maximum):
    ax.set_xscale('symlog',linthresh=1,linscale=.8,base=10);ax.set_xlim(0,maximum*1.03)
    ticks=[0]+[10**k for k in range(0,int(np.floor(np.log10(maximum)))+1)]
    ax.set_xticks(ticks);ax.set_xticklabels(['0' if x==0 else rf'$10^{{{int(round(np.log10(x)))}}}$' for x in ticks])
    labels=ax.get_xticklabels()
    for label in labels:label.set_ha('center')
    if not any(p.get_gid()=='initial-linear-region' for p in ax.patches):
        p=ax.axvspan(0,1,color='.7',alpha=.10,lw=0,zorder=0);p.set_gid('initial-linear-region')
    ax.set_xlabel('Iteration');ax.grid(alpha=.12,lw=.5)

# Axes width/height ratios: sqrt(2)-like for 1–2 rows, golden ratio for 3–4.
AXES_WIDTH_HEIGHT=1.4142
DENSE_AXES_WIDTH_HEIGHT=1.618

def layout_panel_boxes(fig, height_scale=1.0):
    """Solve the canvas height from rendered decorations, never row multiples.

    Measure each row/column's labels/titles in inches, reserve figure legends,
    then give every axes an equal physical box at the requested aspect ratio.
    Iterate because tick locators and text extents can change with box size.
    """
    width=6.843;pad=.085;gap=.10
    specs=[ax.get_subplotspec() for ax in fig.axes]
    rows=max(s.rowspan.stop for s in specs);cols=max(s.colspan.stop for s in specs)
    box_aspect=height_scale/(DENSE_AXES_WIDTH_HEIGHT if rows in (3,4) else AXES_WIDTH_HEIGHT)
    for ax in fig.axes:ax.set_box_aspect(box_aspect)
    for iteration in range(12):
        fig.canvas.draw();renderer=fig.canvas.get_renderer();dpi=fig.dpi
        left=np.zeros(cols);right=np.zeros(cols);top=np.zeros(rows);bottom=np.zeros(rows)
        for ax,spec in zip(fig.axes,specs):
            box=ax.get_window_extent(renderer);tight=ax.get_tightbbox(renderer)
            c=spec.colspan.start;r=spec.rowspan.start
            left[c]=max(left[c],(box.x0-tight.x0)/dpi)
            right[c]=max(right[c],(tight.x1-box.x1)/dpi)
            bottom[r]=max(bottom[r],(box.y0-tight.y0)/dpi)
            top[r]=max(top[r],(tight.y1-box.y1)/dpi)
        legend_height=max([l.get_window_extent(renderer).height/dpi for l in fig.legends]+[0.])
        legend_space=legend_height+gap if fig.legends else 0.
        box_width=(width-2*pad-sum(left+right)-(cols-1)*gap)/cols
        if box_width<=0:raise ValueError('Labels leave no plotting area')
        box_height=box_width*box_aspect
        height=2*pad+legend_space+sum(top+bottom)+rows*box_height+(rows-1)*gap
        old=np.array(fig.get_size_inches());fig.set_size_inches(width,height)
        xs=[];x=pad
        for c in range(cols):xs.append(x+left[c]);x+=left[c]+box_width+right[c]+gap
        ys=[];y=height-pad-legend_space
        for r in range(rows):ys.append(y-top[r]-box_height);y-=top[r]+box_height+bottom[r]+gap
        for ax,spec in zip(fig.axes,specs):
            ax.set_position([xs[spec.colspan.start]/width,ys[spec.rowspan.start]/height,box_width/width,box_height/height])
        for l in fig.legends:l.set_bbox_to_anchor((.5,1-pad/height),transform=fig.transFigure)
        if iteration>=3 and np.max(np.abs(old-[width,height]))<1e-5:break
    fig.canvas.draw()
    ratios=[ax.get_window_extent().height/ax.get_window_extent().width for ax in fig.axes]
    assert np.allclose(ratios,box_aspect,atol=1e-6),ratios
    return ratios

def save_publication(fig,out,name):
    import matplotlib.pyplot as plt
    from matplotlib.text import Text
    for item in fig.findobj(Text):item.set_fontsize(FONT_PT);item.set_fontfamily('STIXGeneral')
    shared_label=getattr(fig,'_supylabel',None)
    if shared_label is not None:
        fig.axes[0].set_ylabel(shared_label.get_text())
        shared_label.remove();fig._supylabel=None
    ratios=layout_panel_boxes(fig,height_scale=1.03 if name in ('Fig2','Fig3') else 1.0)
    legend_font=FONT_PT;legend_height_ratio=None;legend_text_ids=set()
    if name=='Fig3':
        original_legends=[]
        for ax in fig.axes:
            leg=ax.get_legend()
            original_legends.append((ax,list(leg.legend_handles if hasattr(leg,'legend_handles') else leg.legendHandles),[t.get_text() for t in leg.get_texts()]))
        # Largest common legend size, to 0.1 pt, whose (a) box is <= half its axes.
        for size in np.arange(FONT_PT,0.9,-.1):
            for ax,handles,labels in original_legends:
                ax.legend(handles,labels,loc='lower right',frameon=True,fontsize=float(size))
            fig.canvas.draw();renderer=fig.canvas.get_renderer()
            legend_height_ratio=fig.axes[0].get_legend().get_window_extent(renderer).height/fig.axes[0].get_window_extent(renderer).height
            if legend_height_ratio<=.5:
                legend_font=float(size);break
        assert legend_height_ratio<=.5
        legend_text_ids={id(t) for ax in fig.axes for t in ax.get_legend().get_texts()}
    visible=[t for t in fig.findobj(Text) if t.get_visible() and t.get_text()]
    assert all(t.get_fontsize()==FONT_PT for t in visible if id(t) not in legend_text_ids)
    audit=Path(out)/'figure_style_validation.json'
    records=json.loads(audit.read_text()) if audit.exists() else {}
    records[name]={'figsize_inches':list(fig.get_size_inches()),'axes_height_width_ratios':ratios,'font_pt':FONT_PT,'font_family':'STIXGeneral','visible_text_count':len(visible),'legend_font_pt':legend_font,'panel_a_legend_height_fraction':legend_height_ratio}
    audit.write_text(json.dumps(records,indent=2))
    for ext in ('pdf','png'):fig.savefig(Path(out)/(name+'.'+ext),dpi=350,bbox_inches=None)
    plt.close(fig)

def plot_all(data,out):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    publication_style();out=Path(out);saved=[]
    def legend(fig,methods,ncol=3):
        fig.legend(handles=[Line2D([0],[0],color=COLORS[m],lw=1.6,label=LABELS[m]) for m in [methods[i] for col in range(ncol) for i in range(col,len(methods),ncol)]],
                   ncol=ncol,loc='upper center',bbox_to_anchor=(.5,1),frameon=False)
    def finish(fig,name):
        fig.tight_layout(rect=(0,0,1,.91),pad=.65,h_pad=.65,w_pad=.6)
        save_publication(fig,out,name);saved.append(name)
    for name,rows,methods in [('Fig2',2,PLOT_ORDER[:4]),('FigA1',4,PLOT_ORDER)]:
        fig,axes=plt.subplots(rows,3,figsize=FIGSIZE,sharey=False)
        # Accuracy pairs share within dataset; MSE panels remain independent.
        for col in range(3):axes[1,col].sharey(axes[0,col])
        for row in range(rows):
            metric='accuracy' if row<2 else 'loss';split='train' if row%2==0 else 'test'
            for col,d in enumerate(DATASETS):
                ax=axes[row,col];maximum=0
                for m in methods:
                    runs=[data['base'][d,m,s] for s in range(10)];draw(ax,runs,split,metric,COLORS[m]);maximum=max(maximum,int(float(runs[0][-1]['step'])))
                xstyle(ax,maximum)
                if metric=='accuracy':ax.set_ylim(0,1)
                else:ax.set_yscale('log')
                if name=='FigA1' and row==3 and col==2:
                    from matplotlib.ticker import FixedLocator, NullFormatter, LogFormatterMathtext
                    ax.set_ylim(bottom=.1)
                    ax.yaxis.set_major_locator(FixedLocator([.1]))
                    ax.yaxis.set_major_formatter(LogFormatterMathtext())
                    ax.yaxis.set_minor_locator(FixedLocator(np.arange(.2,1.,.1)))
                    ax.yaxis.set_minor_formatter(NullFormatter())
                if row==0:ax.set_title(LABELS[d])
                if row<rows-1:ax.set_xlabel('');ax.tick_params(axis='x',labelbottom=False)
                if col==0:ax.set_ylabel(('Train' if split=='train' else 'Test')+(' accuracy' if metric=='accuracy' else ' MSE'))
        if name=='FigA1':
            for a in axes[2:].flat:
                assert not any(a.get_shared_y_axes().joined(a,b) for b in axes[2:].flat if b is not a)
        legend(fig,methods,ncol=4 if name=='Fig2' else 3);finish(fig,name)
    fig,axes=plt.subplots(4,3,figsize=FIGSIZE,sharey=False)
    for row,m in enumerate(GD):
        for col,d in enumerate(DATASETS):
            ax=axes[row,col]
            for loss,color in [('mse',COLORS['sgd']),('ce',COLORS['adam'])]:
                runs=[data['controls'][d,m,loss,s] for s in range(10)];draw(ax,runs,'test','accuracy',color)
            xstyle(ax,int(float(runs[0][-1]['step'])));ax.set_ylim(0,1);ax.set_yscale('linear')
            if row==0:ax.set_title(LABELS[d])
            if col==0:ax.set_ylabel(LABELS[m]+'\nTest accuracy')
            if row<3:ax.set_xlabel('');ax.tick_params(axis='x',labelbottom=False)
    assert all(not a.get_shared_y_axes().joined(a,b) for a in axes.flat for b in axes.flat if a is not b)
    fig.legend(handles=[Line2D([0],[0],color=c,lw=1.6,label=l) for l,c in [('MSE',COLORS['sgd']),('CE',COLORS['adam'])]],
               ncol=2,loc='upper center',frameon=False)
    finish(fig,'FigA2')
    fig,axes=plt.subplots(1,3,figsize=FIGSIZE)
    for col,d in enumerate(DATASETS):
        ax=axes[col]
        for m,color in [('tp',COLORS['sgd']),('dtp',COLORS['er1'])]:
            runs=[data['tp'][d,m,s] for s in range(10)];draw(ax,runs,'test','accuracy',color)
        xstyle(ax,int(float(runs[0][-1]['step'])));ax.set_title(LABELS[d]);ax.set_ylim(0,1)
        if col==0:ax.set_ylabel('Test accuracy')
    axes[0].legend(handles=[Line2D([0],[0],color=c,lw=1.6,label=l) for l,c in [('TP',COLORS['sgd']),('DTP',COLORS['er1'])]],loc='lower right',frameon=True)
    finish(fig,'FigA4')
    return saved


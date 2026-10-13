import csv
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from ..truth import preimage
import numpy as np
from .common import RESULTS, RUNS

def save(fig,name):
    out=RESULTS/'figures';out.mkdir(exist_ok=True);fig.tight_layout()
    for ext in ('png','pdf','svg'):fig.savefig(out/f'{name}.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)

def learning():
    task_diagram()
    run=RUNS/'candidate_A_seed0'
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for sid,axrow in ((6,axes[0]),(7,axes[1])):
        path=run/f'stage_{sid}'/'curve.json'
        if not path.exists():continue
        rows=json.loads(path.read_text());step=[r['step'] for r in rows]
        for split,color in (('T','#94a3b8'),('V','#2563eb')):
            axrow[0].plot(step,[100*r[split+'_accuracy'] for r in rows],'.-',color=color,label=split+' overall')
        for k in range(sid-3):
            axrow[0].plot(step,[100*r['V_strata']['non_N'][str(k)]['accuracy'] for r in rows],label=f'V k={k}')
        axrow[0].axhline(97,c='black',ls='--',lw=1)
        axrow[0].set(xlabel='Stage updates (existing 5,000 included in stage 6)',ylabel='Accuracy (%)',
                     title=f'Stage {sid}: '+('33 words' if sid==6 else 'original 65 words; V n=8,192'))
        axrow[0].legend(fontsize=8)
        axrow[1].plot(step,[r['T_cross_entropy'] for r in rows],label='T')
        axrow[1].plot(step,[r['V_cross_entropy'] for r in rows],label='V')
        axrow[1].set(xlabel='Stage updates',ylabel='Cross entropy (nat)',title=f'Stage {sid} fixed T/V')
        axrow[1].legend()
    save(fig,'continuous_learning')
    rows=json.loads((RESULTS/'previous_stage6_curve.json').read_text())
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,selection,title in ((axes[0],rows,'Existing stage 6: all 500-step evaluations'),
                               (axes[1],[r for r in rows if r['step']>=3000],'Existing last 2,000 updates')):
        for label,key in (('T overall','T_accuracy'),('V overall','V_accuracy')):
            ax.plot([r['step'] for r in selection],[100*r[key] for r in selection],'.-',label=label)
        ax.plot([r['step'] for r in selection],[100*r['V_strata']['non_N']['2']['accuracy'] for r in selection],'.-',label='V k=2')
        ax.set(xlabel='Stage 6 updates',ylabel='Accuracy (%)',title=title);ax.legend()
    save(fig,'previous_curve_review')
    endpoint=run/'stage_6/two_step_diagnostics/endpoint_summary.json'
    if endpoint.exists():
        snapshots=[('restored step 5,000',RESULTS/'two_step_diagnostics/restore_step5000_summary.json'),('stage 6 endpoint',endpoint)]
        fig,axes=plt.subplots(1,2,figsize=(11,4))
        for ax,(name,path) in zip(axes,snapshots):
            result=json.loads(path.read_text())
            for j,split in enumerate(('T','V')):
                cells=result[split]['full_task']['strata']['true_operation_sequence']
                ops=('II','CC','IC','CI')
                ax.bar(np.arange(4)+(j-.5)*.36,[100*cells[op]['accuracy'] for op in ops],.36,label=split)
            ax.axhline(95,c='black',ls='--',lw=1);ax.set(xticks=range(4),xticklabels=ops,ylabel='Single-query accuracy (%)',title=name);ax.legend()
        save(fig,'two_step_before_after')

def read_csv(path):
    with path.open() as f:return list(csv.DictReader(f))

def formal():
    root=RESULTS/'prediction'
    if not (root/'full_distribution.csv').exists():return
    rows=read_csv(root/'full_distribution.csv');scores=read_csv(root/'scores.csv')
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for cross,color,label in ((0,'#2563eb','same context'),(1,'#dc2626','cross context')):
        group=[r for r in rows if int(float(r['cross_context']))==cross]
        for ax,key,title in ((axes[0],'loss_positive','Positive probability damage'),(axes[1],'new_error','New behavioral errors')):
            art=ax.scatter([float(r['cka']) for r in group],[float(r[key]) for r in group],s=6,alpha=.3,
                           c=[float(r['native_correct']) for r in group],cmap='viridis',vmin=0,vmax=1,
                           marker='o' if cross==0 else '^',label=label)
            ax.set(xlabel='Centered linear CKA on G',ylabel='L (nat)' if key=='loss_positive' else 'R',title=title)
            ax.legend()
    fig.colorbar(art,ax=axes,label='Native accuracy')
    save(fig,'cka_damage')
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    models=('M0','M_CKA','M_raw','M_proc','M_init','M_init_CKA')
    for ax,metric in zip(axes,('mae','rmse')):
        for seed in range(101,107):
            ax.plot(range(6),[float(next(r for r in scores if r['seed']==str(seed) and r['segment']=='unseen_cross' and r['model']==m)[metric]) for m in models],'.-',label=str(seed))
        ax.set(xticks=range(6),xticklabels=models,ylabel=metric.upper()+' (nat)',title='Held seed + unseen cross-context pairs')
        ax.tick_params(axis='x',rotation=25);ax.legend(fontsize=7)
    save(fig,'held_out_prediction')
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for ax,key,title in zip(axes,('new_error','loss_positive','wrong_cf_correct_different'),('New errors R','Probability damage L (nat)','Wrong-content counterfactual accuracy; distinct answers')):
        for d,offset,color in ((0,-.18,'#2563eb'),(10,.18,'#d97706')):
            groups=[[float(r[key]) for r in rows if int(float(r['d']))==d and int(float(r['cross_context']))==c] for c in (0,1)]
            ax.bar(np.arange(2)+offset,[np.mean(v) for v in groups],.35,label=f'd={d}',color=color)
            ax.set(xticks=(0,1),xticklabels=('same','cross'),title=title);ax.legend()
    save(fig,'damage_and_counterfactual')


def task_diagram():
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.set(xlim=(0, 12), ylim=(0, 4.5))
    ax.axis('off')
    q = np.array([2, 0, 1, 4, 3, 6, 5])
    pa, pb = preimage(q, 'AB', 0), preimage(q, 'NN', 1)
    def box(x, y, width, height, label, color):
        ax.add_patch(FancyBboxPatch((x, y), width, height, boxstyle='round,pad=0.1',
                                    facecolor=color, edgecolor='#475569'))
        ax.text(x+width/2, y+height/2, label, ha='center', va='center', fontsize=10)
    def arrow(a, b, color='#475569'):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle='->', mutation_scale=14, color=color))
    box(.2, 2.6, 3.0, 1.0, f'Donor: r=0, H=AB\np={pa.tolist()}', '#e0f2fe')
    box(.2, .8, 3.0, 1.0, f'Recipient: r=1, H=NN\np={pb.tolist()}', '#dcfce7')
    box(4., 2.7, 2.0, .8, 'Natural state\n(m_a(q), c_a)', '#e0f2fe')
    box(4., .9, 2.0, .8, 'Natural state\n(m_b(q), c_b)', '#dcfce7')
    arrow((3.3, 3.1), (3.9, 3.1))
    arrow((3.3, 1.3), (3.9, 1.3))
    ax.text(5, 2.05, f'Identical full content q\n{q.tolist()}', ha='center', fontsize=10)
    box(7., 1.8, 1.8, 1.0, 'Intervention\n(m_a(q), c_b)', '#fef3c7')
    arrow((6.1, 3.1), (7.0, 2.6), '#0284c7')
    arrow((6.1, 1.3), (7.0, 2.0), '#16a34a')
    box(9.6, 1.8, 2., 1., 'Recipient future\nS, then query', '#f1f5f9')
    arrow((8.9, 2.3), (9.5, 2.3))
    ax.text(6, .2, 'Swap after step 21+d; query first appears at step 35+d.  Only final-query CE trains the network.',
            ha='center', fontsize=10)
    ax.text(6, 4.1, 'Preserve recipient control, clock, future input and readout', ha='center', fontsize=14)
    save(fig, '01_task')


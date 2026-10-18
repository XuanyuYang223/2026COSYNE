"""Stage 1 only: editable, independent figures; no manuscript or PDF assembly.

All dimensions are inches. Font sizes stay fixed when width/height change;
layout checks reject clipping/overlap rather than shrink text. Reads CSV/JSON
only: no model imports, network execution, fits, or bootstrap.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import re
import xml.etree.ElementTree as ET

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.text import Text
import cairosvg
import pymupdf as fitz

HERE = Path(__file__).resolve().parent
FIG = HERE / 'figures'
ROOT = HERE.parents[2]
SEEDS = tuple(range(101, 107))
SIZES = {'A': (7.2, 7.5), 'B': (3.45, 3.5), 'C': (3.45, 4.1), 'D': (3.45, 4.45)}
WITHIN, CROSS, NATIVE = '#0072B2', '#D55E00', '#68549A'
INK, LIGHT = '#24282D', '#E4E8EC'
FONT = 12
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': FONT,
    'axes.labelsize': FONT, 'xtick.labelsize': FONT, 'ytick.labelsize': FONT,
    'legend.fontsize': FONT, 'axes.linewidth': .9, 'lines.linewidth': 1.5,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    'svg.hashsalt': 'cosyne-independent-stage1',
    'text.color': INK, 'axes.labelcolor': INK,
    'xtick.color': INK, 'ytick.color': INK, 'figure.facecolor': 'white',
})
DRAWN = {}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(name):
    with (HERE / 'data' / name).open() as f:
        return list(csv.DictReader(f))


def check_inputs():
    manifest = json.loads((HERE / 'data/source_manifest.json').read_text())['sha256']
    for name, digest in manifest.items():
        assert sha(ROOT / name) == digest, f'Historical source changed: {name}'
    old = json.loads((HERE / 'artifact_manifest.json').read_text())
    for name in ('panel_B_seed.csv', 'panel_C_seed.csv', 'panel_D_d0.csv',
                 'panel_D_medians.csv', 'actual_values.json'):
        key = 'data/' + name
        assert sha(HERE / key) == old[key]['sha256'], f'Plot input changed: {key}'
    return len(manifest)


def text_layout_check(fig):
    """Check actual text extents, including ticks and legend, in points."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bounds = fig.bbox
    labels = []
    for obj in fig.findobj(Text):
        if not obj.get_visible() or not obj.get_text().strip():
            continue
        assert obj.get_fontsize() >= FONT
        box = obj.get_window_extent(renderer)
        assert (box.x0 >= bounds.x0 + 1 and box.y0 >= bounds.y0 + 1
                and box.x1 <= bounds.x1 - 1 and box.y1 <= bounds.y1 - 1), (
                    'Text clipped; increase design dimensions', obj.get_text(), box.bounds)
        labels.append((obj.get_text(), box))
    for i, (a, box_a) in enumerate(labels):
        for b, box_b in labels[i + 1:]:
            w = min(box_a.x1, box_b.x1) - max(box_a.x0, box_b.x0)
            h = min(box_a.y1, box_b.y1) - max(box_a.y0, box_b.y0)
            assert w <= .7 or h <= .7, ('Overlapping text; increase dimensions', a, b)


def finish_pdf(panel):
    path = FIG / f'panel_{panel}.pdf'
    doc = fitz.open(path)
    doc.set_metadata({'title': 'Memory-state interchange', 'author': '',
                      'subject': 'Anonymous independent scientific figure',
                      'creator': 'Vector figure', 'producer': 'Vector figure',
                      'creationDate': '', 'modDate': '', 'keywords': ''})
    if doc.xref_xml_metadata():
        doc.del_xml_metadata()
    temp = FIG / f'.panel_{panel}.pdf'
    doc.save(temp, garbage=4, deflate=True)
    doc.close()
    temp.replace(path)
    doc = fitz.open(path)
    doc[0].get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False).save(FIG / f'panel_{panel}.png')
    return doc


def save(fig, panel):
    text_layout_check(fig)
    fig.savefig(FIG / f'panel_{panel}.pdf', facecolor='white',
                metadata={'Author': '', 'CreationDate': None, 'ModDate': None})
    fig.savefig(FIG / f'panel_{panel}.svg', facecolor='white',
                metadata={'Creator': 'Vector figure', 'Date': None})
    path = FIG / f'panel_{panel}.svg'
    path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines()) + '\n')
    plt.close(fig)
    finish_pdf(panel).close()


def canvas(width, height, bottom=52, top=22):
    assert width > 0 and height > 0
    w, h = width * 72, height * 72
    left, right = 52, 11
    assert w > left + right + 110 and h > bottom + top + 110
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_axes([left / w, bottom / h, (w - left - right) / w, (h - bottom - top) / h])
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(width=.9, length=3.5, pad=5)
    ax.grid(axis='y', color=LIGHT, linewidth=.6, zorder=0)
    return fig, ax


def schematic(width=7.2, height=7.5):
    # These examples appear only in captions.txt. Pure list checks, no models.
    q = [2, 4, 1, 7, 3, 6, 5]
    p_d, p_r = [3, 1, 5, 2, 7, 6, 4], [6, 4, 7, 1, 5, 2, 3]
    assert [p_d.index(j) + 1 for j in range(1, 8)] == q  # ctx0: A then N
    assert [8 - j for j in p_r] == q                   # ctx1: N then A
    assert p_d != p_r
    w, h = width * 72, height * 72
    sx, sy = w / 518.4, h / 540
    svg = ET.Element('svg', xmlns='http://www.w3.org/2000/svg',
                     width=f'{w}pt', height=f'{h}pt', viewBox=f'0 0 {w} {h}')
    defs = ET.SubElement(svg, 'defs')
    for name, color in [('flow', INK), ('memory', WITHIN)]:
        marker = ET.SubElement(defs, 'marker', id=name, markerWidth='7', markerHeight='7',
                               refX='6', refY='3.5', orient='auto', markerUnits='userSpaceOnUse')
        ET.SubElement(marker, 'path', d='M0,0 L7,3.5 L0,7 Z', fill=color)
    def txt(x, y, s, size=FONT, bold=False, anchor='start', color=INK):
        e = ET.SubElement(svg, 'text', x=str(x*sx), y=str(y*sy), fill=color,
            **{'font-family': 'DejaVu Sans', 'font-size': str(size),
               'font-weight': 'bold' if bold else 'normal', 'text-anchor': anchor})
        e.text = s
        return e
    def math(x, y, base, sub, size=14, anchor='middle', color=INK):
        # Explicit positions preserve real subscripts in Cairo and SVG editors.
        # Cairo does not consistently implement SVG baseline-shift.
        txt(x+1, y, base, size, anchor='end', color=color)
        return txt(x+2, y+size*.25, sub, size*.72, color=color)
    def arrow(x1, y1, x2, y2, memory=False):
        ET.SubElement(svg, 'line', x1=str(x1*sx), y1=str(y1*sy),
            x2=str(x2*sx), y2=str(y2*sy), stroke=WITHIN if memory else INK,
            **{'stroke-width': '1.6' if memory else '1.1',
               'marker-end': 'url(#memory)' if memory else 'url(#flow)'})
    def box(x, y, bw, bh, fill='white', stroke='#A4ADB5'):
        ET.SubElement(svg, 'rect', x=str(x*sx), y=str(y*sy), width=str(bw*sx), height=str(bh*sy),
                      rx='3', fill=fill, stroke=stroke, **{'stroke-width': '.9'})
    def state(x, y, base, sub, memory=False):
        box(x, y, 60, 32, '#E7F0F6' if memory else '#F5F5F5')
        math(x+30, y+21, base, sub)
    txt(24, 20, 'One fixed network · two trials', bold=True)
    # Separate rule table, no enclosing frame.
    for x, name in [(84, 'Rule context'), (280, 'Cue A'), (435, 'Cue B')]:
        txt(x, 48, name, bold=True, anchor='middle')
    ET.SubElement(svg, 'line', x1=str(24*sx), y1=str(56*sy), x2=str(495*sx), y2=str(56*sy),
                  stroke=LIGHT, **{'stroke-width': '.9'})
    for y, context, a, b in [(75, '0', 'inversion', 'complementation'),
                              (96, '1', 'complementation', 'inversion')]:
        for x, value in [(84, context), (280, a), (435, b)]:
            txt(x, y, value, anchor='middle')
    # 1: histories lead to the same task-defined content, not the same state.
    txt(24, 129, '1. Different inputs and histories, same q', size=13.5, bold=True)
    for x, name in [(135, 'Initial input'), (270, 'History'), (420, 'Current content')]:
        txt(x, 153, name, anchor='middle')
    for y, role, sub in [(181, 'Donor', 'D'), (219, 'Receiver', 'R')]:
        txt(24, y, role)
        math(135, y, 'p', sub)
        math(270, y-15, 'H', sub, size=13)
        arrow(171, y-4, 365, y-4)
        txt(420, y, 'q', size=14, anchor='middle')
    # 2: actual native components and a direct donor-memory -> receiver-slot arrow.
    txt(24, 255, '2. Exchange the memory component', size=13.5, bold=True)
    txt(24, 279, 'm = memory state')
    txt(265, 279, 'c = control state')
    txt(350, 303, 'Donor trial')
    state(350, 314, 'm', 'D', memory=True)
    state(420, 314, 'c', 'D')
    txt(24, 367, 'Receiver before')
    state(24, 379, 'm', 'R')
    state(94, 379, 'c', 'R')
    arrow(175, 395, 331, 395)
    txt(252, 382, 'replace', anchor='end')
    math(268, 382, 'm', 'R', size=12)
    txt(252, 418, 'keep', anchor='end')
    math(268, 418, 'c', 'R', size=12)
    state(350, 379, 'm', 'D', memory=True)
    state(420, 379, 'c', 'R')
    arrow(380, 346, 380, 378, memory=True)
    txt(373, 438, '(', size=14)
    math(392, 438, 'm', 'D')
    txt(408, 438, ',', size=14)
    math(430, 438, 'c', 'R')
    txt(447, 438, ')', size=14)
    # 3: receiver future inputs and original readout; no separate-network cartoon.
    txt(24, 475, '3. Continue the receiver trial', size=13.5, bold=True)
    for x, bw in [(24, 162), (232, 164), (439, 64)]:
        box(x, 493, bw, 37)
    txt(105, 508, 'Subsequent', anchor='middle')
    txt(105, 523, 'receiver inputs', anchor='middle')
    txt(314, 508, 'Original network', anchor='middle')
    txt(314, 523, 'readout', anchor='middle')
    txt(471, 516, 'Answer', anchor='middle')
    arrow(188, 511.5, 225, 511.5)
    arrow(398, 511.5, 432, 511.5)
    ET.indent(svg, space='  ')
    path = FIG / 'panel_A.svg'
    ET.ElementTree(svg).write(path, encoding='unicode', xml_declaration=True)
    cairosvg.svg2pdf(url=str(path), write_to=str(FIG / 'panel_A.pdf'))
    finish_pdf('A').close()
    DRAWN['A'] = {'width_in': width, 'height_in': height,
                  'ordinary_font_pt': 12, 'step_heading_pt': 13.5,
                  'math_base_pt': [12,13,14], 'math_subscript_pt': [8.64,9.36,10.08],
                  'math_example_checked': True,
                  'memory_arrow': 'donor m_D directly to receiver after-exchange memory slot'}


def behavior(width=3.45, height=3.5):
    rows = read('panel_B_seed.csv')
    fig, ax = canvas(width, height)
    means = []
    for cross, color in [(0, WITHIN), (1, CROSS)]:
        selected = sorted([r for r in rows if int(r['cross']) == cross], key=lambda r:int(r['seed']))
        assert [int(r['seed']) for r in selected] == list(SEEDS)
        vals = np.array([float(r['new_error_percent']) for r in selected])
        for r in selected:
            np.testing.assert_allclose(float(r['new_error_n'])/float(r['query_n'])*100,
                                       float(r['new_error_percent']), rtol=0, atol=1e-10)
        mean = float(vals.mean()); means.append(mean)
        ax.hlines(mean, cross-.16, cross+.16, color='black', linewidth=2.1, zorder=3)
        ax.scatter(cross+np.linspace(-.27,.27,6), vals, s=22, marker='o', color=color,
                   edgecolors='white', linewidths=.45, zorder=4)
        ax.text(cross, mean+4.8, f'{mean:.1f}%', ha='center', fontsize=13, weight='bold')
    ax.set(xlim=(-.58,1.58), ylim=(0,64), xticks=[0,1],
           xticklabels=['Within-\ncontext','Cross-\ncontext'], yticks=[0,20,40,60],
           ylabel='New errors (% of all queries)')
    expected = json.loads((HERE/'data/actual_values.json').read_text())
    np.testing.assert_allclose(means, [expected['B_0'],expected['B_1']], rtol=0,atol=1e-10)
    save(fig,'B')
    DRAWN['B'] = {'width_in':width,'height_in':height,'ordinary_font_pt':12,
                  'mean_label_pt':13,'networks_per_group':6,'points':12,
                  'marker':'circle only','mean_marker':'short black horizontal line',
                  'equal_seed_means_percent':means,'y_axis_min':0}


def translation(width=3.45, height=4.1):
    rows=read('panel_C_seed.csv')
    fig,ax=canvas(width,height,bottom=56,top=40)
    fig.text(.5,1-17/(height*72),'Held-out permutations',ha='center',fontsize=12)
    output={}
    for group,color,style in [('Cross-context exchange',CROSS,'-'),('No exchange',NATIVE,'--')]:
        values=[]
        for j,seed in enumerate(SEEDS):
            vals=[]
            for cue in ['original','translated']:
                matches=[r for r in rows if int(r['seed'])==seed and r['treatment']==group and r['cues']==cue]
                assert len(matches)==1
                vals.append(float(matches[0]['accuracy_percent']))
            values.append(vals)
            xs=np.array([0,1])+np.linspace(-.1,.1,6)[j]
            ax.plot(xs,vals,color=color,lw=.7,alpha=.5,zorder=3)
            ax.scatter(xs,vals,s=18,marker='o',color=color,edgecolors='white',linewidths=.25,zorder=5)
        means=np.mean(values,axis=0)
        output[group]={'seed_values_percent':values,'equal_seed_means_percent':means.tolist()}
        ax.plot([0,1],means,color=color,lw=2.3,ls=style,zorder=4)
        for x,val in enumerate(means):
            ax.text(x,val+(5 if val>80 else -10),f'{val:.1f}%',ha='center',
                    va='center',fontsize=12,weight='bold',color=color)
    ax.set(xlim=(-.28,1.28),ylim=(0,114),xticks=[0,1],
           xticklabels=['Original\ncues','Translated\ncues'],yticks=[0,50,100],
           ylabel='Original-task accuracy (%)')
    handles=[Line2D([0],[0],color=CROSS,lw=2.3,label='Cross-context exchange'),
             Line2D([0],[0],color=NATIVE,lw=2.3,ls='--',label='Native / no exchange')]
    ax.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.015),frameon=False,
              fontsize=12,handlelength=1.3,handletextpad=.5,borderaxespad=.2,labelspacing=.4)
    expected=json.loads((HERE/'data/actual_values.json').read_text())
    for group in output:
        np.testing.assert_allclose(output[group]['equal_seed_means_percent'],
            [expected[f'C_{group}_{cue}'] for cue in ['original','translated']],rtol=0,atol=1e-10)
    save(fig,'C')
    DRAWN['C']={'width_in':width,'height_in':height,'ordinary_font_pt':12,
                'networks_per_group':6,'paired_seed_lines':12,'seed_points':24,
                'marker':'circle only','data':output,'all_suffixes_include_NN':True,
                'no_exchange_source':'true unpatched native trials'}


def geometry(width=3.45,height=4.45):
    rows=read('panel_D_d0.csv')
    assert len(rows)==918 and {int(r['d']) for r in rows}=={0}
    keys=[(int(r['seed']),int(r['a']),int(r['b']),int(r['d'])) for r in rows]
    assert len(set(keys))==len(keys) and all(k[1]<k[2] for k in keys)
    fig,ax=canvas(width,height,bottom=80,top=62)
    medians=[]
    for cross,color,style,label,n in [(0,WITHIN,'-','Within-context',432),
                                      (1,CROSS,'--','Cross-context',486)]:
        selected=[r for r in rows if int(r['cross'])==cross]
        assert len(selected)==n
        assert all(sum(int(r['seed'])==s for r in selected)==n//6 for s in SEEDS)
        vals=np.sort([float(r['cka']) for r in selected])
        median=float(np.median(vals)); medians.append(median)
        ax.step(np.r_[.66,vals,1.005],np.r_[0,np.arange(1,n+1)/n,1],where='post',
                color=color,lw=1.7,ls=style,label=f'{label} (n={n})')
    ax.set(xlim=(.66,1.005),ylim=(0,1.04),xticks=[.7,.8,.9,1],yticks=[0,.5,1],
           xlabel='Linear CKA (d = 0)',ylabel='Cumulative fraction')
    ax.legend(loc='lower left',bbox_to_anchor=(0,1.04),fontsize=12,frameon=False,
              handlelength=1.3,handletextpad=.5,borderaxespad=0,labelspacing=.4)
    ax.text(.035,.56,'Medians\n0.892 within\n0.889 cross',transform=ax.transAxes,
            fontsize=12,va='top',linespacing=1.4)
    fig.text(.56,12/(height*72),'Condition-pair measurements\nfrom six networks',
             ha='center',va='bottom',fontsize=12,linespacing=1.3)
    expected=json.loads((HERE/'data/actual_values.json').read_text())
    np.testing.assert_allclose(medians,[expected['D_0_median'],expected['D_1_median']],rtol=0,atol=1e-12)
    save(fig,'D')
    DRAWN['D']={'width_in':width,'height_in':height,'ordinary_font_pt':12,
                'delay':0,'unique_condition_pair_measurements':[432,486],
                'networks':6,'suffix_duplicates':0,'pooled_medians':medians}


def validate(panel):
    path=FIG/f'panel_{panel}.pdf'
    doc=fitz.open(path); assert len(doc)==1
    page=doc[0]
    assert not page.get_images(), 'Raster image object in vector PDF'
    assert not doc.metadata.get('author') and not doc.xref_xml_metadata()
    text=page.get_text()
    assert not re.search(r'Yang|Xuanyu|yangx|@|/home/|Acknowledg|University|Institute',text,re.I)
    assert not re.match(r'^[ABCD]\s',text.strip()), 'Burned-in panel number'
    assert 'B_reserved' not in text and 'ECDF' not in text and 'RNN' not in text
    spans=[s for b in page.get_text('dict')['blocks'] if 'lines' in b
             for line in b['lines'] for s in line['spans'] if s['text'].strip()]
    too_small=[s for s in spans if s['size']<11.99]
    if panel=='A':
        assert all(re.fullmatch(r'[DR]',s['text'].strip()) for s in too_small),too_small
    else:
        assert not too_small,too_small
    for s in spans:
        x0,y0,x1,y1=s['bbox']
        assert min(x0,y0)>=1 and x1<=page.rect.width-1 and y1<=page.rect.height-1,('PDF text clipped',s)
    fonts=[]
    for entry in page.get_fonts(full=True):
        xref,ext,kind,name=entry[:4]
        assert xref>0 and len(doc.extract_font(xref)[3])>0,('Font not embedded',name)
        assert kind!='Type3',('Text outlined as Type3',name)
        fonts.append(name)
    svg=ET.parse(FIG/f'panel_{panel}.svg').getroot()
    tag=lambda s:'{http://www.w3.org/2000/svg}'+s
    assert list(svg.iter(tag('text'))), 'SVG text not editable'
    assert not list(svg.iter(tag('image'))), 'SVG raster image'
    DRAWN[panel].update({'pdf_points':[page.rect.width,page.rect.height],
        'minimum_ordinary_pdf_pt':min(s['size'] for s in spans if s['size']>=11.99),
        'embedded_fonts':sorted(set(fonts)),'raster_objects':0,'editable_svg_text':True,
        'anonymous_metadata':True,'clipping_checks':'passed',
        'files':{f'figures/panel_{panel}.{ext}':sha(FIG/f'panel_{panel}.{ext}') for ext in ['pdf','svg','png']}})
    doc.close()


def check_protected():
    path=HERE/'archive/pre_stage1_2026-10-17/protected_files.json'
    protected=json.loads(path.read_text())
    for p,digest in protected.items():
        assert sha(HERE/p)==digest,f'Protected manuscript, data or legacy figure changed: {p}'
    return len(protected)


def run():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--panel',choices=['A','B','C','D','all'],default='all')
    parser.add_argument('--width',type=float,help='Design width in inches; requires a single --panel')
    parser.add_argument('--height',type=float,help='Design height in inches; requires a single --panel')
    args=parser.parse_args()
    if args.panel=='all' and (args.width or args.height):
        parser.error('Set width/height for one panel at a time; text is never scaled.')
    sources=check_inputs(); check_protected()
    funcs={'A':schematic,'B':behavior,'C':translation,'D':geometry}
    selected=list(funcs) if args.panel=='all' else [args.panel]
    FIG.mkdir(exist_ok=True)
    for panel in selected:
        w,h=SIZES[panel]
        funcs[panel](width=args.width or w,height=args.height or h)
        validate(panel)
        print(f'{panel}: {DRAWN[panel]["width_in"]} x {DRAWN[panel]["height_in"]} in; fonts >=12 pt; PDF/SVG/PNG saved.')
    protected_count=check_protected()
    report_path=HERE/'figure_validation.json'
    previous=json.loads(report_path.read_text()).get('panels',{}) if report_path.exists() else {}
    previous.update(DRAWN)
    report={'stage':1,'status':'independent figures for user review; LaTeX approval pending',
            'source_files_verified':sources,'protected_existing_files_unchanged':protected_count,
            'protected_files':json.loads((HERE/'archive/pre_stage1_2026-10-17/protected_files.json').read_text()),
            'model_forward_calls':0,'training_updates':0,'predictor_refits':0,'bootstrap_runs':0,
            'combined_figure_generated':False,'manuscript_compiled':False,
            'drawing_source_sha256':sha(HERE/'plot_figure.py'),
            'figure_notes_sha256':{name:sha(HERE/name) for name in ['captions.txt','figure_README.md'] if (HERE/name).exists()},
            'source_data_sha256':{str(p.relative_to(HERE)):sha(p) for p in (HERE/'data').glob('*') if p.is_file()},
            'panels':previous}
    report_path.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    run()

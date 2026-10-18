"""Editable SVG schematic, Matplotlib data panels, vector-only PDF assembly."""
from pathlib import Path
import csv
import json
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import cairosvg
from pypdf import PdfReader, PdfWriter, Transformation
import pymupdf as fitz

HERE = Path(__file__).resolve().parent
FIG = HERE / 'figures'
WIDTH = (7.3 * 72 - 24) / 2
HEIGHT = 216
COLORS = {0: '#0072B2', 1: '#D55E00'}
NATIVE = '#68549A'
INK = '#20252B'
MARKERS = ['o', '^', 's', 'v', 'P', 'X']
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
    'axes.labelsize': 10, 'xtick.labelsize': 10, 'ytick.labelsize': 10,
    'axes.linewidth': .8, 'lines.linewidth': 1.5, 'pdf.fonttype': 42,
    'ps.fonttype': 42, 'svg.fonttype': 'none', 'text.color': INK,
    'axes.labelcolor': INK, 'xtick.color': INK, 'ytick.color': INK})


def read(name):
    return list(csv.DictReader((HERE / 'data' / name).open()))


def schematic():
    # Truth is checked with list operations only, without importing model code.
    q = [2, 4, 1, 7, 3, 6, 5]
    donor = [3, 1, 5, 2, 7, 6, 4]
    receiver = [6, 4, 7, 1, 5, 2, 3]
    assert [donor.index(i) + 1 for i in range(1, 8)] == q  # ctx0 A = inverse
    assert [8 - j for j in receiver] == q                 # ctx1 A = complement
    svg = ET.Element('svg', xmlns='http://www.w3.org/2000/svg',
                     width=f'{WIDTH}pt', height=f'{HEIGHT}pt',
                     viewBox=f'0 0 {WIDTH} {HEIGHT}')
    defs = ET.SubElement(svg, 'defs')
    marker = ET.SubElement(defs, 'marker', id='arrow', markerWidth='5', markerHeight='5',
                           refX='4', refY='2.5', orient='auto')
    ET.SubElement(marker, 'path', d='M0,0 L5,2.5 L0,5 Z', fill=INK)
    def txt(x, y, s, size=9.5, weight='normal', anchor='start', fill=INK):
        el = ET.SubElement(svg, 'text', x=str(x), y=str(y), fill=fill,
                           **{'font-family': 'DejaVu Sans', 'font-size': str(size),
                              'font-weight': weight, 'text-anchor': anchor})
        el.text = s
    def line(x1, y1, x2, y2, arrow=True):
        attrs = {'x1': str(x1), 'y1': str(y1), 'x2': str(x2), 'y2': str(y2),
                 'stroke': INK, 'stroke-width': '.8'}
        if arrow:
            attrs['marker-end'] = 'url(#arrow)'
        ET.SubElement(svg, 'line', attrs)
    def box(x, y, w, h, fill='#F4F6F8', radius=3):
        ET.SubElement(svg, 'rect', x=str(x), y=str(y), width=str(w), height=str(h),
                      rx=str(radius), fill=fill, stroke='#B7BEC6', **{'stroke-width': '.65'})
    txt(2, 13, 'A', 12, 'bold')
    txt(20, 13, 'Task and memory interchange', 11.3, 'bold')
    txt(2, 32, 'Context 0: A = inversion; B = complement', 9.5)
    txt(2, 46, 'Context 1: A = complement; B = inversion', 9.5)
    box(1, 57, WIDTH - 2, 154, '#FAFBFC', 5)
    txt(WIDTH / 2, 71, 'One trained RNN · fixed weights', 10, 'bold', 'middle')
    for y, role, context, p, hist, sub in [
            (88, 'Donor', 0, '3 1 5 2 7 6 4', 'A N', 'D'),
            (125, 'Receiver', 1, '6 4 7 1 5 2 3', 'N A', 'R')]:
        txt(9, y, f'{role} trial · context {context}', 9.5, 'bold')
        txt(9, y + 16, '[' + p + ']', 9.5)
        txt(122, y + 8, hist, 9.5, anchor='middle')
        line(102, y + 14, 144, y + 14)
        txt(151, y + 17, 'q', 11)
        line(162, y + 14, 174, y + 14)
        box(180, y + 4, 29, 20, '#E3EEF5')
        box(211, y + 4, 29, 20, '#EEEAF3')
        txt(194.5, y + 18, 'm' + ('ᴅ' if sub == 'D' else 'ʀ'), 11, anchor='middle')
        txt(225.5, y + 18, 'c' + ('ᴅ' if sub == 'D' else 'ʀ'), 11, anchor='middle')
    txt(9, 162, 'Identical current q = [2 4 1 7 3 6 5]', 9.5)
    txt(9, 182, 'Replace mʀ ← mᴅ:', 10, 'bold')
    box(130, 168, 110, 21, '#FFFFFF')
    txt(185, 182, '(mᴅ, cʀ)', 11, 'bold', 'middle')
    txt(WIDTH / 2, 202, 'Receiver inputs → same RNN → original readout', 9.25, anchor='middle')
    # Text elements remain editable; no bitmap or outlined text is used.
    path = FIG / 'panel_A.svg'
    ET.ElementTree(svg).write(path, encoding='unicode', xml_declaration=True)
    cairosvg.svg2pdf(url=str(path), write_to=str(FIG / 'panel_A.pdf'))


def canvas(letter, title):
    fig = plt.figure(figsize=(WIDTH / 72, HEIGHT / 72))
    fig.text(2 / WIDTH, 1 - 13 / HEIGHT, letter, fontsize=12, weight='bold')
    fig.text(20 / WIDTH, 1 - 13 / HEIGHT, title, fontsize=11.3, weight='bold')
    ax = fig.add_axes([.195, .18, .77, .66])
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(width=.8, length=3)
    ax.grid(axis='y', color='#E5E8EB', linewidth=.6, zorder=0)
    return fig, ax


def save(fig, panel):
    fig.savefig(FIG / f'panel_{panel}.pdf', metadata={'Author': '', 'Creator': 'Matplotlib',
                   'Title': f'Panel {panel}', 'CreationDate': None, 'ModDate': None})
    fig.savefig(FIG / f'panel_{panel}.svg', metadata={'Creator': 'Matplotlib', 'Date': None})
    svg_path = FIG / f'panel_{panel}.svg'
    svg_path.write_text('\n'.join(line.rstrip() for line in svg_path.read_text().splitlines()) + '\n')
    plt.close(fig)


def behavior():
    rows = read('panel_B_seed.csv')
    fig, ax = canvas('B', 'Context-dependent damage')
    for cross in (0, 1):
        vals = [float(r['new_error_percent']) for r in rows if int(r['cross']) == cross]
        mean = np.mean(vals)
        for j, v in enumerate(vals):
            ax.scatter(cross + np.linspace(-.16, .16, 6)[j], v, marker=MARKERS[j],
                       s=30, facecolors='white', edgecolors=COLORS[cross], linewidth=.9, zorder=4)
        ax.scatter(cross, mean, marker='D', s=39, color=COLORS[cross], edgecolor=INK,
                   linewidth=.65, zorder=5)
        ax.text(cross, mean + 4.1, f'{mean:.1f}%', ha='center', fontsize=11, weight='bold',
                color=COLORS[cross])
    ax.set(xlim=(-.42, 1.42), ylim=(-1, 63), xticks=[0, 1],
           xticklabels=['Within-context', 'Cross-context'], yticks=[0, 20, 40, 60],
           ylabel='New errors (% of all queries)')
    ax.tick_params(axis='x', labelsize=9.5)
    fig.text(.55, .03, 'Small symbols: 6 networks; ◆: mean', ha='center', fontsize=9.5)
    save(fig, 'B')


def translation():
    rows = read('panel_C_seed.csv')
    fig, ax = canvas('C', 'Cue translation reverses the effect')
    ax.set_position([.195, .20, .77, .64])
    for group, color, style in [('Cross-context exchange', COLORS[1], '-'),
                                ('No exchange', NATIVE, '--')]:
        ys = []
        for j, seed in enumerate(range(101, 107)):
            vals = [float(next(r for r in rows if int(r['seed']) == seed
                       and r['treatment'] == group and r['cues'] == cue)['accuracy_percent'])
                    for cue in ('original', 'translated')]
            ys.append(vals)
            xs = np.array([0, 1]) + np.linspace(-.065, .065, 6)[j]
            ax.plot(xs, vals, color=color, lw=.7, alpha=.38, zorder=2)
            ax.scatter(xs, vals, color=color, s=20, marker=MARKERS[j], alpha=.85, zorder=4)
        means = np.mean(ys, axis=0)
        ax.plot([0, 1], means, style, color=color, linewidth=2.2, zorder=3)
        ax.scatter([0, 1], means, marker='D', s=35, color=color,
                   edgecolor=INK, linewidth=.6, zorder=5)
        for x, val in enumerate(means):
            ax.text(x, val + (5 if val > 80 else -10), f'{val:.1f}%',
                    ha='center', fontsize=10, color=color, weight='bold')
    ax.set(xlim=(-.22, 1.22), ylim=(0, 112), xticks=[0, 1],
           xticklabels=['Original cues', 'Translated cues'], yticks=[0, 50, 100],
           ylabel='Original-task accuracy (%)')
    ax.tick_params(axis='x', labelsize=9.5)
    handles = [Line2D([0], [0], color=COLORS[1], lw=2, label='Cross-context exchange'),
               Line2D([0], [0], color=NATIVE, lw=2, ls='--', label='No exchange (native)')]
    ax.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, -.015),
              fontsize=9, frameon=False, handlelength=1.6, labelspacing=.3)
    fig.text(.57, .035, 'B_reserved · all suffixes, including NN', ha='center', fontsize=9.5)
    save(fig, 'C')


def geometry():
    rows = read('panel_D_d0.csv')
    fig, ax = canvas('D', 'Geometry distributions overlap')
    ax.set_position([.195, .235, .77, .605])
    for cross, style, label in [(0, '-', 'Within-context'), (1, '--', 'Cross-context')]:
        values = np.sort([float(r['cka']) for r in rows if int(r['cross']) == cross])
        xs = np.r_[.66, values, 1.005]
        ys = np.r_[0, np.arange(1, len(values) + 1) / len(values), 1]
        ax.step(xs, ys, where='post', c=COLORS[cross], ls=style, lw=1.7,
                label=f'{label} (n={len(values)})')
    ax.set(xlim=(.66, 1.005), ylim=(0, 1.03), xticks=[.7, .8, .9, 1.0],
           yticks=[0, .5, 1], xlabel='Linear CKA (d = 0)', ylabel='Cumulative fraction')
    ax.legend(loc='upper left', fontsize=9, frameon=False, handlelength=1.8)
    ax.text(.03, .63, 'Medians\n0.892 within\n0.889 cross', transform=ax.transAxes,
            fontsize=9.5, linespacing=1.4, va='top')
    fig.text(.56, .025, 'Condition pairs pooled across 6 networks', ha='center', fontsize=9.5)
    save(fig, 'D')


def assemble():
    writer = PdfWriter()
    page = writer.add_blank_page(width=2 * WIDTH + 24, height=2 * HEIGHT + 24)
    for panel, x, y in [('A', 0, HEIGHT + 24), ('B', WIDTH + 24, HEIGHT + 24),
                        ('C', 0, 0), ('D', WIDTH + 24, 0)]:
        source = PdfReader(FIG / f'panel_{panel}.pdf').pages[0]
        page.merge_transformed_page(source, Transformation().translate(x, y))
    writer.add_metadata({'/Title': 'Content-matched memory-state interchange',
                         '/Author': '', '/Creator': 'Vector figure assembly'})
    with (FIG / 'figure_1.pdf').open('wb') as f:
        writer.write(f)
    for name in ['figure_1', 'panel_A', 'panel_B', 'panel_C', 'panel_D']:
        doc = fitz.open(FIG / f'{name}.pdf')
        assert len(doc[0].get_images()) == 0, 'Unexpected raster image'
        doc[0].get_pixmap(matrix=fitz.Matrix(2, 2)).save(HERE / 'preview' / f'{name}.png')


if __name__ == '__main__':
    FIG.mkdir(exist_ok=True)
    schematic()
    behavior()
    translation()
    geometry()
    assemble()
    print('Saved editable SVG, vector PDFs, and PNG previews.')

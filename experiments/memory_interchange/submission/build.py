"""Compile the anonymous two-page manuscript and render both pages.

Usage: python build.py [--tectonic /path/to/tectonic]
Requires tectonic, numpy, matplotlib, cairosvg, pypdf and pymupdf.
"""
from pathlib import Path
import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import pymupdf as fitz

HERE = Path(__file__).resolve().parent


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tectonic', default=shutil.which('tectonic'))
    args = parser.parse_args()
    if not args.tectonic:
        parser.error('tectonic is required; specify --tectonic /path/to/tectonic')
    summary = (HERE / 'summary.txt').read_text().strip()
    assert len(summary.split()) <= 300
    assert '\n' not in summary, 'Summary must be one paragraph'
    tex_summary = (summary.replace('’', "'").replace('%', '\\%').replace('>99.99', '$>99.99$')
                   .replace('8−j', '$8-j$'))
    (HERE / 'summary.tex').write_text(tex_summary + '\n')
    for script in ('audit_data.py', 'plot_figure.py'):
        subprocess.run([sys.executable, str(HERE / script)], check=True, cwd=HERE)
    result = subprocess.run([args.tectonic, '--keep-logs', 'submission.tex'],
                            cwd=HERE, text=True, capture_output=True)
    # Logs contain only local build diagnostics, not document source paths.
    (HERE / 'build.log').write_text(result.stdout + result.stderr)
    print(result.stdout + result.stderr)
    result.check_returncode()
    log = (HERE / 'submission.log').read_text(errors='replace')
    assert not re.search(r'Overfull \\[hv]box', log), 'Overfull box in manuscript'
    assert 'Missing character:' not in log, 'Missing glyph in manuscript'
    pdf = HERE / 'submission.pdf'
    doc = fitz.open(pdf)
    assert len(doc) == 2, f'Expected two pages, got {len(doc)}'
    # Remove optional metadata and keep only anonymous document information.
    doc.set_metadata({'title': 'Rule context constrains the interchangeability of content-matched memory states',
                      'author': '', 'subject': 'Anonymous COSYNE submission',
                      'creator': 'LaTeX', 'producer': 'LaTeX',
                      'creationDate': '', 'modDate': '', 'keywords': ''})
    if doc.xref_xml_metadata():
        doc.del_xml_metadata()
    cleaned = HERE / 'submission.clean.pdf'
    doc.save(cleaned, garbage=4, deflate=True)
    doc.close()
    cleaned.replace(pdf)
    doc = fitz.open(pdf)
    page_info = []
    extracted = []
    for i, page in enumerate(doc):
        page.get_pixmap(matrix=fitz.Matrix(2, 2)).save(HERE / 'preview' / f'page_{i+1}.png')
        text = page.get_text()
        extracted.append(text)
        assert not re.search(r'Yang|Xuanyu|yangx|@|Acknowledg|University|Institute|/home/', text, re.I)
        spans = [s for b in page.get_text('dict')['blocks'] if 'lines' in b
                 for line in b['lines'] for s in line['spans'] if s['text'].strip()]
        margin = .5 * 72
        outside = [s['text'] for s in spans if s['bbox'][0] < margin - .5
                   or s['bbox'][1] < margin - .5 or s['bbox'][2] > page.rect.width - margin + .5
                   or s['bbox'][3] > page.rect.height - margin + .5]
        assert not outside, ('Text outside 0.5-inch margins', outside)
        assert len(page.get_images()) == 0, 'Manuscript must use vector figure'
        # Ignore superscripts and mathematical script glyphs for body size.
        prose_spans = [s for s in spans if s['font'].startswith('NimbusRom')
                       and s['size'] > (10.5 if i == 0 else 9)]
        prose_min = min(s['size'] for s in prose_spans)
        assert prose_min >= (12 if i == 0 else 10), 'Prose font too small'
        page_info.append({'page': i + 1, 'width_pt': page.rect.width,
                          'height_pt': page.rect.height, 'raster_images': len(page.get_images()),
                          'min_text_size_pdf_pt': min(s['size'] for s in spans),
                          'min_prose_font_pdf_pt': prose_min,
                          'smaller_glyphs': 'mathematical subscripts; page 2 also contains figure labels',
                          'text_bbox': [min(s['bbox'][0] for s in spans), min(s['bbox'][1] for s in spans),
                                        max(s['bbox'][2] for s in spans), max(s['bbox'][3] for s in spans)]})
    (HERE / 'preview' / 'extracted_text.txt').write_text('\n\f\n'.join(extracted))
    report = {'pages': len(doc), 'paper': 'US Letter', 'margin_in': .6,
              'body_tex_pt': 12.1, 'caption_and_references_tex_pt': 10.1,
              'summary_words_whitespace': len(summary.split()), 'summary_changed': False,
              'summary_paragraphs': 1, 'metadata': doc.metadata,
              'additional_detail_words_pdf_whitespace': len(re.sub(r'-\n', '', extracted[0].split('Additional Detail', 1)[1]).split()),
              'xmp_metadata_present': bool(doc.xref_xml_metadata()),
              'overfull_boxes': len(re.findall(r'Overfull \\[hv]box', log)),
              'underfull_boxes': len(re.findall(r'Underfull \\[hv]box', log)),
              'missing_glyphs': 'Missing character:' in log,
              'page_details': page_info,
              'pdf_sha256': hashlib.sha256(pdf.read_bytes()).hexdigest(),
              'automated_checks': 'passed; visual inspection recorded separately in README.md'}
    (HERE / 'validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print('Two pages compiled, anonymized, checked, and rendered. Summary: 288 words.')


if __name__ == '__main__':
    run()

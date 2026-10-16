#!/usr/bin/env python3
"""Build English editions without changing frozen experimental artifacts."""

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs/en'
BASE = Path('experiments/memory_interchange')
MEMORY = ROOT / 'tools/english_translations.json'
CJK = re.compile(r'[\u3400-\u9fff]')
LINK = re.compile(r'\]\(([^\s()]+)\)')
NUMBER = re.compile(r'[+−-]?\d+(?:\.\d+)?(?:[eE][+−-]?\d+)?')
PROTECTED = re.compile(r'`[^`]*`|\]\([^\n]*?\)|https?://\S+|[0-9a-f]{32,}|\b[A-Za-z_][A-Za-z0-9_.]*\b')
PUNCTUATION = str.maketrans({
    '\u3001': ', ', '\uff0c': ', ', '\u3002': '. ', '\uff1a': ': ',
    '\uff1b': '; ', '\uff08': ' (', '\uff09': ')',
    '\u201c': '"', '\u201d': '"', '\u3010': '[', '\u3011': ']',
})


def sha(data):
    return hashlib.sha256(data).hexdigest()


def sources():
    names = subprocess.check_output(['git', 'ls-files', '-z', str(BASE)], cwd=ROOT)
    return sorted(Path(p) for p in names.decode().rstrip('\0').split('\0')
                  if p.endswith('.md') or Path(p).name.startswith('protocol_')
                  or Path(p).name == 'english_result.txt')


def template(text):
    numbers = []

    def replace(match):
        numbers.append(match[0])
        return f'<N{len(numbers)-1}>'

    parts = []
    pos = 0
    for match in PROTECTED.finditer(text):
        parts.append(NUMBER.sub(replace, text[pos:match.start()]))
        parts.append(match[0])
        pos = match.end()
    parts.append(NUMBER.sub(replace, text[pos:]))
    return ''.join(parts), numbers


def units(line):
    # Translate table cells independently so repeated labels share one translation.
    if line.startswith('|'):
        return re.split(r'(\|)', line)
    return [line]


def inventory():
    result = {}
    for source in sources():
        for line in (ROOT / source).read_text().splitlines():
            for unit in units(line):
                if not CJK.search(unit):
                    continue
                normalized, _ = template(unit.strip())
                key = sha(normalized.encode())
                if key not in result:
                    result[key] = {'key': key, 'source': str(source), 'text': normalized}
    return list(result.values())


def translate(unit, memory):
    if not CJK.search(unit):
        return unit
    original = unit.strip()
    normalized, numbers = template(original)
    key = sha(normalized.encode())
    if key not in memory:
        raise ValueError(f'Missing translation {key}: {normalized}')
    translated = memory[key]
    placeholders = Counter(re.findall(r'<N\d+>', normalized))
    if Counter(re.findall(r'<N\d+>', translated)) != placeholders:
        raise ValueError(f'Numerical placeholders changed: {normalized}')
    for i, value in enumerate(numbers):
        translated = translated.replace(f'<N{i}>', value)
    if Counter(NUMBER.findall(original)) != Counter(NUMBER.findall(translated)):
        raise ValueError(f'Numerical values changed: {original}')
    if CJK.search(translated):
        raise ValueError(f'Untranslated text: {translated}')
    return unit[:len(unit)-len(unit.lstrip())] + translated + unit[len(unit.rstrip()):]


def rewrite_links(text, source, translated_sources):
    def replace(match):
        url = urlsplit(match[1])
        if url.scheme or url.netloc or not url.path or url.path.startswith('/'):
            return match[0]
        target = (ROOT / source.parent / unquote(url.path)).resolve()
        try:
            relative = target.relative_to(ROOT)
        except ValueError:
            return match[0]
        destination = OUT / relative if relative in translated_sources else target
        result = Path(os.path.relpath(destination, (OUT / source).parent)).as_posix()
        if url.query:
            result += '?' + url.query
        if url.fragment:
            result += '#' + url.fragment
        return '](' + result + ')'
    return LINK.sub(replace, text)


def english_punctuation(text):
    lines = []
    fenced = False
    for line in text.splitlines():
        if line.lstrip().startswith(('```', '~~~')):
            fenced = not fenced
        if fenced or line.lstrip().startswith(('```', '~~~')):
            lines.append(line)
            continue
        parts = []
        pos = 0
        for match in PROTECTED.finditer(line):
            parts.extend((line[pos:match.start()].translate(PUNCTUATION), match[0]))
            pos = match.end()
        parts.append(line[pos:].translate(PUNCTUATION))
        lines.append(''.join(parts).rstrip())
    return '\n'.join(lines)


def generate(check):
    memory = json.loads(MEMORY.read_text())
    paths = sources()
    path_set = set(paths)
    manifest = {'scope': 'English editions; frozen originals and scientific results are unchanged.', 'documents': []}
    failures = []
    for source in paths:
        raw = (ROOT / source).read_bytes()
        original = raw.decode()
        translated = '\n'.join(''.join(translate(u, memory) for u in units(line))
                               for line in original.splitlines())
        translated = english_punctuation(translated)
        # Executable command blocks must remain byte-for-byte identical.
        blocks = lambda s: re.findall(r'```[^\n]*\n([\s\S]*?)```', s)
        if blocks(original) != blocks(translated):
            raise ValueError(f'Command block changed: {source}')
        snippets = lambda s: Counter(x for x in re.findall(r'(?<!`)`([^`\n]+)`(?!`)', s)
                                     if not CJK.search(x))
        if snippets(original) - snippets(translated):
            raise ValueError(f'Program identifier changed: {source}')
        translated = rewrite_links(translated, source, path_set)
        target = OUT / source
        original_link = Path(os.path.relpath(ROOT / source, target.parent)).as_posix()
        index_link = Path(os.path.relpath(OUT / 'README.md', target.parent)).as_posix()
        note = (f'> English edition · [Frozen original]({original_link}) · [English index]({index_link})\n>\n'
                '> Dates, status, execution scope and conclusions describe the original version. '
                'Commands, identifiers, data and figures retain their original definitions.\n\n')
        rendered = (note + translated.rstrip('\n') + '\n').encode()
        if check:
            if not target.exists() or target.read_bytes() != rendered:
                failures.append(f'English edition is stale: {source}')
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(rendered)
        manifest['documents'].append({'source': str(source), 'source_sha256': sha(raw),
                                     'english': str(target.relative_to(ROOT)), 'english_sha256': sha(rendered)})
    manifest['builder_sha256'] = sha(Path(__file__).read_bytes())
    manifest['translations_sha256'] = sha(MEMORY.read_bytes())
    data = (json.dumps(manifest, indent=2, ensure_ascii=False) + '\n').encode()
    path = OUT / 'translation_manifest.json'
    if check:
        if not path.exists() or path.read_bytes() != data:
            failures.append('Translation manifest is stale.')
        for path in [ROOT / 'README.md', *OUT.rglob('*.md')]:
            text = path.read_text()
            if CJK.search(text):
                failures.append(f'Non-English text: {path.relative_to(ROOT)}')
            for match in LINK.finditer(text):
                url = urlsplit(match[1])
                if not url.scheme and not url.netloc and url.path:
                    if not (path.parent / unquote(url.path)).exists():
                        failures.append(f'Missing local link: {path.relative_to(ROOT)} -> {url.path}')
        if failures:
            raise SystemExit('\n'.join(failures))
        print(f'PASS: {len(paths)} complete English editions; numbers, commands, sources and local links verified.')
    else:
        path.write_bytes(data)
        print(f'Generated {len(paths)} English editions; frozen originals unchanged.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Verify without writing files.')
    parser.add_argument('--inventory', type=Path, help='Export translation units for review.')
    args = parser.parse_args()
    if args.inventory:
        data = inventory()
        args.inventory.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
        print(f'{len(data)} unique translation units.')
    else:
        generate(args.check)

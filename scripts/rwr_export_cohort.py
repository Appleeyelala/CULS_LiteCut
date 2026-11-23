#!/usr/bin/env python3
"""Step 3 v3 cohort export: rebuild the candidate export for every added circuit.

Reuses the step-2 driver (patched ABC, six acceptance checks) and extends the
cohort from four pilots to the 14 development circuits plus 30 sampled from the
905 manifest. The external 24-design panel is excluded by SHA-256 and by name
family before any path is opened; its AIG files are never read here.
"""
import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rwr_dump_experiment import (  # noqa: E402
    check_circuit, export_candidates, file_digest, identifier_mapping, run_abc,
)
from global_timing.structure import aig_structure_digest, parse_aig  # noqa: E402

REPO = Path('/home/wushangge/CULS_LiteCut')
DATA = Path('/data/wushangge/CULS_LiteCut')
DEV_DIR = DATA / 'runs/20261119T020347Z-litecut-ours14-af7ccd61/input/circuits'
PILOT_DIR = DATA / 'runs/20261123T051432Z-rwr-dump-four-f29ceabd/input/circuits'
PILOT_EXPORT = DATA / 'runs/20261123T051432Z-rwr-dump-four-f29ceabd/output/dump'
MANIFEST = REPO / 'benchmark905_manifest.json'
PANEL = DATA / 'runs/20261119T070904Z-paper-split24-399b690c/output/data/panel.json'
SEED = 20261123
BANDS = [('100-299', 100, 299), ('300-999', 300, 999), ('1000+', 1000, None)]
PER_BAND = 10
AND_CAP = 100000


def sha256(path):
    checksum = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            checksum.update(chunk)
    return checksum.hexdigest()


def family(name):
    stem = Path(name).stem.lower()
    stem = re.sub(r'_(orig|comb|comb_1|1)$', '', stem)
    return re.sub(r'[^a-z0-9]', '', stem)


def aig_ands(path):
    with open(path, 'rb') as stream:
        head = stream.readline().decode('ascii', 'replace').split()
    if not head or head[0] != 'aig':
        raise SystemExit(f'not a binary AIG: {path}')
    return int(head[5])


def rank_key(row):
    return hashlib.sha256(f'{SEED}:{row["sha256"]}'.encode()).hexdigest()


def build_cohort():
    pilots = []
    for path in sorted(PILOT_DIR.glob('*.aig')):
        pilots.append(dict(name=path.stem, path=str(path), sha256=sha256(path),
                           ands=aig_ands(path), group='pilot', origin='step2 export'))
    dev = []
    for path in sorted(DEV_DIR.glob('*.aig')):
        dev.append(dict(name=path.stem, path=str(path), sha256=sha256(path),
                        ands=aig_ands(path), group='development'))
    pilot_sha = {row['sha256'] for row in pilots}
    dev_new, dev_overlap = [], []
    for row in dev:
        if row['sha256'] in pilot_sha:
            dev_overlap.append(row['name'])
        else:
            dev_new.append(row)

    panel_rows = json.loads(PANEL.read_text())['rows']
    panel_sha = {row['aig_sha256'] for row in panel_rows}
    panel_families = {family(row['source']) for row in panel_rows}
    known_sha = pilot_sha | {row['sha256'] for row in dev}
    known_families = {family(row['name']) for row in pilots} | {family(row['name']) for row in dev}

    manifest = json.loads(MANIFEST.read_text())
    pool, excluded = [], []
    for row in manifest['rows']:
        reason = None
        if row['sha256'] in known_sha:
            reason = 'known_pilot_or_development'
        elif row['sha256'] in panel_sha:
            reason = 'external_panel_sha256'
        elif family(row['relative_path']) in panel_families:
            reason = 'external_panel_family'
        elif family(row['relative_path']) in known_families:
            reason = 'known_family'
        if reason:
            excluded.append(dict(relative_path=row['relative_path'], sha256=row['sha256'],
                                 ands=row['and_count'], reason=reason))
        else:
            pool.append(row)

    selections, skipped, shortfalls = [], [], []
    for label, low, high in BANDS:
        rows = [row for row in pool
                if row['and_count'] >= low and (high is None or row['and_count'] <= high)]
        rows.sort(key=rank_key)
        over = [row for row in rows if row['and_count'] > AND_CAP]
        rows = [row for row in rows if row['and_count'] <= AND_CAP]
        if len(rows) < PER_BAND:
            shortfalls.append(dict(band=label, available=len(rows), requested=PER_BAND))
        for row in rows[:PER_BAND]:
            selections.append(dict(row=row, band=label, substitute=False))
        if over:
            skipped.append(dict(band=label, skipped_over_and_cap=len(over),
                                smallest=min(row['and_count'] for row in over)))

    deficit = PER_BAND * len(BANDS) - len(selections)
    if deficit:
        used = {item['row']['sha256'] for item in selections}
        extra = [row for row in pool
                 if row['and_count'] >= 1000 and row['and_count'] <= AND_CAP
                 and row['sha256'] not in used]
        extra.sort(key=rank_key)
        for row in extra[:deficit]:
            selections.append(dict(row=row, band='substitute', substitute=True))

    sampled = []
    for item in selections:
        row = item['row']
        sampled.append(dict(name=Path(row['relative_path']).stem,
                            path=row['staged_path'], sha256=row['sha256'],
                            ands=row['and_count'], group='sample905', band=item['band'],
                            substitute=item['substitute'],
                            relative_path=row['relative_path'], source=row['source']))
    names = [row['name'] for row in pilots + dev_new + sampled]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise SystemExit(f'duplicate circuit names in the cohort: {duplicates}')

    return dict(
        seed=SEED,
        rule='hash order = sha256(f"{seed}:{circuit_sha256}"); ascending; per band cap 10; '
             'skip >100,000 AND and register; substitute the shortfall from the >=1000 band '
             'in the same hash order',
        excluded_panel=dict(count=len(panel_sha), families=sorted(panel_families)[:5],
                            overlap_with_cohort=len(panel_sha & known_sha)),
        exclusions=excluded,
        exclusions_by_reason={reason: sum(1 for row in excluded if row['reason'] == reason)
                              for reason in {row['reason'] for row in excluded}},
        shortfalls=shortfalls,
        skipped=skipped,
        pilots=pilots,
        development_all=dev,
        development_overlap=dev_overlap,
        development_new=dev_new,
        sampled=sampled,
        total_unique=len(pilots) + len(dev_new) + len(sampled),
    )


def close_checks(run, abc_orig, abc_dump, path, name, folder):
    """Step-2 check 1: the patched and unpatched builds agree on the same source."""
    target = run / 'output/aig'
    target.mkdir(parents=True, exist_ok=True)
    written = {}
    record = {}
    for tag, binary in (('orig', abc_orig), ('dump', abc_dump)):
        out = target / f'{name}.{tag}.aig'
        log = run / 'output/logs' / f'{name}.{tag}.log'
        proc = run_abc(binary, f'read_aiger {path}; print_stats; rewrite; print_stats; write_aiger {out}', log)
        text = log.read_text()
        blocks = re.findall(r'and\s*=\s*(\d+)\s+lev\s*=\s*(\d+)', text)
        if len(blocks) < 2:
            raise SystemExit(f'{name}/{tag}: missing print_stats blocks')
        record[tag] = dict(before=blocks[0], after=blocks[1],
                           digest=aig_structure_digest(out) if out.exists() else None,
                           returncode=proc.returncode)
        written[tag] = out
    cec_log = run / 'output/logs' / f'{name}.cec.log'
    run_abc(abc_orig, f'read_aiger {written["orig"]}; cec {written["dump"]}', cec_log)
    text = cec_log.read_text().lower()
    record['cec'] = ('equivalent' in text) and ('not equivalent' not in text)
    record['match'] = (record['orig']['after'] == record['dump']['after']
                       and record['orig']['digest'] == record['dump']['digest'])
    if not record['match'] or not record['cec']:
        raise SystemExit(f'{name}: patched and unpatched builds disagree '
                         f'(match={record["match"]}, cec={record["cec"]})')
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--build', required=True, type=Path)
    parser.add_argument('--limit', type=int, default=None, help='debug: only the first N circuits')
    parser.add_argument('--only', default=None, help='debug: comma-separated circuit names')
    args = parser.parse_args()
    run = args.run.resolve()
    if not os.getenv('SLURM_JOB_ID'):
        raise SystemExit('run this only inside the committed /data Slurm job')
    if not run.is_relative_to('/data/wushangge'):
        raise SystemExit('run this only under /data/wushangge')
    (run / 'output/logs').mkdir(parents=True, exist_ok=True)
    cohort = build_cohort()
    (run / 'output/cohort.json').write_text(json.dumps(cohort, ensure_ascii=False, indent=2) + '\n')
    print('cohort:', cohort['total_unique'], 'pilots', len(cohort['pilots']),
          'dev_new', len(cohort['development_new']), 'sampled', len(cohort['sampled']), flush=True)
    print('shortfalls', cohort['shortfalls'], 'skipped', cohort['skipped'], flush=True)

    abc_orig = args.build / 'build/abc_orig/abc'
    abc_dump = args.build / 'build/abc_dump/abc'
    for binary in (abc_orig, abc_dump):
        if not binary.is_file():
            raise SystemExit(f'missing binary: {binary}')

    jobs = [dict(name=row['name'], path=Path(row['path']), sha=row['sha256'], group='pilot')
            for row in cohort['pilots']]
    jobs += [dict(name=row['name'], path=Path(row['path']), sha=row['sha256'], group='development')
             for row in cohort['development_new']]
    jobs += [dict(name=row['name'], path=Path(row['path']), sha=row['sha256'], group='sample905')
             for row in cohort['sampled']]
    if args.only:
        wanted = {item.strip() for item in args.only.split(',') if item.strip()}
        jobs = [job for job in jobs if job['name'] in wanted]
    if args.limit:
        jobs = jobs[:args.limit]

    results = []
    for job in jobs:
        name = job['name']
        path = job['path']
        if not path.is_file():
            raise SystemExit(f'{name}: input missing at {path}')
        if file_digest(path) != job['sha']:
            raise SystemExit(f'{name}: input digest changed at {path}')
        if job['group'] == 'pilot':
            folder = PILOT_EXPORT / name
            result = dict(circuit=name, group=job['group'], export=str(folder), reused=True,
                          input_sha256=job['sha'], ands=aig_ands(path))
        else:
            folder = export_candidates(run, abc_dump, path, name)
            info = json.loads((folder / 'export_check.json').read_text())
            aig = parse_aig(path)
            mapping = identifier_mapping(folder / 'nodes.tsv', aig, name)
            checks = check_circuit(folder, path, mapping['offset'], name)
            close = close_checks(run, abc_orig, abc_dump, path, name, folder)
            result = dict(circuit=name, group=job['group'], export=str(folder), reused=False,
                          input_sha256=job['sha'], ands=aig_ands(path), untouched=info['untouched'],
                          mapping=mapping, checks=checks, close=close)
            (run / 'output' / f'{name}.checks.json').write_text(
                json.dumps(result, ensure_ascii=False, indent=2) + '\n')
        results.append(result)
        print(f"{name}: ands={result['ands']} reused={result['reused']}", flush=True)
        (run / 'output/export_manifest.json').write_text(
            json.dumps(dict(cohort=dict(total=cohort['total_unique']), results=results),
                       ensure_ascii=False, indent=2) + '\n')
    print('exported', len(results), flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())

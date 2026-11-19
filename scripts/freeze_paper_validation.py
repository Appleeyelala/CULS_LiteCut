"""Freeze a prospective validation panel using source, ancestry and size only."""
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys

from audit_replay_noop_v3 import fingerprints
from paper_joint_evidence import ROOT, digest, read, save_json
from global_timing.common import run_tool, tool_path, ToolError
from global_timing.structure import parse_aig

SOURCES = [('/data/public/opensource_benchmark/EPFL_Benchmark', '*/*.aig'),
           ('/data/public/hdl-benchmarks/mcnc/Combinational/blif', '*.blif'),
           ('/data/public/hdl-benchmarks/lgsynth91/blif', '*.blif')]


def family(path):
    stem = path.stem.lower()
    stem = re.sub(r'(_(orig|comb|opt|mapped|aig)|_v?\d+)+$', '', stem)
    word = re.sub('[^a-z0-9]', '', stem)
    if re.fullmatch(r'x\d+dn',word):
        return 'x'
    if re.fullmatch(r'bc[a-d]',word):
        return 'bc'
    word = re.sub(r'\d+[a-z]?$', '', word) or word
    aliases = {'multiplier':'mult','multiply':'mult','barrelshifter':'bar','priorityencoder':'priority',
               'divisor':'div','addition':'adder','decoder':'dec'}
    return aliases.get(word, word)


def identity(p):
    aig = parse_aig(p)
    codes = fingerprints(aig)['output_hashes']
    checksum = hashlib.sha256((str(aig.pi_count)+'/'+','.join(codes)).encode()).hexdigest()
    return checksum, aig.and_count


def main(out, staged):
    out.mkdir()
    groups = set()
    hashes = set()
    structures = set()
    historical = read(ROOT / 'benchmark905_manifest.json')['rows']
    historical = [(staged / r['relative_path'], r['sha256']) for r in historical]
    with (ROOT/'analysis/paper14_fullflow/results.csv').open() as stream:
        for row in csv.DictReader(stream):
            inputs = read(Path(row['work']) / 'inputs.json')
            historical.append((Path(inputs['original']), inputs['original_sha256']))
    for src, expected in historical:
        groups.add(family(src))
        if expected not in hashes:
            assert digest(src)==expected
            structures.add(identity(src)[0])
            hashes.add(expected)
    candidates = out / 'candidates'
    candidates.mkdir()
    binary = Path(read(ROOT/'config/load_rewrite_tools.json')['frozen_abc'])
    order = lambda p: hashlib.sha256(('CULS-paper-validation-v1/'+str(p)).encode()).digest()
    files = {p for root,pattern in SOURCES for p in Path(root).glob(pattern) if 'best_results' not in p.parts}
    audit = []
    eligible = []
    for index,src in enumerate(sorted(files,key=order)):
        group = family(src)
        row = dict(source=str(src),family=group,source_sha256=digest(src))
        audit.append(row)
        if group in groups:
            row['status']='prior_or_duplicate_family'
            continue
        w = candidates / str(index)
        w.mkdir()
        target = w/'raw.aig'
        if src.suffix=='.aig':
            shutil.copyfile(src,target)
        else:
            if re.search(r'^\s*\.latch\b',src.read_text(errors='replace'),re.M):
                row['status']='sequential_source'
                continue
            staged = w/'source.blif'
            shutil.copyfile(src,staged)
            try:
                run_tool([binary,'-c',f'read_blif {tool_path(staged)}; strash; write_aiger {tool_path(target)}'],
                         w,'import',120)
            except ToolError:
                row['status']='import_failure'
                continue
        if not target.exists():
            row['status']='import_failure'
            continue
        with target.open('rb') as stream:
            header=stream.readline().split()
        if len(header)!=6 or header[0]!=b'aig' or int(header[3]) or int(header[4])==0 or int(header[5])==0:
            row['status']='out_of_scope'
            continue
        checksum, count = identity(target)
        raw = digest(target)
        row.update(aig=str(target),aig_sha256=raw,ordered_cones_sha256=checksum,ands=count)
        if raw in hashes or checksum in structures:
            row['status']='prior_or_duplicate_structure'
            continue
        if count < 100:
            row['status']='below_100_ands'
            continue
        hashes.add(raw)
        structures.add(checksum)
        groups.add(group)
        row['status']='eligible'
        eligible.append(row)
    ordered = sorted(eligible,key=lambda r:(r['ands'],r['family']))
    if len(ordered)<24:
        save_json(out/'audit.json',audit)
        raise ValueError('Fewer than 24 eligible families')
    panel = []
    for lower,upper in ((100,300),(300,1000),(1000,float('inf'))):
        group = [r for r in ordered if lower <= r['ands'] < upper]
        if len(group) < 8:
            raise ValueError('Insufficient families in size band '+str(lower))
        selected = sorted(group,key=lambda r:order(Path(r['source'])))[:8]
        panel.extend(dict(**r,size_band_min_ands=lower) for r in selected)
    save_json(out/'audit.json',audit)
    save_json(out/'panel.json',dict(rows=panel,original_rows=len(historical),eligible_families=len(ordered),
              import_abc_sha256=digest(binary),scope='No QoR labels used. Student training data ancestry is unverified.'))
    print(len(audit),len(ordered),len(panel),flush=True)


if __name__=='__main__':
    if 'SLURM_JOB_ID' not in os.environ:
        raise SystemExit('Slurm required')
    main(Path(sys.argv[1]),Path(sys.argv[2]))

"""Inspect archived no-edit structural fingerprints, retaining PI identities and PO order."""
import json
from pathlib import Path
import sys

from audit_replay_noop_v3 import fingerprints
from global_timing.structure import parse_aig
from global_timing.common import save_json
from paper14_frozen_choices import read

def main(audit, output):
    rows={}
    for name,case in read(audit/'freeze.json').items():
        paths=[Path(read(audit/name/k/'attempt.json')['aig']) for k in ('original','noop')]
        a,b=(fingerprints(parse_aig(p)) for p in paths)
        rows[name]={'aigs':list(map(str,paths)),'ands':[a['ands'],b['ands']],
                    'duplicates':[a['duplicate_structural_nodes'],b['duplicate_structural_nodes']],
                    'same_ordered_po_cones':a['output_hashes']==b['output_hashes'],
                    'same_internal_multiset':a['internal_hash_multiset']==b['internal_hash_multiset'],
                    'same_labeled_edges':a['labeled_graph_digest']==b['labeled_graph_digest']}
    save_json(output,rows)

if __name__=='__main__':
    main(Path(sys.argv[2]),Path(sys.argv[1])/'fingerprints.json')

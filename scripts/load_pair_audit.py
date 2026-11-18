"""Compare PO arrivals on a composed cover and an independently archived physical cover."""
import argparse
import csv
import json
from pathlib import Path
import sys
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from global_timing.boolean_candidates import parse_library
from global_timing.candidates import Catalogue
from global_timing.common import ROOT, digest, save_json
from global_timing.feasibility import CandidateFeasibility
from global_timing.liberty_cover import Liberty, TimingGraph, mapped_graph
from global_timing.load_bank import LoadBank
from global_timing.structure import parse_aig

def read(path):
    return json.loads(path.read_text())

@torch.no_grad()
def audit(args):
    lib = Liberty(args.tables, 'cuda')
    rules = parse_library(ROOT / 'projects/culs/src/algorithms/rewrite_library.inc')
    scores = {(int(r['circuit']), int(r['index'])): float(r['score'])
              for r in csv.DictReader(args.predictions.read_text().splitlines()) if float(r['alpha']) == 0.}
    rows = []
    for case, pair in [(75, (6, 8)), (179, (1, 7)), (421, (1, 7))]:
        frame = args.panel / f'case_{case:04d}'
        frozen = read(frame / 'frozen.json')
        original = Path(frozen['source']) / 'input.aig'
        assert digest(original) == frozen['input_sha256']
        aig = parse_aig(original)
        catalogue = Catalogue(**read(frame / 'catalogue.json'))
        bank = LoadBank(aig, catalogue, CandidateFeasibility(aig, catalogue, rules),
                        read(frame / 'scores.json')['scores'], rules, lib, args.abc, args.library,
                        args.output / f'case_{case}')
        vectors = {r['index']: r['choices'] for r in frozen['combinations']}
        for index in pair:
            estimated = bank.forward(bank.hard(vectors[index]), detail=True)[0]
            assert abs(float(estimated.max()) - scores[case,index]) < .01
            directory = frame / 'measurements' / f'combo_{index:02d}'
            record = read(directory / 'observation.json')
            metrics = record['metrics']
            assert record['choices'] == vectors[index] and metrics['cec'] == 'PASS'
            verilog = directory / 'evaluation' / 'mapped.v'
            assert digest(verilog) == metrics['mapped_sha256']
            pins, outputs, gates = mapped_graph(verilog, lib)
            physical = TimingGraph(lib, gates, pins, outputs)
            predicted = physical.forward(torch.empty(0, device='cuda'), detail=True)[0]
            abc = estimated.new_tensor([[r['rise_ps'], r['fall_ps']] for r in metrics['po_rise_fall']])
            torch.testing.assert_close(predicted, abc, atol=.015, rtol=3e-5)
            assert estimated.shape == abc.shape
            truth = abc.amax(dim=1)
            model = estimated.amax(dim=1)
            true_po, model_po = int(truth.argmax()), int(model.argmax())
            rows.append(dict(circuit=case, index=index, true_po=true_po, model_po=model_po,
                             real_ps=float(truth[true_po]), proxy_ps=float(model[model_po]),
                             proxy_at_true_po_ps=float(model[true_po]), real_at_model_po_ps=float(truth[model_po]),
                             fixed_cover_max_error_ps=float((predicted-abc).abs().max()),
                             true_mapped_sha256=metrics['mapped_sha256']))
    save_json(args.output/'endpoints.json',rows)

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('abc','library','tables','panel','predictions','output'):
        parser.add_argument('--'+name, type=Path, required=True)
    torch.set_num_threads(1)
    audit(parser.parse_args())

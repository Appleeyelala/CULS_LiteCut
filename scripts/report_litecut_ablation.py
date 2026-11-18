"""Report paired selections from the controlled three-seed local-delay ablation."""
import argparse
import csv
import json
import statistics
from pathlib import Path

from evaluate_v3_ranking import ranks

SEEDS = (20261117, 20261118, 20261119)
MODES = ('with', 'without')
TOL = .001
BASE = Path(__file__).resolve().parents[1] / 'analysis/joint_gradient_v3'


def read_csv(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with path.open('x') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def get_predictions(run, mode, seed):
    data = json.loads((run / 'output' / f'{mode}_{seed}' / 'summary.json').read_text())
    assert data['seed'] == seed and data['omit_litecut_delay'] == (mode == 'without')
    assert (data['variant'], data['global_context'], data['loss_mode']) == ('global', 'original', 'original')
    preds = [r for r in data['validation']['predictions'] if r['domain'] == 'sampled']
    assert len(preds) == 96
    return {(r['circuit_id'], r['index']): r for r in preds}


def write_circuit_table(path, selected):
    lines = ['# 有无 LiteCut 局部 Delay 的逐电路对照', '',
             '原 v3 全局残差，两组均从头训练。无 LiteCut 组在所有训练和评价域将归一化局部 Delay 变化置零，保留结构特征与编辑标志。每个种子在同一开发电路的 8 个冻结非空组合中选一个，预测并列容差为 0.001 ps，并列取最小编号。所有候选的历史实测记录均通过 CEC 和面积约束。',
             '', 'Regret 为选中组合实测 Delay 超出该域最低实测 Delay 的百分比。差值为有 LiteCut 减无 LiteCut 的实测 Delay，负值表示有 LiteCut 更快。三个种子重复评价同一批电路。',
             '', '[逐种子精确数值](selected.csv) · [全部组合预测和实测](all_combinations.csv) · [汇总](summary.csv)', '']
    for seed in SEEDS:
        lines.extend([f'## 种子 {seed}', '',
                      '| 电路 ID | 真实最优编号 | 最优 ps | 有 LiteCut 编号 | 实测 ps | Regret % | 无 LiteCut 编号 | 实测 ps | Regret % | 差值 ps |',
                      '|---:|:---|---:|---:|---:|---:|---:|---:|---:|---:|'])
        for r in selected:
            if r['seed'] == seed:
                lines.append('| {circuit_id} | {best_combinations} | {best_actual_ps:.3f} | '
                             '{with_combination} | {with_actual_ps:.3f} | {with_regret_percent:.3f} | '
                             '{without_combination} | {without_actual_ps:.3f} | {without_regret_percent:.3f} | '
                             '{with_minus_without_ps:+.3f} |'.format(**r))
        lines.append('')
    path.write_text('\n'.join(lines).rstrip() + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    labels = {(int(r['circuit_id']), int(r['index'])): r for r in read_csv(BASE / 'pilot/measured_combinations.csv')
              if r['role'] == 'validation' and int(r['index']) > 0}
    assert len(labels) == 96 and all(r['cec'] == 'PASS' and r['status'] == 'PASS' and
                                     r['area_feasible'] == 'True' for r in labels.values())
    ids = sorted({c for c, _ in labels})
    assert len(ids) == 12
    inputs = {int(r['circuit_id']): r['input'] for r in read_csv(BASE / 'pilot/circuit_accounting.csv')}
    old = {(int(r['seed']), int(r['circuit_id']), int(r['index'])): float(r['predicted_delay_ps'])
           for r in read_csv(BASE / 'ranking_loss/results/all_combinations.csv')
           if r['role'] == 'development' and r['model'] == 'v3_global'}
    all_combos, selected, summaries = [], [], []
    for seed in SEEDS:
        pred = {mode: get_predictions(args.run, mode, seed) for mode in MODES}
        assert all(set(p) == set(labels) for p in pred.values())
        for cid in ids:
            values = []
            for index in range(1, 9):
                key = cid, index
                label = labels[key]
                actual = float(label['candidate_delay_ps'])
                row = dict(seed=seed, circuit_id=cid, combination=index, kind=label['kind'], actual_ps=actual)
                for mode in MODES:
                    r = pred[mode][key]
                    assert r['actual_delay_ps'] == actual and r['area_feasible']
                    row[f'{mode}_pred_ps'] = r['predicted_delay_ps']
                row['archived_with_pred_ps'] = old[seed, cid, index]
                values.append(row)
            actual = [r['actual_ps'] for r in values]
            best = min(actual)
            best_indices = [i+1 for i, v in enumerate(actual) if v-best <= TOL]
            outcome = dict(seed=seed, circuit_id=cid, input=inputs[cid], best_combinations=';'.join(map(str,best_indices)),
                           best_actual_ps=best)
            for mode in MODES:
                predicted = [r[f'{mode}_pred_ps'] for r in values]
                rank = ranks(predicted, TOL)
                idx = rank.index(min(rank))
                for i, row in enumerate(values):
                    row[f'{mode}_rank'] = rank[i]
                    row[f'{mode}_selected'] = i == idx
                outcome.update({f'{mode}_combination': idx+1, f'{mode}_pred_ps': predicted[idx],
                                f'{mode}_actual_ps': actual[idx], f'{mode}_regret_percent': 100*(actual[idx]-best)/best,
                                f'{mode}_top1_hit': idx+1 in best_indices})
            outcome['with_minus_without_ps'] = outcome['with_actual_ps'] - outcome['without_actual_ps']
            selected.append(outcome)
            all_combos.extend(values)
    for seed_label in (*SEEDS, 'all'):
        rows = [r for r in selected if seed_label == 'all' or r['seed'] == seed_label]
        d = [r['with_minus_without_ps'] for r in rows]
        summary = dict(seed=seed_label, paired_cases=len(rows), with_wins=sum(v < -TOL for v in d),
                       same=sum(abs(v) <= TOL for v in d), without_wins=sum(v > TOL for v in d))
        for mode in MODES:
            regrets = [r[f'{mode}_regret_percent'] for r in rows]
            summary.update({f'{mode}_mean_regret_percent':statistics.mean(regrets),
                            f'{mode}_median_regret_percent':statistics.median(regrets),
                            f'{mode}_worst_regret_percent':max(regrets),
                            f'{mode}_hits':sum(r[f'{mode}_top1_hit'] for r in rows)})
        summaries.append(summary)
    args.output.mkdir(parents=True, exist_ok=False)
    write_csv(args.output / 'selected.csv', selected)
    write_csv(args.output / 'all_combinations.csv', all_combos)
    write_csv(args.output / 'summary.csv', summaries)
    write_circuit_table(args.output / 'SELECTED.md', selected)
    print('Maximum change from archived enabled predictions (ps):',
          max(abs(r['archived_with_pred_ps'] - r['with_pred_ps']) for r in all_combos))
    print(json.dumps(summaries, indent=2))


if __name__ == '__main__':
    main()

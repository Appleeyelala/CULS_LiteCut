"""Standalone scientific figures from audited real measurements only."""
from pathlib import Path
import math

from .common import digest, save_json
from .joint_results import geomean, matrix, paired


LABELS = {'legacy_cpu': 'Frozen STE CPU', 'adaptive_batch_cpu': 'Joint CPU',
          'adaptive_batch_auto': 'Joint auto', 'adaptive_no_inner_batch_auto': 'No inner updates',
          'random_batch': 'Random', 'greedy_litecut': 'LiteCut greedy',
          'original': 'Original', 'native_abc': 'Native ABC', 'native_culs': 'Native CULS'}
COLORS = {'legacy_cpu': '#202020', 'adaptive_batch_cpu': '#e69f00',
          'adaptive_batch_auto': '#d55e00', 'adaptive_no_inner_batch_auto': '#cc79a7',
          'random_batch': '#0072b2', 'greedy_litecut': '#56b4e9',
          'original': '#999999', 'native_abc': '#606c80', 'native_culs': '#009e73'}


def make_figures(rows, panel, protocol, summary, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({'font.size': 9, 'pdf.fonttype': 42, 'ps.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False})
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    lookup = matrix(rows, panel, protocol)
    artifacts = {}

    def save(fig, name):
        for extension in ('png', 'pdf'):
            path = output / (name + '.' + extension)
            if path.exists():
                raise FileExistsError(path)
            fig.savefig(path, dpi=170, bbox_inches='tight')
            artifacts[path.name] = digest(path)
        plt.close(fig)

    # Every plotted point is one design with all prescribed seeds measured.
    fig, ax = plt.subplots(figsize=(8.5, 6.))
    for variant in protocol['variants']:
        result = paired(lookup, panel, protocol, variant, 'legacy_cpu')
        points = [r for r in result['designs'] if r['complete_seeds']
                  and r['delay_ratio_geomean'] is not None and r['median_shared_allocation_speed']]
        ax.scatter([1 / r['median_shared_allocation_speed'] for r in points],
                   [r['delay_ratio_geomean'] for r in points], s=28, alpha=.78,
                   color=COLORS[variant], label=f'{LABELS[variant]} ({len(points)}/{len(panel["rows"])})')
    ax.axvline(1, color='#777777', lw=.8, ls='--')
    ax.axhline(1, color='#777777', lw=.8, ls='--')
    ax.set_xscale('log')
    ax.set_xlabel('E2E runtime / frozen STE CPU runtime (design median)')
    ax.set_ylabel('Mapped delay / frozen STE CPU delay (seed geometric mean)')
    ax.set_title('Measured quality and complete cost; area-feasible final points')
    ax.grid(alpha=.2)
    ax.legend(fontsize=8, ncol=2)
    fig.text(.02, -.015, 'Each point requires all seeds of one design. Failed/infeasible cases remain in tables; lower-left is better.', fontsize=8)
    fig.tight_layout()
    save(fig, 'delay_runtime_by_design')

    # Piecewise constant curves; no interpolation or prediction before a label.
    designs = panel['rows']
    columns = min(4, len(designs))
    lines = math.ceil(len(designs) / columns)
    fig, axes = plt.subplots(lines, columns, figsize=(4.1 * columns, 2.9 * lines), squeeze=False)
    curve_methods = ('legacy_cpu', protocol['primary_variant'], 'random_batch', 'greedy_litecut', 'native_abc', 'native_culs')
    for ax, design in zip(axes.flat, designs):
        for variant in curve_methods:
            selected = [lookup[design['index'], seed, variant] for seed in protocol['seeds']]
            if any(r['status'] != 'PASS' or r.get('noop_delay_ps', 0) <= 0 for r in selected):
                continue
            times = sorted({p['completed_s'] for r in selected for p in r['points']
                            if 0 < p['completed_s'] <= 600} | {600.})
            xx, yy = [], []
            for time in times:
                values = []
                for r in selected:
                    measured = [p['delay_ps'] for p in r['points'] if p['completed_s'] <= time
                                and p['area'] <= 1.05 * r['noop_area'] + 1e-8]
                    if not measured:
                        break
                    values.append(min(measured) / r['noop_delay_ps'])
                if len(values) == len(protocol['seeds']):
                    xx.append(time)
                    yy.append(geomean(values))
            if xx:
                ax.step(xx, yy, where='post', color=COLORS[variant], lw=1.25)
        ax.set_xscale('log')
        ax.set_xlim(.1, 600)
        ax.axhline(1, color='#999999', lw=.6, ls='--')
        ax.set_title(f'{design["index"]:04d} {Path(design["relative_path"]).stem[:24]}', fontsize=8)
        ax.ticklabel_format(axis='y', style='plain', useOffset=False)
        ax.grid(alpha=.2)
    for ax in list(axes.flat)[len(designs):]:
        ax.set_visible(False)
    handles = [Line2D([], [], color=COLORS[v], label=LABELS[v]) for v in curve_methods]
    # Put the legend outside the figure frame so its two rows cannot cover
    # the evidence note. bbox_inches='tight' retains it in PNG and PDF.
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, 1.01), ncol=3, fontsize=9)
    fig.text(.5, .01, 'Complete E2E seconds including input staging (log scale)', ha='center')
    fig.text(.005, .5, 'Best validated delay / measured no-op delay', rotation=90, va='center')
    fig.text(.5, .985, 'All seeds must have a validated area-feasible point; absent curves are not imputed.', ha='center', fontsize=8)
    fig.tight_layout(rect=(.02, .035, 1, .95))
    save(fig, 'validated_delay_time_curves')

    fig, ax = plt.subplots(figsize=(11, 4.5))
    order = sorted(designs, key=lambda r: (r['and_count'], r['index']))
    positions = {d['index']: i for i, d in enumerate(order)}
    kinds = {'matched': ('#009e73', 'o', 'CUDA, matching requests/work'),
             'changed': ('#d55e00', 'X', 'CUDA, mismatched work/requests'),
             'cpu': ('#999999', 'o', 'CPU or no proposal work')}
    for pair in summary['gpu']['pairs']:
        if pair['status'] != 'MEASURED' or pair['speed_ratio'] is None:
            continue
        qualified = (pair['same_request_sequence'] and pair['same_algorithm_and_inner_work']
                     and pair['same_eda_call_counts'] and pair['shared_allocation'])
        kind = 'cpu' if not pair['primary_cuda_used'] else 'matched' if qualified else 'changed'
        color, marker, _ = kinds[kind]
        seed_position = protocol['seeds'].index(pair['seed'])
        offset = (seed_position - (len(protocol['seeds']) - 1) / 2) * .13
        ax.scatter(positions[pair['index']] + offset, pair['speed_ratio'], color=color, marker=marker, s=35)
    ax.axhline(1, color='#777777', lw=.8)
    ax.axhline(1.5, color='#777777', lw=.8, ls='--')
    ax.set_yscale('log')
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([f'{d["index"]:04d}\n{d["and_count"]} AND' for d in order], fontsize=7)
    ax.set_ylabel('Same-algorithm E2E runtime: CPU / auto')
    ax.set_xlabel('Frozen designs ordered by input size; points are repeated seeds')
    ax.legend(handles=[Line2D([], [], color=c, marker=m, linestyle='', label=l) for c, m, l in kinds.values()], fontsize=8)
    ax.grid(axis='y', alpha=.2)
    fig.tight_layout()
    save(fig, 'gpu_contribution_by_design')

    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.6))
    for x, control in enumerate(('random_batch', 'adaptive_no_inner_batch_auto')):
        data = summary['comparisons'][control]['designs']
        available = [d for d in data if d['complete_seeds'] and d['delay_ratio_geomean'] is not None]
        for i, d in enumerate(available):
            offset = ((i % 5) - 2) * .035
            left.scatter(x + offset, d['delay_ratio_geomean'], s=25, alpha=.8, color=COLORS[control])
        left.text(x, 1.02, f'{len(available)}/{len(designs)} designs', transform=left.get_xaxis_transform(), ha='center', fontsize=8)
    left.axhline(1, color='#777777', lw=.8, ls='--')
    left.set_xticks([0, 1])
    left.set_xticklabels(['vs Random', 'vs No inner updates'])
    left.set_ylabel('Joint auto delay / control delay (design seed geomean)')
    left.grid(axis='y', alpha=.2)
    variants = ('legacy_cpu', protocol['primary_variant'], 'adaptive_no_inner_batch_auto')
    sources = [('joint_selection_gradient', 'Direct gradient', '#d55e00'),
               ('explicit_feasibility_repair', 'Explicit repair', '#e69f00'),
               ('calibration_warmup', 'Warmup', '#0072b2'),
               ('replay_noop', 'No-op', '#999999'),
               ('initialization_without_inner_updates', 'Initialization', '#009e73'),
               ('exploration_after_duplicate_projection', 'Fallback exploration', '#cc79a7')]
    bottom = [0] * len(variants)
    for source, label, color in sources:
        values = [summary['methods'][v]['final_sources'].get(source, 0) for v in variants]
        right.bar(range(len(variants)), values, bottom=bottom, color=color, label=label)
        bottom = [a + b for a, b in zip(bottom, values)]
    other = [summary['methods'][v]['successful_points'] - bottom[i] for i, v in enumerate(variants)]
    if any(other):
        right.bar(range(len(variants)), other, bottom=bottom, color='#555555', label='Other')
        bottom = [a + b for a, b in zip(bottom, other)]
    failed = [summary['methods'][v]['failed_method_points'] for v in variants]
    if any(failed):
        right.bar(range(len(variants)), failed, bottom=bottom, color='white', edgecolor='black', hatch='//', label='Failed process')
    right.set_xticks(range(len(variants)))
    right.set_xticklabels([LABELS[v] for v in variants], fontsize=8)
    right.set_ylabel('Final method points (design x seed)')
    right.legend(fontsize=7, loc='upper left', bbox_to_anchor=(1, 1))
    fig.text(.01, -.015, 'No-inner retains calibration / large-branch adjoint. Final source counts are not causal attribution of every timing gain.', fontsize=8)
    fig.tight_layout()
    save(fig, 'gradient_control_and_final_sources')
    save_json(output / 'manifest.json', {'artifacts_sha256': artifacts,
        'scope': 'Audited measured timing only. No Student/proxy predictions used as actual labels; repeated seeds retain design identity.'})
    return artifacts

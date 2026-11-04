"""Plot PPL decile scores and length-adjusted BC associations in one figure."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
import numpy as np

LANGUAGES = (('cpp', 'C++'), ('python', 'Python'), ('java', 'Java'))
PROMPTS = (('zs', 'ZS', '#0072B2', 'o', '-'),
           ('fs', 'FS', '#D55E00', 's', '--'),
           ('cot', 'CoT', '#009E73', '^', '-.'))


def plot(source, output):
    result = json.loads(source.read_text(encoding='utf8'))
    assert result['protocol']['controls'] == ['log mean token length', 'absolute log token length ratio']
    assert result['protocol']['bootstrap_repetitions'] == 2000
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8,
                        'axes.labelsize': 8.5, 'axes.titlesize': 10,
                        'xtick.labelsize': 7.5, 'ytick.labelsize': 8,
                        'pdf.fonttype': 42, 'axes.linewidth': .6})
    fig = plt.figure(figsize=(7.16, 5.65))
    grid = fig.add_gridspec(3, 3, left=.085, right=.985, bottom=.115,
                           top=.90, wspace=.23, hspace=.45,
                           height_ratios=[1, 1, .92])
    axes = [[fig.add_subplot(grid[r, c]) for c in range(3)] for r in range(3)]
    handles, plotted = [], []
    x = np.arange(1, 11)
    for col, (language, title) in enumerate(LANGUAGES):
        axes[0][col].set_title(title, pad=9, fontweight='bold')
        for row, target in enumerate(('accuracy', 'bc')):
            ax = axes[row][col]
            ax.set_xlim(.7, 10.3)
            ax.set_ylim(35, 100)
            ax.set_xticks(x)
            ax.set_yticks([40, 60, 80, 100])
            ax.grid(axis='y', color='#d9dfe5', linewidth=.55)
            if row == 0:
                ax.tick_params(labelbottom=False)
            else:
                ax.set_xlabel('PPL decile (low to high)', labelpad=4)
            if col == 0:
                ax.set_ylabel('Accuracy (%)' if row == 0 else 'BC (%)', labelpad=7)
            else:
                ax.tick_params(labelleft=False)
            for prompt, label, color, marker, style in PROMPTS:
                rows = sorted((r for r in result['deciles'] if r['language']==language
                              and r['feature']=='whole' and r['prompt']==prompt
                              and r['target']==target), key=lambda r:r['decile'])
                assert [r['decile'] for r in rows] == list(range(1, 11))
                assert all(r['pairs'] == 90 for r in rows)
                y = np.array([r['mean'] for r in rows])*100
                ci = np.array([r['ci95'] for r in rows])*100
                assert np.isfinite(ci).all() and ci.min()>=35 and ci.max()<=100
                ax.fill_between(x, ci[:, 0], ci[:, 1], color=color, alpha=.075, linewidth=0)
                line, = ax.plot(x, y, color=color, linewidth=1.05, marker=marker,
                                markersize=3, linestyle=style, markerfacecolor='white',
                                markeredgewidth=.8, label=label)
                if row == 0 and col == 0:
                    handles.append(line)
                plotted.extend(rows)
        ax = axes[2][col]
        ax.axvline(1, color='#737d87', linestyle='--', linewidth=.85, zorder=0)
        ax.set_xscale('log')
        ax.set_xlim(.5, 1.65)
        ax.xaxis.set_major_locator(FixedLocator([.5, .75, 1, 1.5]))
        ax.xaxis.set_major_formatter(FixedFormatter(['0.5', '0.75', '1', '1.5']))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_ylim(-.55, 2.55)
        ax.set_yticks([2, 1, 0], ['ZS', 'FS', 'CoT'])
        ax.set_xlabel('Adjusted BC odds ratio', labelpad=4)
        ax.grid(axis='x', color='#e4e8ec', linewidth=.5, zorder=0)
        for pi, (prompt, label, color, _, _) in enumerate(PROMPTS):
            for feature, offset, marker, fill in (('whole', .13, 'o', color),
                                                 ('block_median', -.13, 's', 'white')):
                rows = [r for r in result['regressions'] if r['language']==language
                        and r['feature']==feature and r['prompt']==prompt and r['target']=='bc']
                assert len(rows)==1
                r = rows[0]
                value = r['odds_ratio']
                lo, hi = r['ci95']
                assert .5 <= lo <= value <= hi <= 1.65
                ax.errorbar(value, 2-pi+offset, xerr=[[value-lo], [hi-value]],
                            fmt=marker, markersize=4, color=color, markerfacecolor=fill,
                            markeredgewidth=.9, elinewidth=1, capsize=2, capthick=.8)
                plotted.append(r)
        for row in range(3):
            ax = axes[row][col]
            ax.set_axisbelow(True)
            for side in ('top', 'right'):
                ax.spines[side].set_visible(False)
            for side in ('bottom', 'left'):
                ax.spines[side].set_color('#9ba7b3')
            ax.tick_params(length=2.5, width=.55, color='#8895a2', pad=3)
    fig.legend(handles, [p[1] for p in PROMPTS], loc='upper center',
               bbox_to_anchor=(.53, 1.001), ncol=3, frameon=False,
               handlelength=2.7, columnspacing=2.8, fontsize=8.5)
    summary_handles = [Line2D([], [], marker='o', linestyle='none', color='#444444',
                              markersize=4.5, label='Whole-program PPL'),
                       Line2D([], [], marker='s', linestyle='none', color='#444444',
                              markerfacecolor='white', markersize=4.5, label='Median block PPL')]
    fig.legend(handles=summary_handles, loc='lower center', bbox_to_anchor=(.53, .004),
               ncol=2, frameon=False, columnspacing=2.3, fontsize=8)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, metadata={'Title': 'Qwen3.8-27B code familiarity', 'Author': '',
                                 'Subject': 'PPL deciles and length-adjusted BC odds ratios',
                                 'Creator': 'Matplotlib'})
    plt.close(fig)
    assert len(plotted)==198
    return plotted


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, default=Path(__file__).resolve().parents[1]/'data/ppl/association_results.json')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    points = plot(args.input, args.output)
    print(json.dumps({'plotted_points':len(points), 'figure':str(args.output)}))

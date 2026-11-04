"""Render the model-by-language Accuracy and Both-Correct panels in Figure 3."""
import argparse
import numpy as np
from paper_tables import main as estimates

LANGS = ('cpp', 'python', 'java')
MODES = ('zs', 'fs', 'cot')


def render(report, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':9,
                         'legend.fontsize':8, 'pdf.fonttype':42})
    fig = plt.figure(figsize=(7.1, 4.3))
    axes = [fig.add_axes([.07, .26, .36, .59], projection='polar'),
            fig.add_axes([.57, .26, .36, .59], projection='polar')]
    conditions = [(lang, mode) for lang in LANGS for mode in MODES]
    language_names = {'cpp':'C++', 'python':'Python', 'java':'Java'}
    prompt_names = {'zs':'ZS', 'fs':'FS', 'cot':'CoT'}
    angles = np.arange(len(conditions)) * 2 * np.pi / len(conditions)
    closed_angles = np.append(angles, angles[0])
    models = [
        ('qwen3.8-27b', 'Qwen3.8-27B', '#2166ac', '-', 'o'),
        ('qwen3.8-flash', 'Qwen3.8-Flash', '#67a9cf', '--', 's'),
        ('glm-5.3-flash', 'GLM-5.3-Flash', '#238463', '-', '^'),
        ('deepseek-v4.1-flash', 'DeepSeek-V4.1-Flash', '#c56021', '--', 'D'),
        ('claude-haiku-5-5', 'Claude-Haiku-5-5', '#7b4da5', '-', 'v'),
        ('gpt-6-luna', 'GPT-6-Luna', '#a93b44', '--', 'P'),
    ]
    assert {m[0] for m in models} == set(report['model_means'])
    for ax, metric in zip(axes, ('accuracy', 'bc')):
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_thetagrids(np.degrees(angles),
                         [language_names[lang] + '\n' + prompt_names[mode]
                          for lang, mode in conditions],
                         fontsize=8.5, fontweight='bold')
        ax.tick_params(axis='x', pad=7)
        ax.set_ylim(50, 95)
        ax.set_yticks([50, 60, 70, 80, 90])
        ax.set_yticklabels(['50', '60', '70', '80', '90'],
                           color='#777777', fontsize=7)
        ax.set_rlabel_position(225)
        ax.grid(color='#d9d9d9', linestyle=':', linewidth=.7)
        ax.spines['polar'].set_color('#d9d9d9')
        for model, name, color, linestyle, marker in models:
            values = [report['model_language_prompt']['/'.join((model, lang, mode))][metric]
                      for lang, mode in conditions]
            assert all(50 <= value <= 95 for value in values)
            ax.plot(closed_angles, values + values[:1], color=color, linestyle=linestyle,
                    marker=marker, markersize=2.5, linewidth=1.2, label=name)
    handles, labels = axes[0].get_legend_handles_labels()
    # Matplotlib fills legend columns first; preserve the main-table row order.
    order = [0, 3, 1, 4, 2, 5]
    fig.legend([handles[i] for i in order], [labels[i] for i in order],
               frameon=False, ncol=3, loc='lower center', bbox_to_anchor=(.5, .015),
               handlelength=2.5, columnspacing=1.6, labelspacing=.8)
    fig.text(.25, .97, '(a) Accuracy (%)', ha='center', va='center', fontsize=10.5)
    fig.text(.75, .97, '(b) BC (%)', ha='center', va='center', fontsize=10.5)
    fig.savefig(output)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    render(estimates(), args.output)

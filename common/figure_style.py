"""Shared figure style (Frontiers figure requirements).

Process
  - Fixed colours for age groups, classifiers and phyla; Arial, smallest text 8 pt, lines at least 1 pt.
  - roc_panel(), confusion_panel(): ROC curves per age group and confusion matrix (counts and row percentages).
  - save(): writes a TIFF (LZW, RGB, 300 dpi, exactly 85 or 180 mm wide) and a PNG preview to the given folder.
"""
import os
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

GROUP_COLOR = {'10s': '#2a78d6', '20s_40s': '#eb6834', '50s_plus': '#1baf7a'}
GROUP_LABEL = {'10s': '10s', '20s_40s': '20s–40s', '50s_plus': '50s+'}
INK, MUTED, GRID, NEUTRAL = '#1f2328', '#6b7280', '#e5e7eb', '#9ca3af'
MODEL_COLOR = {'RF': '#2a78d6', 'XGB': '#eb6834', 'SVM': '#1baf7a', 'KNN': '#eda100', 'DT': '#e87ba4'}
PHYLUM_COLOR = {'Firmicutes': '#6250d6', 'Bacteroidota': '#e87ba4', 'Proteobacteria': '#008300'}
TYPE_LABEL = {'adolescent-associated': 'Adolescent-\nassociated', 'older-adult-associated': 'Older-adult-\nassociated',
              'progressive': 'Progressive'}

# Frontiers figure rules: width 85 mm (1 column) or 180 mm (2 columns), >= 300 dpi at final size, RGB,
# smallest text >= 8 pt; lines: plotted data lines 2 pt by default, every other visible line (axes, ticks, grid, outlines) at least 1 pt.
MM = 1 / 25.4
WIDTH = {'single': 85 * MM, 'double': 180 * MM}
MIN_FONT = 8
MIN_LINE = 1.0

plt.rcParams.update({
    'font.family': 'Arial', 'font.size': 8, 'axes.titlesize': 9, 'axes.labelsize': 8,
    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8, 'legend.title_fontsize': 8,
    'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED,
    'axes.linewidth': 1.0, 'xtick.major.width': 1.0, 'ytick.major.width': 1.0, 'xtick.minor.width': 1.0,
    'ytick.minor.width': 1.0, 'lines.linewidth': 2.0, 'patch.linewidth': 1.0,
    'boxplot.boxprops.linewidth': 1.0, 'boxplot.whiskerprops.linewidth': 1.0, 'boxplot.capprops.linewidth': 1.0,
    'boxplot.medianprops.linewidth': 2.0,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': False, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.04,
    # italic taxon names are written as mathtext; render them in Arial like the rest of the text
    'mathtext.fontset': 'custom', 'mathtext.rm': 'Arial', 'mathtext.it': 'Arial:italic', 'mathtext.bf': 'Arial:bold',
})


def panel_label(ax, letter, x=-0.12, y=1.04):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=11, fontweight='bold', color=INK, va='bottom')


def ygrid(ax):
    ax.grid(axis='y', color=GRID, linewidth=1.0)
    ax.set_axisbelow(True)


def taxon(name):
    """Italic taxon name (genus, phylum) as mathtext. Placeholder names of uncultured lineages such as UMGS1071 stay
    upright, as in the manuscript text."""
    if re.match(r'^[A-Z]{2,}[0-9]', name):
        return name
    return r'$\mathit{' + name.replace('_', r'\_') + '}$'


def short_genus(name):
    parts = name.split(';')
    g = parts[-1] if parts[-1] != '__' else parts[-2] + ' (unclassified genus)'
    return g.replace('g__', '').replace('f__', '')


def enforce_minimums(fig):
    """Raise every text below MIN_FONT pt and every visible line below MIN_LINE pt (Frontiers rules)."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.text import Text
    fig.canvas.draw()   # tick labels are created at draw time
    for ax in fig.axes:
        for name, axis in (('x', ax.xaxis), ('y', ax.yaxis)):
            for which, labels in (('major', axis.get_majorticklabels()), ('minor', axis.get_minorticklabels())):
                if labels and min(t.get_fontsize() for t in labels) < MIN_FONT:
                    ax.tick_params(axis=name, which=which, labelsize=MIN_FONT)
    for t in fig.findobj(Text):
        if t.get_text().strip() and t.get_fontsize() < MIN_FONT:
            t.set_fontsize(MIN_FONT)
    for ln in fig.findobj(Line2D):
        if ln.get_visible() and ln.get_linestyle() not in ('None', '') and 0 < ln.get_linewidth() < MIN_LINE:
            ln.set_linewidth(MIN_LINE)
    for pt in fig.findobj(Patch):
        lw = pt.get_linewidth()
        if pt.get_visible() and pt.get_edgecolor()[3] > 0 and 0 < lw < MIN_LINE:
            pt.set_linewidth(MIN_LINE)


def save(fig, name, outdir, width='double'):
    """Write a Frontiers-ready TIFF (exactly 85 or 180 mm wide at 300 dpi, RGB, LZW) and a PNG preview.
    The figure is rendered with a tight bounding box; narrower output is padded with white to the exact column
    width, wider output raises an error so that the layout is fixed instead of the figure being shrunk."""
    import io
    from PIL import Image
    os.makedirs(outdir, exist_ok=True)
    enforce_minimums(fig)
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=300, facecolor='white')
    plt.close(fig)
    im = Image.open(buf).convert('RGB')
    target = round(WIDTH[width] * 300)
    if im.width > target:
        raise SystemExit(f'{name}: {im.width / 300 / MM:.0f} mm wide, exceeds {width} column ({target / 300 / MM:.0f} mm)')
    if im.height > round(230 * MM * 300):
        raise SystemExit(f'{name}: {im.height / 300 / MM:.0f} mm tall, exceeds one page')
    canvas = Image.new('RGB', (target, im.height), 'white')
    canvas.paste(im, ((target - im.width) // 2, 0))
    canvas.save(os.path.join(outdir, f'{name}.tiff'), dpi=(300, 300), compression='tiff_lzw')
    canvas.save(os.path.join(outdir, f'{name}.png'), dpi=(300, 300))
    print(f'saved {name}: {canvas.width / 300 / MM:.0f} x {canvas.height / 300 / MM:.0f} mm, RGB, 300 dpi '
          f'(content {im.width / 300 / MM:.0f} mm wide)')


ROC_GROUPS = ['10s', '20s_40s', '50s_plus']


def roc_panel(ax, y, prob, ci_table=None, title=''):
    for c, g in enumerate(ROC_GROUPS):
        fpr, tpr, _ = roc_curve(y == c, prob[:, c])
        auc = roc_auc_score(y == c, prob[:, c])
        lab = f'{GROUP_LABEL[g]}  {auc:.2f}'
        if ci_table is not None:
            r = ci_table.loc[f'auc_{g}']
            lab += f' [{r.ci_low:.2f}–{r.ci_high:.2f}]'
        ax.plot(fpr, tpr, color=GROUP_COLOR[g], linewidth=2, label=lab)
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=0.8)
    style_roc(ax, title)


def style_roc(ax, title):
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.set_aspect('equal')
    ax.set_xlabel('False positive rate')
    ax.set_ylabel('True positive rate')
    ax.set_title(title, fontsize=7.8, color=INK)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.24), title='AUC', handlelength=1.2, handletextpad=0.4, frameon=False)


def confusion_panel(ax, y, pred, title):
    cm = confusion_matrix(y, pred, labels=range(3))
    share = cm / cm.sum(axis=1, keepdims=True)
    ax.imshow(share, cmap='Blues', vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f'{cm[i, j]}\n({share[i, j]:.0%})', ha='center', va='center', fontsize=6.8,
                    color='white' if share[i, j] > 0.55 else INK)
    ax.set_xticks(range(3))
    ax.set_xticklabels([GROUP_LABEL[g] for g in ROC_GROUPS])
    ax.set_yticks(range(3))
    ax.set_yticklabels([GROUP_LABEL[g] for g in ROC_GROUPS])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, fontsize=7.8, color=INK)

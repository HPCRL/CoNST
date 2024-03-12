import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.patches
from matplotlib.legend_handler import HandlerTuple
from matplotlib.ticker import ScalarFormatter
from process_log import get_data_3c

FONTSIZE=15


def make_graph(nofilter_frame, name="nofilter"):
    # relative speedup
    m1_t = nofilter_frame
    # make a new pandas frame by dividing each element of m1_t by the corresponding element of m1_t['const']
    new_frame = pd.DataFrame()
    for col in m1_t.columns:
        new_frame[col] = m1_t[col] / m1_t['const']
    xfactor = new_frame


    fig, ax1 = plt.subplots()
    ax1.spines["top"].set_visible(True)
    ax1.spines["right"].set_visible(True)
    ax1.spines["bottom"].set_visible(True)
    ax1.spines["left"].set_visible(True)
    x_label = ['small']
    if len(m1_t) > 1:
        x_label.append('medium')
    if len(m1_t) > 2:
        x_label.append('large')
    x = np.arange(len(x_label))

    #hatch_dict = {'const': '', 'TACO Unfused': 'xx', 'TACO N-ary': '++', 'Sparta': '...'}
    hatch_dict = {'const': '', 'SparseLNR': '////', 'TACO-unfused': 'xx', 'TACO-Nary': '++', 'Sparta': '...'}
    #labels = ['const', 'TACO Unfused', 'TACO N-ary', 'Sparta']
    labels = ['const', 'SparseLNR', 'TACO-unfused', 'TACO-Nary', 'Sparta']
    colors = {'TACO-Nary': 'red', 'SparseLNR': 'greenyellow', 'const':'#FF7722', 'TACO-unfused': 'dodgerblue', 'Sparta': 'yellow'}

    width = 0.3

    all_rects = []
    for ind,key in enumerate(m1_t.keys()):
        all_rects.append(ax1.bar(x - ind*width, m1_t[key], width/2, label=key, hatch=hatch_dict[key], alpha=.99,
            color=colors[key]))

    ax1.tick_params(axis='y', labelsize=FONTSIZE)
    ax1.set_yscale('log', base=10)
    plt.xticks(x, x_label)
    for axis in [ax1.yaxis]:
        formatter = ScalarFormatter()
        formatter.set_scientific(False)
        axis.set_major_formatter(formatter)

    plt.tick_params(axis='x', which='major', labelsize=FONTSIZE)
    plt.xticks(rotation=0)

    ax1.yaxis.grid(linestyle='--', linewidth='0.5')
    ax1.set_ylabel('Wall time in milliseconds, log scale', size=FONTSIZE)

    def autolabel(rects, vals):
        """Attach a text label above each bar in *rects*, displaying its height."""
        i = 0
        for rect in rects:
            height = rect.get_height()
            val = vals[i]
            i = i + 1
            ax1.annotate('{:4.2f}'.format(val),
                    xy=(rect.get_x()+width/2, height),
                         xytext=(0, 10),  # 3 points vertical offset
                         textcoords="offset points",
                         ha='right', va='center', size=FONTSIZE, fontweight='bold', rotation=0)

    for ind,k in enumerate(m1_t.keys()):
        autolabel(all_rects[ind], xfactor[k])

    leg_artists = []
    methods = ['const', 'SparseLNR', 'TACO-unfused', 'TACO-Nary', 'Sparta']
    for i in range(len(methods)):
        p = matplotlib.patches.Patch(
            facecolor=colors[methods[i]], hatch=hatch_dict[methods[i]], alpha=.99)
        leg_artists.append((p))

    ax1.legend(leg_artists, labels, loc='upper center', bbox_to_anchor=(0.45, 1.2), ncol=6,
               handler_map={tuple: HandlerTuple(ndivide=None)}, fontsize=10)

    #fig.set_figheight(10)
    #fig.set_figwidth(15)
    fig.tight_layout()

    plt.show()

    fig.savefig(f'mp2_{name}.pdf')


if __name__ == "__main__":
    # execute only if run as a script
    filter_frame, nofilter_frame = get_data_3c()
    make_graph(nofilter_frame, "nofilter")
    make_graph(filter_frame, "filter")

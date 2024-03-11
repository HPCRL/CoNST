import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.patches
from matplotlib.legend_handler import HandlerTuple
from matplotlib.ticker import ScalarFormatter
from process_log import get_data

FONTSIZE=30


def nofilter(nofilter_frame):
    # relative speedup
    #m1_t = pd.DataFrame({
    #   'TACO N-ary': [2142.74, 1181460, 5526100],
    #   'Sparta': [563.5, 65164.6, 202910],
    #   'TACO Unfused': [119.566, 24936.1, 82652.4],
    #   'const': [25.98, 4841.5, 15729.2],
    #   'SparseLNR': [2140.7, 1165700, 5446290]
    #})
    m1_t = nofilter_frame

    xfactor = pd.DataFrame({
        'TACO N-ary': [82.44, 244, 351],
        'Sparta': [21.68, 13.45, 12.9],
        'TACO Unfused': [5.55, 5.25, 5.15],
        'const': [1.0, 1.0, 1.0],
        'SparseLNR': [82.47, 240.84, 346.45]
    })

    fig, ax1 = plt.subplots()
    ax1.spines["top"].set_visible(True)
    ax1.spines["right"].set_visible(True)
    ax1.spines["bottom"].set_visible(True)
    ax1.spines["left"].set_visible(True)
    x_label = ['small', 'medium', 'large']
    x = np.arange(len(x_label))

    #hatch_dict = {'const': '', 'TACO Unfused': 'xx', 'TACO N-ary': '++', 'Sparta': '...'}
    hatch_dict = {'const': '', 'SparseLNR': '////', 'TACO Unfused': 'xx', 'TACO N-ary': '++', 'Sparta': '...'}
    #labels = ['const', 'TACO Unfused', 'TACO N-ary', 'Sparta']
    labels = ['const', 'SparseLNR', 'TACO Unfused', 'TACO N-ary', 'Sparta']
    colors = {'TACO N-ary': 'red', 'SparseLNR': 'greenyellow', 'const':'#FF7722', 'TACO Unfused': 'dodgerblue', 'Sparta': 'yellow'}

    width = 0.3

    #rects1 = ax1.bar(x - 0.75*width, m1_t['const'], width/2, label=labels[0], hatch=hatch_dict['const'], alpha=.99,
    #                 color=colors['const'])
    #rects2 = ax1.bar(x - 0.25*width, m1_t['TACO Unfused'], width/2, label=labels[1], hatch=hatch_dict['TACO Unfused'], alpha=.99,
    #                 color=colors['TACO Unfused'])
    #rects3 = ax1.bar(x + 0.25*width, m1_t['TACO N-ary'], width/2, label=labels[2], hatch=hatch_dict['TACO N-ary'], alpha=.99,
    #                 color=colors['TACO N-ary'])
    #rects4 = ax1.bar(x + 0.75*width, m1_t['Sparta'], width/2, label=labels[3], hatch=hatch_dict['Sparta'], alpha=.99,
    #                 color=colors['Sparta'])
    rects1 = ax1.bar(x - 1*width, m1_t['const'], width/2, label="const", hatch=hatch_dict["const"], alpha=.99,
                     color=colors['const'])
    rects2 = ax1.bar(x - 0.5*width, m1_t['SparseLNR'], width/2, label="SparseLNR", hatch=hatch_dict['SparseLNR'], alpha=.99,
                     color=colors['SparseLNR'])
    rects3 = ax1.bar(x, m1_t['TACO Unfused'], width/2, label='TACO Unfused', hatch=hatch_dict['TACO Unfused'], alpha=.99,
                     color=colors['TACO Unfused'])
    rects4 = ax1.bar(x + 0.5*width, m1_t['TACO N-ary'], width/2, label='TACO N-ary', hatch=hatch_dict['TACO N-ary'], alpha=.99,
                     color=colors['TACO N-ary'])
    rects5 = ax1.bar(x + 1.0*width, m1_t['Sparta'], width/2, label='Sparta', hatch=hatch_dict['Sparta'], alpha=.99,
                     color=colors['Sparta'])

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
    #ax1.set_ylabel('Wall time in milliseconds, log scale', size=FONTSIZE)

    def autolabel(rects, vals):
        """Attach a text label above each bar in *rects*, displaying its height."""
        i = 0
        for rect in rects:
            height = rect.get_height()
            val = vals[i]
            i = i + 1
            ax1.annotate('{:4.0f}'.format(val),
                    xy=(rect.get_x()+width/2, height),
                         xytext=(0, 10),  # 3 points vertical offset
                         textcoords="offset points",
                         ha='right', va='center', size=FONTSIZE, fontweight='bold', rotation=0)

    autolabel(rects1, xfactor['const'])
    autolabel(rects2, xfactor['SparseLNR'])
    autolabel(rects3, xfactor['TACO Unfused'])
    autolabel(rects4, xfactor['TACO N-ary'])
    autolabel(rects5, xfactor['Sparta'])

    leg_artists = []
    methods = ['const', 'SparseLNR', 'TACO Unfused', 'TACO N-ary', 'Sparta']
    for i in range(len(methods)):
        p = matplotlib.patches.Patch(
            facecolor=colors[methods[i]], hatch=hatch_dict[methods[i]], alpha=.99)
        leg_artists.append((p))

    ax1.legend(leg_artists, labels, loc='upper center', bbox_to_anchor=(0.45, 1.2), ncol=6,
               handler_map={tuple: HandlerTuple(ndivide=None)}, fontsize=21)

    fig.set_figheight(10)
    fig.set_figwidth(15)
    fig.tight_layout()

    plt.show()

    fig.savefig('mp2_nofilter.pdf')


def with_filter(filter_frame):
    # relative speedup
    m1_t = filter_frame

    xfactor = pd.DataFrame({
        'TACO N-ary': [51.93, 207, 220],
        'TACO Unfused': [11.92, 38.29, 21.68],
        'const': [1.0, 1.0, 1.0],
        'SparseLNR': [51.13, 205.92, 218.9],
    })

    fig, ax1 = plt.subplots()
    ax1.spines["top"].set_visible(True)
    ax1.spines["right"].set_visible(True)
    ax1.spines["bottom"].set_visible(True)
    ax1.spines["left"].set_visible(True)
    x_label = ['small', 'medium', 'large']
    x = np.arange(len(x_label))

    hatch_dict = {'const': '', 'SparseLNR': '////', 'TACO Unfused': 'xx', 'TACO N-ary': '++', 'Sparta': '...'}
    labels = ['const', 'SparseLNR', 'TACO Unfused', 'TACO N-ary', 'Sparta']
    colors = {'TACO N-ary': 'red', 'SparseLNR': 'greenyellow', 'const':'#FF7722', 'TACO Unfused': 'dodgerblue', 'Sparta': 'yellow'}
    width = 0.3

    #rects1 = ax1.bar(x - width, m1_t['const'], width, label=labels[0], hatch=hatch_dict['const'], alpha=.99,
    #                 color=colors['const'])
    #rects2 = ax1.bar(x, m1_t['TACO Unfused'], width, label=labels[1], hatch=hatch_dict['TACO Unfused'], alpha=.99,
    #                 color=colors['TACO Unfused'])
    #rects3 = ax1.bar(x + width, m1_t['TACO N-ary'], width, label=labels[2], hatch=hatch_dict['TACO N-ary'], alpha=.99,
    #                 color=colors['TACO N-ary'])
    rects1 = ax1.bar(x - 0.75*width, m1_t['const'], width/2, label=labels[0], hatch=hatch_dict['const'], alpha=.99,
                     color=colors['const'])
    rects2 = ax1.bar(x - 0.25*width, m1_t['SparseLNR'], width/2, label=labels[1], hatch=hatch_dict['SparseLNR'], alpha=.99,
                     color=colors['SparseLNR'])
    rects3 = ax1.bar(x + 0.25*width, m1_t['TACO Unfused'], width/2, label=labels[2], hatch=hatch_dict['TACO Unfused'], alpha=.99,
                     color=colors['TACO Unfused'])
    rects4 = ax1.bar(x + 0.75*width, m1_t['TACO N-ary'], width/2, label=labels[3], hatch=hatch_dict['TACO N-ary'], alpha=.99,
                     color=colors['TACO N-ary'])

    ax1.tick_params(axis='y', labelsize=FONTSIZE)
    ax1.set_yscale('log', base=10)
    plt.xticks(x + width / 4, x_label)
    for axis in [ax1.yaxis]:
        formatter = ScalarFormatter()
        formatter.set_scientific(False)
        axis.set_major_formatter(formatter)


    plt.tick_params(axis='x', which='major', labelsize=FONTSIZE)
    plt.xticks(rotation=0)

    ax1.yaxis.grid(linestyle='--', linewidth='0.5')
    #ax1.set_ylabel('Wall time in milliseconds, log scale', size=FONTSIZE)

    def autolabel(rects, vals):
        """Attach a text label above each bar in *rects*, displaying its height."""
        i = 0
        for rect in rects:
            height = rect.get_height()
            val = vals[i]
            i = i + 1
            ax1.annotate('{:4.0f}'.format(val),
                         xy=(rect.get_x()+width/2, height),
                         xytext=(0, 10),  # 3 points vertical offset
                         textcoords="offset points",
                         ha='right', va='center', size=FONTSIZE, fontweight='bold', rotation=0)

    autolabel(rects1, xfactor['const'])
    autolabel(rects2, xfactor['SparseLNR'])
    autolabel(rects3, xfactor['TACO Unfused'])
    autolabel(rects4, xfactor['TACO N-ary'])

    leg_artists = []
    methods = ['const', 'SparseLNR', 'TACO Unfused', 'TACO N-ary']
    for i in range(len(methods)):
        p = matplotlib.patches.Patch(
            facecolor=colors[methods[i]], hatch=hatch_dict[methods[i]], alpha=.99)
        # linem = plt.plot([], [], markers[i], markerfacecolor=colors[i], markeredgecolor=colors[i], color=colors[i])[0]
        leg_artists.append((p))

    ax1.legend(leg_artists, labels, loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=5,
               handler_map={tuple: HandlerTuple(ndivide=None)}, fontsize=22)

    # tt = 15.7
    # textstr = ('machine peak = %.1f TFLOPS' % (tt))
    #
    # # place a text box in upper left in axes coords
    # props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    # ax1.text(0.03, 0.95, textstr, transform=ax1.transAxes, fontsize=18,
    #          verticalalignment='top', bbox=props)

    fig.set_figheight(10)
    fig.set_figwidth(13)
    fig.tight_layout()

    plt.show()

    fig.savefig('mp2_filter.pdf')


if __name__ == "__main__":
    # execute only if run as a script
    filter_frame, nofilter_frame = get_data()
    nofilter(nofilter_frame)
    with_filter(filter_frame)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.patches
from process_log import get_data_ttmc
from matplotlib.legend_handler import HandlerTuple
from matplotlib.ticker import ScalarFormatter

FONTSIZE=15
SPEEDUPSIZE=20

hatch_dict = {'const': '', 'SparseLNR': '////', 'TACO-unfused': 'xx', 'TACO-Nary': '++', 'Sparta': '...'}
labels = ['const', 'SparseLNR', 'TACO-unfused', 'TACO-Nary', 'Sparta']
colors = {'TACO-Nary': 'red', 'SparseLNR': 'greenyellow', 'const':'#FF7722', 'TACO-unfused': 'dodgerblue', 'Sparta': 'yellow'}

def draw(tensor_name, frame):
    # relative speedup
    m1_t = frame
    new_frame = pd.DataFrame()
    for col in m1_t.columns:
        new_frame[col] = m1_t[col] / m1_t['const']
    xfactor = new_frame

    fig, ax1 = plt.subplots()
    ax1.spines["top"].set_visible(True)
    ax1.spines["right"].set_visible(True)
    ax1.spines["bottom"].set_visible(True)
    ax1.spines["left"].set_visible(True)
    #x_label = ['Nell2-1', 'Nell2-2', 'Nell2-3', 'Total']
    x_label = [tname+f"-{i+1}" for i in range(len(m1_t))]
    x = np.arange(len(x_label))

    width = 0.4


    #rects1 = ax1.bar(x - 1*width, m1_t['CoNST'], width/2, label="CoNST", hatch=hatch_dict["CoNST"], alpha=.99,
    #                 color=colors['CoNST'])
    #rects2 = ax1.bar(x - 0.5*width, m1_t['SparseLNR'], width/2, label="SparseLNR", hatch=hatch_dict['SparseLNR'], alpha=.99,
    #                 color=colors['SparseLNR'])
    #rects3 = ax1.bar(x, m1_t['TACO Unfused'], width/2, label='TACO Unfused', hatch=hatch_dict['TACO Unfused'], alpha=.99,
    #                 color=colors['TACO Unfused'])
    #rects4 = ax1.bar(x + 0.5*width, m1_t['TACO N-ary'], width/2, label='TACO N-ary', hatch=hatch_dict['TACO N-ary'], alpha=.99,
    #                 color=colors['TACO N-ary'])

    all_rects = []
    for ind,key in enumerate(m1_t.keys()):
        all_rects.append(ax1.bar(x - ind*width, m1_t[key], width/2, label=key, hatch=hatch_dict[key], alpha=.99,
            color=colors[key]))

    ax1.set_yscale('log', base=2)
    for axis in [ax1.yaxis]:
        formatter = ScalarFormatter()
        formatter.set_scientific(False)
        axis.set_major_formatter(formatter)

    ax1.tick_params(axis='y', labelsize=FONTSIZE)
    #ax1.set_ylim(0, 1600)
    # ax1.set_xticklabels(x_label)
    plt.xticks(x, x_label)

    plt.tick_params(axis='x', which='major', labelsize=FONTSIZE)
    plt.xticks(rotation=0)

    # markers = ['o-', '^-', 'x-']
    # ax2 = ax1.twinx()
    # ax2.plot(x_label, m2_t['tvm_time'], markers[0], label='tvm_time', color="blue")
    # ax2.tick_params(axis='y', labelsize=16, labelcolor='blue')
    # ax2.set_ylim(0, 0.55)
    # ax2.set_yticks(ax2.get_yticks()[-3:])

    ax1.yaxis.grid(linestyle='--', linewidth='0.5')
    ax1.set_ylabel('Wall time in milliseconds, log scaled', size=FONTSIZE)
    # ax2.set_ylabel('TVM execution time', size=18, )
    # ax2.yaxis.label.set_color('blue')

    def autolabel(rects, vals):
        """Attach a text label above each bar in *rects*, displaying its height."""
        i = 0
        for rect in rects:
            height = rect.get_height()
            val = vals[i]
            i = i + 1
            ax1.annotate('{:4.2f}'.format(val),
                         xy=(rect.get_x() + 0.17, height+(i % 2 == 0)*20),
                         xytext=(0, 10),  # 3 points vertical offset
                         textcoords="offset points",
                         ha='right', va='center', size=SPEEDUPSIZE, fontweight='bold', rotation=0)

    #def autolabel(rects, vals):
    #    """Attach a text label above each bar in *rects*, displaying its height."""
    #    i = 0
    #    for rect in rects:
    #        height = rect.get_height()
    #        val = vals[i]
    #        i = i + 1
    #        import pdb
    #        pdb.set_trace()
    #        ax1.annotate(val,
    #                     xy=(rect.get_x() + 0.17, height+(i % 2 == 0)*20),
    #                     xytext=(0, 10),  # 3 points vertical offset
    #                     textcoords="offset points",
    #                     ha='right', va='center', size=SPEEDUPSIZE, fontweight='bold', rotation=0)

    #autolabel(rects1, xfactor['CoNST'])
    #autolabel(rects2, xfactor['SparseLNR'])
    #autolabel(rects3, xfactor['TACO Unfused'])
    #autolabel(rects4, xfactor['TACO N-ary'])
    for ind,k in enumerate(m1_t.keys()):
        autolabel(all_rects[ind], xfactor[k])

    leg_artists = []
    methods = ['const', 'SparseLNR', 'TACO-unfused', 'TACO-Nary', 'Sparta']
    for i in range(len(methods)):
        p = matplotlib.patches.Patch(
            facecolor=colors[methods[i]], hatch=hatch_dict[methods[i]], alpha=.99)
        # linem = plt.plot([], [], markers[i], markerfacecolor=colors[i], markeredgecolor=colors[i], color=colors[i])[0]
        leg_artists.append((p))

    ax1.legend(leg_artists, labels, loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=5,
               handler_map={tuple: HandlerTuple(ndivide=None)}, fontsize=FONTSIZE)

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

    fig.savefig(f"ttmc_{tensor_name}.pdf")



if __name__ == "__main__":
    # execute only if run as a script
    # drawline()

    tensor_frame_dict = get_data_ttmc()
    for tname, frame in tensor_frame_dict.items():
        draw(tname, frame)

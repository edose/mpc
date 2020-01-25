__author__ = "Eric Dose :: New Mexico Mira Project, Albuquerque"

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.text import Text
from matplotlib.image import AxesImage
from matplotlib.colors import to_rgba
import numpy as np
from numpy.random import rand


def try_simple_pick():
    # simple picking, lines, rectangles and text
    # from first example in https://matplotlib.org/examples/event_handling/pick_event_demo.html
    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.set_title('click on points, rectangles or text', picker=True)
    ax1.set_ylabel('ylabel', picker=True, bbox=dict(facecolor='red'))
    line, = ax1.plot(rand(100), 'o', picker=5)  # 5 points tolerance
    bars = ax2.bar(range(10), rand(10), picker=True)  # pick the rectangle

    for label in ax2.get_xticklabels():  # make the xtick labels pickable
        label.set_picker(True)

    def onpick1(event):
        if isinstance(event.artist, Line2D):
            thisline = event.artist
            xdata = thisline.get_xdata()
            ydata = thisline.get_ydata()
            i = event.ind[0]
            print('onpick1 line...', i, xdata[i], ydata[i])
        elif isinstance(event.artist, Rectangle):
            patch = event.artist
            print('onpick1 patch:', patch.get_path())
        elif isinstance(event.artist, Text):
            text = event.artist
            print('onpick1 text:', text.get_text())

    fig.canvas.mpl_connect('pick_event', onpick1)
    plt.show()


def try_scatter_pick():
    # picking on a scatter plot (matplotlib.collections.RegularPolyCollection)
    # adapted from third example in https://matplotlib.org/examples/event_handling/pick_event_demo.html

    x, y, sizes = rand(3, 100)
    original_colors = 25 * ['gray'] + 75 * ['yellow']
    original_rgba = [to_rgba(color) for color in original_colors]
    selected_rgba = to_rgba('red')
    now_selected = 100 * [False]

    def onpick3(event):
        index_list = event.ind
        i = index_list[0]
        print('onpick3 scatter:', i, np.take(x, index_list), np.take(y, index_list))
        if now_selected[i]:
            new_rgba = original_rgba[i]
            col._facecolors[index_list, :] = new_rgba
            now_selected[i] = False
        else:
            new_rgba = selected_rgba
            col._facecolors[index_list, :] = new_rgba
            now_selected[i] = True
        print('color now:', i, new_rgba)
        fig.canvas.draw()

    fig, ax = plt.subplots()
    col = ax.scatter(x, y, 100 * sizes, original_colors, picker=True)
    # fig.savefig('pscoll.eps')
    fig.canvas.mpl_connect('pick_event', onpick3)
    plt.show()


def try_tooltips():
    import matplotlib.pyplot as plt
    import numpy as np;
    np.random.seed(1)

    x = np.random.rand(15)
    y = np.random.rand(15)
    names = np.array(list("ABCDEFGHIJKLMNO"))
    c = np.random.randint(1, 5, size=15)

    norm = plt.Normalize(1, 4)
    cmap = plt.cm.RdYlGn

    fig, ax = plt.subplots()
    sc = plt.scatter(x, y, c=c, s=100, cmap=cmap, norm=norm)

    annot = ax.annotate("", xy=(0, 0), xytext=(20, 20), textcoords="offset points",
                        bbox=dict(boxstyle="round", fc="w"),
                        arrowprops=dict(arrowstyle="->"))
    annot.set_visible(False)

    def update_annot(ind):
        # adapted from https://stackoverflow.com/questions/7908636/
        #     possible-to-make-labels-appear-when-hovering-over-a-point-in-matplotlib .
        # (I can't figure advice in comment to check for change in ind['ind'])
        pos = sc.get_offsets()[ind["ind"][0]]
        annot.xy = pos
        text = "{}, {}".format(" ".join(list(map(str, ind["ind"]))),
                               " ".join([names[n] for n in ind["ind"]]))
        annot.set_text(text)
        annot.get_bbox_patch().set_facecolor(cmap(norm(c[ind["ind"][0]])))
        annot.get_bbox_patch().set_alpha(0.4)

    def hover(event):
        vis = annot.get_visible()
        if event.inaxes == ax:
            cont, ind = sc.contains(event)
            if cont:
                update_annot(ind)
                annot.set_visible(True)
                fig.canvas.draw_idle()
            else:
                if vis:
                    annot.set_visible(False)
                    fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", hover)

    plt.show()

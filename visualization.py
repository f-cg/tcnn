import os
import random
import sys
from os.path import join

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from networkx.drawing.nx_agraph import graphviz_layout, write_dot
from numpy import array
'''
0.airplane
1.automobile
2.bird
3.cat
4.deer
5.dog
6.frog
7.horse
8.ship
9.truck
'''
labels = ['airplane', 'automobile', 'bird', 'cat', 'deer',
          'dog', 'frog', 'horse', 'ship', 'truck']


def gen_colors(n):
    ret = []
    r = int(random.random() * 256)
    g = int(random.random() * 256)
    b = int(random.random() * 256)
    step = 256 / n
    for i in range(n):
        r += step
        g += step
        b += step
        r = int(r) % 256
        g = int(g) % 256
        b = int(b) % 256
        ret.append((r/256, g/256, b/256))
    return ret


testtrain = ['test', 'train']
methods = ['cutout', 'targeted']
linestyles = ['-', ':', '-.', '--', '-']
colors = ['b', 'g', 'r', 'c', 'purple', 'y',
          'orange', 'hotpink', 'sienna', 'dimgray']
total_color = 'black'
print(colors)
labelstr = '{} {}'


def get_figname(train, method='', other=''):
    method = ''
    if train is None:
        return '{}{}'.format(method, other)
    return '{}{}{}'.format(testtrain[train], method, other)


def draw(metrics, pretty, folder='', hard=''):
    """
    metrics is a dict: {method: metric}.
    metric:{True:{'class_accs':[array(10)...],'total_acc':[int]*total}, False:}
    """
    global total_color, colors, linestyles
    methods = metrics.keys()
    num = len(methods)
    for idx, (method, metric, style) in enumerate(zip(methods, metrics.values(), linestyles[:num])):
        print(idx)
        for train in [True, False]:
            mode = testtrain[train]
            figname = get_figname(train, method, folder)
            plt.figure(figname)
            plt.tight_layout()
            d = metric[train]
            class_accs = d[hard+'class_accs']
            class_accs = array(class_accs)
            for classid in range(10):
                if pretty == 'many':
                    plt.subplot(3, 4, classid+1)
                acc = class_accs[:, classid]
                if pretty == 'many' or idx == 0:
                    label = labelstr.format(method, classid)
                else:
                    label = None
                if pretty == 'many':
                    classid = idx
                    style = None
                plt.plot(acc, label=label, linestyle=style,
                         color=colors[classid])
                plt.subplots_adjust(0, 0, 1.0, 1.0, 0, 0)

            if pretty == 'many':
                plt.subplot(3, 4, 12)
                plt.text(0.5, 0.5, figname, horizontalalignment='center',
                         verticalalignment='center')
            if pretty == 'many':
                plt.subplot(3, 4, 11)
                total_color = colors[idx]
                style = None
            plt.tight_layout()
            total_acc = d[hard+'total_acc']
            label = labelstr.format(method, hard+'tot')
            plt.plot(total_acc, label=label,
                     linestyle=style, color=total_color)

    for train in [True, False]:
        for method in methods:
            figname = get_figname(train, method, folder)
            plt.figure(figname)
            plt.tight_layout(pad=0.1)
            plt.ylim(-10, 100)
            plt.legend(prop={'size': 6})


def get_metric_from_file(f):
    code = open(f, 'rt').read()
    try:
        metric = eval(code)
    except SyntaxError:
        print('Syntex error when eval the file ', f)
        exit(-1)
    return metric


def get_metrics_from_folder(folder, including=[]):
    """
    return a dict of result on different methods
    """
    d = {}
    methods = os.listdir(folder)
    for m in methods:
        if len(including) > 0 and m not in including:
            continue
        f = os.path.join(folder, m)
        d[m] = get_metric_from_file(f)
    return d


def draw_leaves(leaves):
    """
    metrics is a dict: {method: metric}.
    metric:{True:{'class_accs':[array(10)...],'total_acc':[int]*total}, False:}
    """
    global total_color, colors, linestyles
    figname = 'leaves'
    plt.figure(figname)
    plt.tight_layout()
    leaves = array(leaves).squeeze()
    print(leaves.shape)
    freq = np.zeros(10)
    leaves_num = leaves.shape[1]  # 16 or 32
    for leafid in range(leaves_num):
        freq[leaves[-1, leafid].argmax()] += 1
        plt.subplot(leaves_num//4, 4, leafid + 1)
        for classid in range(10):
            acc = leaves[:, leafid, classid]
            label = str(classid)
            plt.plot(acc, label=label, color=colors[classid])
            plt.subplots_adjust(0, 0, 1.0, 1.0, 0, 0)

        plt.tight_layout(pad=0.1)
        plt.ylim(0, 1)
        plt.legend(prop={'size': 6})

    plt.figure('leaves_freq')
    plt.bar(range(10), freq)


def draw_leaves_fr(leaves):
    """
    metrics is a dict: {method: metric}.
    metric:{True:{'class_accs':[array(10)...],'total_acc':[int]*total}, False:}
    """
    global total_color, colors, linestyles
    figname = 'leaves_fr'
    plt.figure(figname)
    plt.tight_layout()
    leaves = array(leaves)
    print(leaves.shape)
    # freq = np.zeros(10)
    leaves_num = leaves.shape[1]  # 16 or 32
    for leafid in range(leaves_num):
        # freq[leaves[-1, leafid].argmax()] += 1
        plt.subplot(leaves_num//4, 4, leafid + 1)
        for classid in range(10):
            acc = leaves[:, leafid, classid]
            label = str(classid)
            plt.plot(acc, label=label, color=colors[classid])
            plt.subplots_adjust(0, 0, 1.0, 1.0, 0, 0)

        plt.tight_layout(pad=0.1)
        plt.legend(prop={'size': 6})
    plt.figure('tree')
    G = nx.DiGraph()
    last = leaves[-1]
    tree = [0]*(leaves_num*2-1)
    for leafid, leaf in enumerate(last):
        node = []
        for classid, fr in enumerate(leaf):
            if fr > 100:
                node.append(classid)
        node.sort()
        tree[leaves_num-1+leafid] = node
    for nodeid in range(len(tree)-1, 0, -2):
        r = nodeid
        l = nodeid-1
        f = l//2
        tree[f] = tree[l]+tree[r]
        tree[f] = list(set(tree[f]))
        tree[f].sort()
        ln = '{}:{}'.format(l, tree[l])
        rn = '{}:{}'.format(r, tree[r])
        fn = '{}:{}'.format(f, tree[f])
        G.add_node(rn)
        G.add_node(ln)
        G.add_node(fn)
        G.add_edge(fn, rn)
        G.add_edge(fn, ln)
    # write dot file to use with graphviz
    # run "dot -Tpng test.dot >test.png"
    plt.tight_layout()
    write_dot(G, 'tree.dot')
    pos = graphviz_layout(G, prog='dot')
#    nx.draw_networkx_nodes(G, pos, alpha=0.9)
#    nx.draw_networkx_edges(G, pos)
#    nx.draw_networkx_labels(G, pos)
    nx.draw_networkx(G, pos, node_color='none')
#    nx.draw(G, pos, with_labels=True, arrows=True)


if __name__ == '__main__':
    usage = """
    python3 ./visualization.py [folder...] [files...] [--pretty=one|many] [--methods=targeted,cutout]
    python3 ./visualization.py --help
    Put result under the ./result folder
    --pretty=    in one or many subplots
    e.g.
        python3 ./visualization.py
        python3 ./visualization.py normal --pretty=many
        python3 ./visualization.py aug16 --pretty=many
        python3 ./visualization.py aug04 normal aug08/cutout
    """
    root = './result/'
    args = sys.argv[1:]
    pretty = 'one'
    folders = []
    including = []
    additions = []
    for arg in args:
        if arg.startswith('--pretty='):
            pretty = arg[len('--pretty='):]
            assert(pretty == 'one' or pretty == 'many')
        elif arg == '--help' or arg == '-h':
            print(usage)
            exit(0)
        elif arg.startswith('--methods='):
            including = arg[len('--methods='):].split(',')
        else:
            if os.path.isdir(join(root, arg)):
                folders.append(arg)
            else:
                additions.append(arg)
    if len(folders) == 0:
        folders = ['']
    for folder in folders:
        if folder != '':
            fname = join(root, folder)
            if os.path.exists(fname):
                metrics = get_metrics_from_folder(fname, including)
            else:
                metrics = get_metrics_from_folder(folder, including)
        else:
            metrics = {}
        for f in additions:
            fpath = join(root, f)
            if os.path.exists(fpath):
                print(fpath)
                metrics[f] = get_metric_from_file(fpath)
            else:
                print(f)
                metrics[f] = get_metric_from_file(f)

        print(metrics.keys())
        # print(metrics.args())
        # draw(metrics, pretty, folder)
        #draw(metrics, pretty, folder, hard='hard_')
        first_metric = list(metrics.values())[0]
        if 'leaves' in first_metric.keys():
            #draw_leaves(first_metric['leaves'])
            draw_leaves_fr(first_metric['fr_leaves'])
    plt.show()

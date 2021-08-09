import utils
import numpy as np
from utils import Genotype
import os
import torch
import shutil
from graphviz import Digraph


PRIV_PRIMITIVES = [
    'zero',
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect',

    'conv_3x3_relu',
    'conv_3x3_elu',
    'conv_3x3_tanh',
    'conv_3x3_linear',
    'conv_3x3_htanh',
    'conv_3x3_sigmoid',

]
arch = [(5, 0, 1), (1, 0, 2), (5, 1, 2), (6, 0, 3), (1, 1, 3),
        (0, 2, 3), (5, 0, 4), (0, 1, 4), (0, 2, 4), (3, 3, 4),
        (5, 0, 5), (5, 1, 5), (6, 2, 5), (1, 3, 5), (6, 4, 5)]
steps = 5

def arch_to_genotype(arch_normal, arch_reduce, n_nodes,):
    primitives = PRIV_PRIMITIVES

    gene_normal = [(primitives[op], f, t) for op, f, t in arch_normal]
    gene_reduce = [(primitives[op], f, t) for op, f, t in arch_reduce]
    concat = range(2, 2 + n_nodes)
    genotype = Genotype(normal=gene_normal, normal_concat=concat,
                        reduce=gene_reduce, reduce_concat=concat)
    return genotype

def draw_genotype(genotype, n_nodes, filename):

    g = Digraph(
        format='pdf',
        edge_attr=dict(fontsize='20', fontname="times"),
        node_attr=dict(style='filled', shape='rect', align='center', fontsize='20', height='0.5',
                       width='0.5', penwidth='2', fontname="times"), engine='dot')
    g.body.extend(['rankdir=LR'])

    g.node("input", fillcolor='darkseagreen2')

    steps = n_nodes
    for i in range(steps+1):
        g.node(str(i), fillcolor='lightblue')

    for op, source, target in genotype:
        u = str(source)
        v = str(target)
        g.edge(u, v, label=op, fillcolor="gray")

    g.edge('input', '0', label='conv_1x1', fillcolor="gray")

    for i in range(steps):
        g.edge(str(i+1), "output", fillcolor="darkseagreen2")

    g.render(filename, view=False)


genotype = arch_to_genotype(arch, arch, steps)
draw_genotype(genotype.normal, steps, os.path.join('result_arch', "normals" ))
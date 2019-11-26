#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
table_tools.py: tools for printing tables to be readily imported in latex.
"""

import numpy as np
import pandas as pd

METHOD_DICT = {
    'gt': 'GT',
    'srls raw': 'SRLS',
    'srls': 'SRLS fitted',
    'rls raw': 'RLS',
    'rls': 'RLS fitted',
    'lm-ellipse': 'LM ellipse/line',
    'lm-line': 'LM ellipse/line',
    'lm-ours-weighted': 'LM ours weighted',
    'ours': 'ours',
    'ours-weighted': 'ours weighted'
}


def pretty_print_table(print_table, methods=None, value='mse'):
    print_table.rename(columns={'n_measurements': 'N', 'n_complexity': 'K'}, inplace=True)
    pt = pd.pivot_table(print_table, values=value, index='method', columns=['N', 'K'], aggfunc=['mean', 'std'])
    if methods is not None:
        pt = pt.reindex(methods)
    #styler = pt.style.apply(highlight_min, axis=0)
    styler = pt.style.apply(highlight_both, axis=0)
    pd.set_option('precision', 2)
    pd.set_option('max_columns', 100)
    return styler, pt


def highlight_min(data, exclude=[0], color='red', index=0):
    """
    :param exclude: rows indices to exclude for calculating min.
    :param index: set to 0 for smallest, 1 for second smallest, etc.
    """
    attr = 'background-color: {}'.format(color)

    if data.ndim == 1:  # Series from .apply(axis=0) or axis=1
        indices = [i for i in range(len(data)) if i not in exclude]
        is_min = data == sorted(data[indices])[index]
        if sum(is_min) > 1:
            return [''] * len(data)
        return [attr if v else '' for v in is_min]


def highlight_both(data, exclude=[0]):
    attr1 = highlight_min(data, exclude=exclude, color='red', index=0)
    attr2 = highlight_min(data, exclude=exclude, color='orange', index=1)
    return [a1 + a2 for a1, a2 in zip(attr1, attr2)]


def latex_print(pt, methods, fname='', index_names=True, **kwargs):
    method_names = [METHOD_DICT.get(m, 'unknown') for m in methods]
    pt.index = method_names

    min_vals = np.sort(pt['mean'].values[1:, :], axis=0)[0, :].round(4)
    second_vals = np.sort(pt['mean'].values[1:, :], axis=0)[1, :].round(4)
    print(min_vals.shape)
    print(min_vals)
    print(second_vals.shape)

    if index_names:
        column_format = 'l|'
    else:
        column_format = '|'
    N_levels = len(pt.columns.levels[1])
    K_levels = len(pt.columns.levels[2])
    for _ in range(N_levels):
        cols_K = ''.join(['c'] * K_levels)
        column_format += cols_K + '|'

    latex = pt['mean'].to_latex(column_format=column_format,
                                multicolumn_format='c|',
                                index_names=index_names,
                                **kwargs,
                                bold_rows=True)
    for min_val in min_vals.round(2):
        string = " \\cellcolor{{\\firstcolor}}{}".format(min_val)
        latex = latex.replace(' ' + str(min_val), string, 20)

    for min_val in second_vals.round(2):
        string = " \\cellcolor{{\\secondcolor}}{}".format(min_val)
        latex = latex.replace(' ' + str(min_val), string, 20)

    latex = latex.replace('K &', '\\multicolumn{1}{r|}{K} &')
    latex = latex.replace('N &', '\\multicolumn{1}{r|}{N} &')
    latex = latex.replace('SRLS  ', '\\midrule SRLS ')

    print(latex)
    if fname != '':
        with open(fname, 'w+') as f:
            f.write(latex)
        print('wrote as', fname)

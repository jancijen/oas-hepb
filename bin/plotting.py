"""
Utility functions for plotting.

Author: Ing. David Prihoda
"""

import pandas as pd
import numpy as np


def barplot(series, ax=None, limit=None, limit_agg_func='sum', remaining_label='(remaining)', title=None, neglog=False, fmt='{:,}', na_label='N/A', xlogtickstep=2, padding=1.3, **kwargs):
    series = series[::-1]
    if 'color' in kwargs and isinstance(kwargs['color'], list):
        kwargs['color'] = kwargs['color'][::-1]
    if limit:
        series = series.sort_values()
        if len(series) > limit:
            remaining_sum = pd.Series([series[:-limit].agg(limit_agg_func)], [remaining_label])
            top = series[-limit:]
            series = pd.concat([remaining_sum, top])
            series.name = '{} (Top {})'.format(top.name, limit)
    
    title = series.name if title is None else title
    if neglog:
        neglog_series = -np.log10(series)
        ax = neglog_series.plot.barh(ax=ax, title=title, **kwargs)
        xmax = int(np.ceil(neglog_series.max() * padding))
        ax.set_xlim([0, xmax])
        ax.set_xticks(list(range(0, xmax, xlogtickstep)))
        ax.set_xticklabels(['$\mathregular{10^{'+str(-x)+'}}$' for x in ax.get_xticks()]);
        for i, v in enumerate(series.values):
            text = fmt(v) if callable(fmt) else fmt.format(v)
            ax.text(-np.log10(v), i, ' {} '.format(text), va='center', ha='left')

    else:
        ax = series.plot.barh(ax=ax, title=title, **kwargs)
        cmin = series.min()
        cmax = series.max()
        ax.set_xlim(0 if cmin > 0 else cmin * padding, 0 if cmax < 0 else cmax * padding)
        for i, v in enumerate(series.values):
            text = fmt(v) if callable(fmt) else fmt.format(v)
            if pd.isnull(v): 
                if na_label:
                    text = na_label
                    v = 0
            ax.text(v, i, ' {} '.format(text), va='center', ha='right' if v < 0 else 'left')
    return ax
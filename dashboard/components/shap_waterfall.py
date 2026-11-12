"""SHAP waterfall + class-probability charts."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from dashboard import theme as T
from ml.feature_engine import FEATURE_NAMES


def shap_waterfall_figure(shap: dict, max_display: int = 10, height: int = 430) -> go.Figure:
    """Waterfall from the base value (log-odds) to the explained class' output.
    Bright bars push toward the verdict, dim bars push away from it."""
    full = np.asarray(shap.get("full_shap_values"), float) if shap.get("full_shap_values") is not None else None
    if full is None or full.size != len(FEATURE_NAMES):
        names = [t[0] for t in shap["top_features"]]
        vals = [t[1] for t in shap["top_features"]]
        actual = [t[2] for t in shap["top_features"]]
        rest = 0.0
    else:
        order = np.argsort(-np.abs(full))
        top = order[:max_display]
        names = [FEATURE_NAMES[i] for i in top]
        vals = full[top].tolist()
        lookup = {n: a for n, _, a in shap["top_features"]}
        actual = [lookup.get(n) for n in names]
        rest = float(full[order[max_display:]].sum())
    labels = [f"{n} = {a:.4g}" if a is not None else n for n, a in zip(names, actual)][::-1]
    vals = vals[::-1]
    if rest:
        labels = [f"{len(FEATURE_NAMES) - len(names)} other measurements"] + labels
        vals = [rest] + vals
    base = float(shap.get("base_value", 0.0))
    fig = go.Figure(go.Waterfall(
        orientation="h", measure=["absolute"] + ["relative"] * len(vals) + ["total"],
        y=["starting point"] + labels + ["final score"],
        x=[base] + vals + [0],
        increasing=dict(marker=dict(color=T.TEXT)),
        decreasing=dict(marker=dict(color=T.DIM)),
        totals=dict(marker=dict(color=T.MUTED)),
        connector=dict(line=dict(color=T.HAIRLINE, width=1)),
    ))
    fig.update_layout(title=f"Why the model chose {shap.get('explained_class')}",
                      height=height, xaxis_title="Contribution to the score (log-odds)")
    return fig


def probability_figure(probs: dict, height: int = 340) -> go.Figure:
    """Normal in the proceed colour, attacks in the abort colour; only the
    winning class is drawn at full strength."""
    items = sorted(probs.items(), key=lambda kv: kv[1])
    top = items[-1][0] if items else None
    colors = [T.PROCEED if k == "Normal" else T.ABORT for k, _ in items]
    opac = [1.0 if k == top else 0.38 for k, _ in items]
    fig = go.Figure(go.Bar(x=[v for _, v in items], y=[k for k, _ in items], orientation="h",
                           marker=dict(color=colors, opacity=opac, line=dict(width=0)),
                           text=[f"{v * 100:.1f}%" for _, v in items], textposition="outside",
                           textfont=dict(color=T.TEXT)))
    fig.update_layout(title="How likely each situation is", height=height,
                      xaxis=dict(range=[0, 1.32], tickformat=".0%", showgrid=False,
                                 tickvals=[0, 0.5, 1.0]),
                      yaxis=dict(showgrid=False))
    return fig

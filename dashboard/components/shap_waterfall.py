"""SHAP waterfall + class-probability charts."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from ml.feature_engine import FEATURE_NAMES


def shap_waterfall_figure(shap: dict, max_display: int = 10, height: int = 420) -> go.Figure:
    """Waterfall from the base value (log-odds) to the explained class' output."""
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
        labels = [f"{len(FEATURE_NAMES) - len(names)} other features"] + labels
        vals = [rest] + vals
    base = float(shap.get("base_value", 0.0))
    fig = go.Figure(go.Waterfall(
        orientation="h", measure=["absolute"] + ["relative"] * len(vals) + ["total"],
        y=["base value"] + labels + ["model output"],
        x=[base] + vals + [0],
        increasing=dict(marker=dict(color="#d1495b")),
        decreasing=dict(marker=dict(color="#2b7bba")),
        totals=dict(marker=dict(color="#555")),
        connector=dict(line=dict(color="#aaa", width=1)),
    ))
    fig.update_layout(title=f"SHAP waterfall - why '{shap.get('explained_class')}'",
                      height=height, margin=dict(l=10, r=10, t=40, b=10),
                      xaxis_title="contribution to class log-odds")
    return fig


def probability_figure(probs: dict, height: int = 320) -> go.Figure:
    items = sorted(probs.items(), key=lambda kv: kv[1])
    colors = ["#1b7f3b" if k == "Normal" else "#d1495b" for k, _ in items]
    fig = go.Figure(go.Bar(x=[v for _, v in items], y=[k for k, _ in items], orientation="h",
                           marker_color=colors, text=[f"{v * 100:.1f}%" for _, v in items],
                           textposition="outside"))
    fig.update_layout(title="Classifier probabilities", height=height,
                      margin=dict(l=10, r=10, t=40, b=10),
                      xaxis=dict(range=[0, 1.12], tickformat=".0%"))
    return fig

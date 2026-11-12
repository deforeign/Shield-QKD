"""Real-time temporal QBER chart."""
from __future__ import annotations

import time

import numpy as np
import plotly.graph_objects as go

from config import settings as S


def qber_figure(qber, rect=None, diag=None, upto: int | None = None, title: str = "Temporal QBER",
                height: int = 360) -> go.Figure:
    q = np.asarray(qber, float)
    n = len(q) if upto is None else upto
    x = np.arange(1, len(q) + 1)
    fig = go.Figure()
    if rect is not None:
        fig.add_trace(go.Scatter(x=x[:n], y=np.asarray(rect, float)[:n], name="rectilinear basis",
                                 mode="lines", line=dict(width=1, dash="dot", color="#59a14f")))
    if diag is not None:
        fig.add_trace(go.Scatter(x=x[:n], y=np.asarray(diag, float)[:n], name="diagonal basis",
                                 mode="lines", line=dict(width=1, dash="dot", color="#f28e2b")))
    fig.add_trace(go.Scatter(x=x[:n], y=q[:n], name="window QBER (W=50)", mode="lines+markers",
                             line=dict(width=2.5, color="#2b7bba"), marker=dict(size=5)))
    if n:
        fig.add_hline(y=float(q[:n].mean()), line_dash="dash", line_color="#2b7bba",
                      annotation_text=f"mean {q[:n].mean() * 100:.2f}%", annotation_position="top left")
    fig.add_hline(y=S.FIXED_QBER_THRESHOLD, line_color="#d1495b", line_dash="dash",
                  annotation_text="fixed 11% threshold", annotation_position="top right")
    ymax = max(0.15, float(np.nanmax(q)) * 1.15 if len(q) else 0.15)
    fig.update_layout(title=title, height=height, margin=dict(l=10, r=10, t=40, b=10),
                      xaxis=dict(title="window index", range=[0.5, len(q) + 0.5]),
                      yaxis=dict(title="QBER", range=[0, ymax], tickformat=".0%"),
                      legend=dict(orientation="h", y=-0.25))
    return fig


def render_live(st, placeholder, qber, rect=None, diag=None, animate: bool = True,
                frames: int = 25, delay: float = 0.04):
    """Progressively draw the series to mimic windows arriving in real time."""
    T = len(qber)
    if animate and T > 2:
        for k in np.unique(np.linspace(2, T, min(frames, T)).astype(int)):
            placeholder.plotly_chart(qber_figure(qber, rect, diag, upto=int(k)),
                                     use_container_width=True, key=f"live_{k}_{time.time()}")
            time.sleep(delay)
    placeholder.plotly_chart(qber_figure(qber, rect, diag), use_container_width=True,
                             key=f"live_final_{time.time()}")

"""Real-time temporal QBER chart."""
from __future__ import annotations

import time

import numpy as np
import plotly.graph_objects as go

from config import settings as S
from dashboard import theme as T


def qber_figure(qber, rect=None, diag=None, upto: int | None = None,
                title: str = "Error rate over the session", height: int = 380) -> go.Figure:
    q = np.asarray(qber, float)
    n = len(q) if upto is None else upto
    x = np.arange(1, len(q) + 1)
    fig = go.Figure()
    if rect is not None:
        fig.add_trace(go.Scatter(x=x[:n], y=np.asarray(rect, float)[:n], name="Rectilinear basis",
                                 mode="lines", line=dict(width=1, dash="dot", color=T.RECT),
                                 opacity=0.75))
    if diag is not None:
        fig.add_trace(go.Scatter(x=x[:n], y=np.asarray(diag, float)[:n], name="Diagonal basis",
                                 mode="lines", line=dict(width=1, dash="dot", color=T.DIAG),
                                 opacity=0.75))
    fig.add_trace(go.Scatter(x=x[:n], y=q[:n], name="All bits (50-bit windows)", mode="lines+markers",
                             line=dict(width=2.4, color=T.TEXT), marker=dict(size=4, color=T.TEXT)))
    if n:
        fig.add_hline(y=float(q[:n].mean()), line_dash="dash", line_color=T.MUTED, line_width=1,
                      annotation_text=f"average {q[:n].mean() * 100:.2f}%",
                      annotation_position="top left", annotation_font_color=T.MUTED)
    fig.add_hline(y=S.FIXED_QBER_THRESHOLD, line_color=T.ABORT, line_dash="dash", line_width=1,
                  annotation_text="old 11% rule", annotation_position="top right",
                  annotation_font_color=T.ABORT)
    peaks = [float(np.nanmax(s)) for s in (q, rect, diag)
             if s is not None and len(s) and not np.all(np.isnan(np.asarray(s, float)))]
    ymax = max([0.15] + [p * 1.12 for p in peaks])
    fig.update_layout(title=title, height=height,
                      xaxis=dict(title="Window (time →)", range=[0.5, len(q) + 0.5]),
                      yaxis=dict(title="Bits that disagree", range=[0, ymax], tickformat=".0%"),
                      legend=dict(orientation="h", y=-0.22, x=0))
    return fig


def render_live(st, placeholder, qber, rect=None, diag=None, animate: bool = True,
                frames: int = 25, delay: float = 0.04):
    """Progressively draw the series to mimic windows arriving in real time."""
    T_ = len(qber)
    if animate and T_ > 2:
        for k in np.unique(np.linspace(2, T_, min(frames, T_)).astype(int)):
            placeholder.plotly_chart(qber_figure(qber, rect, diag, upto=int(k)),
                                     use_container_width=True, key=f"live_{k}_{time.time()}")
            time.sleep(delay)
    placeholder.plotly_chart(qber_figure(qber, rect, diag), use_container_width=True,
                             key=f"live_final_{time.time()}")

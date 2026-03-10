"""Plotly figure builders for the HDM-TG error dashboard."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import welch


# Colour palette for comparison overlays
_COMPARE_COLORS = ["#27AE60", "#8E44AD", "#F39C12", "#1ABC9C", "#E74C3C",
                   "#3498DB", "#D35400", "#2C3E50", "#16A085", "#C0392B"]


def make_time_plot(df: pd.DataFrame, station_name: str = "",
                  compare_data: list | None = None) -> go.Figure:
    """Time-domain: TG observation (blue) vs DKSS model (red), two panels."""
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
        row_heights=[0.6, 0.4],
        subplot_titles=["Sea Level", "Error  ε(t) = DKSS − TG obs"],
    )

    if df.empty:
        fig.add_annotation(text="No data for selection", showarrow=False,
                           font=dict(size=16, color="#888"), row=1, col=1)
        fig.update_layout(height=480, template="plotly_white")
        return fig

    # Panel 1: observations + model
    fig.add_trace(go.Scattergl(
        x=df["valid_time"], y=df["tg_obs_m"],
        mode="lines", line=dict(width=0.8, color="#2980B9"),
        name="TG obs",
    ), row=1, col=1)

    fig.add_trace(go.Scattergl(
        x=df["valid_time"], y=df["dkss_p82_m"],
        mode="lines", line=dict(width=0.8, color="#E74C3C"),
        name="DKSS model",
    ), row=1, col=1)

    # Panel 2: error with fill to zero
    fig.add_trace(go.Scattergl(
        x=df["valid_time"], y=df["error_m"],
        mode="lines", line=dict(width=0.8, color="#8E44AD"),
        fill="tozeroy", fillcolor="rgba(142,68,173,0.15)",
        name="ε(t)",
    ), row=2, col=1)

    mean_err = df["error_m"].mean()
    fig.add_hline(y=mean_err, line_dash="dot", line_color="#E74C3C",
                  line_width=1, row=2, col=1,
                  annotation_text=f"bias = {mean_err:+.4f} m",
                  annotation_position="top left")
    fig.add_hline(y=0, line_dash="dash", line_color="grey",
                  line_width=0.5, row=2, col=1)

    title = "TG obs vs DKSS"
    if station_name:
        title += f"  —  {station_name}"

    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        template="plotly_white",
        margin=dict(l=60, r=20, t=60, b=40),
        height=480,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_yaxes(title_text="Sea level [m]", row=1, col=1)
    fig.update_yaxes(title_text="Error [m]", row=2, col=1)
    fig.update_xaxes(title_text="Time", row=2, col=1)

    # ── Overlay comparison stations ──────────────────────────────
    if compare_data:
        for i, (cdf, cname) in enumerate(compare_data):
            if cdf.empty:
                continue
            clr = _COMPARE_COLORS[i % len(_COMPARE_COLORS)]
            fig.add_trace(go.Scattergl(
                x=cdf["valid_time"], y=cdf["error_m"],
                mode="lines", line=dict(width=0.8, color=clr),
                name=f"ε  {cname}",
                opacity=0.7,
            ), row=2, col=1)

    return fig


def make_psd_plot(df: pd.DataFrame, station_name: str = "",
                 compare_data: list | None = None) -> go.Figure:
    """Spectral amplitude of error vs frequency, with hover info."""
    fig = go.Figure()

    signal = df["error_m"].dropna().values if not df.empty else np.array([])
    if len(signal) < 32:
        fig.add_annotation(text="Insufficient data for spectrum", showarrow=False,
                           font=dict(size=16, color="#888"))
        fig.update_layout(height=340, template="plotly_white")
        return fig

    # Normalise error to z-score (ε/σ) so spectrum is dimensionless
    sigma = np.std(signal)
    if sigma == 0:
        sigma = 1.0
    signal = signal / sigma

    # Detect sampling interval from data
    dt_hours = df["valid_time"].diff().median().total_seconds() / 3600.0
    dt_hours = max(dt_hours, 0.5)  # safety floor
    fs = 1.0 / dt_hours  # samples per hour
    nperseg = min(512, len(signal) // 2)
    freqs, psd = welch(signal, fs=fs, nperseg=nperseg)

    # Skip DC component (freq=0)
    mask = freqs > 0
    freqs = freqs[mask]
    psd = psd[mask]

    # Convert: period in days, amplitude in cm
    period_days = 1.0 / (freqs * 24.0)          # hours→days
    # Amplitude spectral density: sqrt(PSD * df) ~ sqrt(PSD) for shape
    # More useful: amplitude = sqrt(2 * PSD * delta_f) per bin → cm
    df_freq = freqs[1] - freqs[0] if len(freqs) > 1 else freqs[0]
    amplitude = np.sqrt(2.0 * psd * df_freq)  # dimensionless (normalised)

    # Sort by period ascending
    order = np.argsort(period_days)
    period_days = period_days[order]
    amplitude = amplitude[order]
    freqs_sorted = freqs[order]

    # Hover text
    hover_text = [
        f"T = {p:.2f} d  ({p*24:.1f} h)<br>"
        f"f = {f:.5f} cyc/h<br>"
        f"Amplitude = {a:.4f}"
        for p, f, a in zip(period_days, freqs_sorted, amplitude)
    ]

    fig.add_trace(go.Scattergl(
        x=freqs_sorted, y=amplitude,
        mode="lines", line=dict(width=1.5, color="#E67E22"),
        name=station_name or "Primary",
        hovertext=hover_text,
        hoverinfo="text",
    ))

    # Reference lines at key periods
    # Reference lines at key frequencies (cyc/h)
    ref_lines = [
        ("6h", 1/6), ("12h", 1/12),
        ("1d", 1/24), ("2d", 1/48), ("7d", 1/168), ("30d", 1/720),
    ]
    f_min, f_max = freqs_sorted[0], freqs_sorted[-1]
    for label, f_ref in ref_lines:
        if f_min <= f_ref <= f_max:
            fig.add_vline(x=f_ref, line_dash="dot", line_color="#ccc",
                          line_width=0.7)
            fig.add_annotation(
                x=f_ref, y=1, yref="paper", yshift=-15,
                text=f"<sub>{label}</sub>",
                showarrow=False, font=dict(size=8, color="#999"),
            )

    title = "Normalised Error Amplitude Spectrum"
    if station_name:
        title += f"  —  {station_name}"

    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        xaxis_title="Frequency [cyc/h]",
        yaxis_title="Normalised Amplitude [ε/σ]",
        xaxis=dict(type="linear"),
        yaxis=dict(type="linear"),
        template="plotly_white",
        margin=dict(l=60, r=20, t=60, b=40),
        height=340,
        showlegend=bool(compare_data),
        hovermode="x unified",
    )

    # ── Overlay comparison stations ──────────────────────────────
    if compare_data:
        for i, (cdf, cname) in enumerate(compare_data):
            csig = cdf["error_m"].dropna().values if not cdf.empty else np.array([])
            if len(csig) < 32:
                continue
            c_sigma = np.std(csig)
            if c_sigma == 0:
                c_sigma = 1.0
            csig = csig / c_sigma

            c_dt = cdf["valid_time"].diff().median().total_seconds() / 3600.0
            c_dt = max(c_dt, 0.5)
            c_fs = 1.0 / c_dt
            c_nperseg = min(512, len(csig) // 2)
            c_freqs, c_psd = welch(csig, fs=c_fs, nperseg=c_nperseg)
            c_mask = c_freqs > 0
            c_freqs = c_freqs[c_mask]
            c_psd = c_psd[c_mask]
            c_df = c_freqs[1] - c_freqs[0] if len(c_freqs) > 1 else c_freqs[0]
            c_amp = np.sqrt(2.0 * c_psd * c_df)
            c_order = np.argsort(c_freqs)
            c_freqs = c_freqs[c_order]
            c_amp = c_amp[c_order]

            clr = _COMPARE_COLORS[i % len(_COMPARE_COLORS)]
            fig.add_trace(go.Scattergl(
                x=c_freqs, y=c_amp,
                mode="lines", line=dict(width=1.2, color=clr),
                name=cname,
                opacity=0.7,
            ))
        fig.update_layout(showlegend=True)

    return fig


def compute_stats(df: pd.DataFrame) -> dict:
    """Compute error & data quality statistics from a station slice."""
    if df.empty:
        return {"total": 0, "valid": 0, "pct_valid": 0.0,
                "bias": np.nan, "rmse": np.nan, "mae": np.nan, "std": np.nan}

    err = df["error_m"].dropna()
    total = len(df)
    valid = len(err)
    bias = float(err.mean()) if valid else np.nan
    rmse = float(np.sqrt((err ** 2).mean())) if valid else np.nan
    mae = float(err.abs().mean()) if valid else np.nan
    std = float(err.std()) if valid else np.nan

    return {
        "total": total,
        "valid": valid,
        "pct_valid": round(100 * valid / total, 1) if total else 0.0,
        "bias": round(bias, 5),
        "rmse": round(rmse, 5),
        "mae": round(mae, 5),
        "std": round(std, 5),
    }

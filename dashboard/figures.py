"""Plotly figure builders for the HDM-TG error dashboard."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import welch
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import acf
from statsmodels.stats.stattools import durbin_watson


# ── Colour helpers ───────────────────────────────────────────────────────
_COMPARE_COLORS = [
    "#27AE60", "#8E44AD", "#F39C12", "#1ABC9C", "#E74C3C",
    "#3498DB", "#D35400", "#2C3E50", "#16A085", "#C0392B",
]
_IRF_COLORS = ["#E74C3C", "#3498DB", "#2ECC71", "#F39C12"]


def _hex_to_rgb(hex_color: str) -> str:
    """'#RRGGBB' → 'R,G,B' for rgba() strings."""
    h = hex_color.lstrip("#")
    return ",".join(str(int(h[i : i + 2], 16)) for i in (0, 2, 4))


# ═════════════════════════════════════════════════════════════════════════
# OVERVIEW tab figures (unchanged logic, kept as-is)
# ═════════════════════════════════════════════════════════════════════════

def make_time_plot(
    df: pd.DataFrame,
    station_name: str = "",
    compare_data: list | None = None,
) -> go.Figure:
    """Time-domain: TG observation (blue) vs DKSS model (red), two panels."""
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
        row_heights=[0.6, 0.4],
        subplot_titles=["Sea Level", "Error  ε(t) = FORCOAST − TG obs"],
    )

    if df.empty:
        fig.add_annotation(text="No data for selection", showarrow=False,
                           font=dict(size=16, color="#888"), row=1, col=1)
        fig.update_layout(height=480, template="plotly_white")
        return fig

    fig.add_trace(go.Scattergl(
        x=df["valid_time"], y=df["tg_obs_m"],
        mode="lines", line=dict(width=0.8, color="#2980B9"),
        name="TG obs",
    ), row=1, col=1)

    fig.add_trace(go.Scattergl(
        x=df["valid_time"], y=df["dkss_p82_m"],
        mode="lines", line=dict(width=0.8, color="#E74C3C"),
        name="FORCOAST (HDM)",
    ), row=1, col=1)

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

    title = "TG obs vs FORCOAST (HDM)"
    if station_name:
        title += f"  —  {station_name}"

    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        template="plotly_white",
        margin=dict(l=60, r=20, t=60, b=40),
        height=480,
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1),
    )
    fig.update_yaxes(title_text="Sea level [m]", row=1, col=1)
    fig.update_yaxes(title_text="Error [m]", row=2, col=1)
    fig.update_xaxes(title_text="Time", row=2, col=1)

    if compare_data:
        for i, (cdf, cname) in enumerate(compare_data):
            if cdf.empty:
                continue
            clr = _COMPARE_COLORS[i % len(_COMPARE_COLORS)]
            fig.add_trace(go.Scattergl(
                x=cdf["valid_time"], y=cdf["error_m"],
                mode="lines", line=dict(width=0.8, color=clr),
                name=f"ε  {cname}", opacity=0.7,
            ), row=2, col=1)

    return fig


def make_psd_plot(
    df: pd.DataFrame,
    station_name: str = "",
    compare_data: list | None = None,
) -> go.Figure:
    """Spectral amplitude of error vs period (log-log)."""
    fig = go.Figure()

    signal = df["error_m"].dropna().values if not df.empty else np.array([])
    if len(signal) < 32:
        fig.add_annotation(text="Insufficient data for spectrum",
                           showarrow=False, font=dict(size=16, color="#888"))
        fig.update_layout(height=340, template="plotly_white")
        return fig

    sigma = np.std(signal)
    if sigma == 0:
        sigma = 1.0
    signal = signal / sigma

    dt_hours = df["valid_time"].diff().median().total_seconds() / 3600.0
    dt_hours = max(dt_hours, 0.5)
    fs = 1.0 / dt_hours
    nperseg = min(512, len(signal) // 2)
    freqs, psd = welch(signal, fs=fs, nperseg=nperseg)

    mask = freqs > 0
    freqs, psd = freqs[mask], psd[mask]
    period_h = 1.0 / freqs

    df_freq = freqs[1] - freqs[0] if len(freqs) > 1 else freqs[0]
    amplitude = np.sqrt(2.0 * psd * df_freq)

    order = np.argsort(period_h)
    period_h, amplitude = period_h[order], amplitude[order]
    freqs_sorted = freqs[order]

    hover_text = [
        f"T = {p:.2f} h  ({p/24:.2f} d)<br>"
        f"f = {f:.5f} cyc/h<br>"
        f"Amplitude = {a:.4f} ε/σ"
        for p, f, a in zip(period_h, freqs_sorted, amplitude)
    ]

    fig.add_trace(go.Scattergl(
        x=period_h, y=amplitude,
        mode="lines", line=dict(width=1.5, color="#E67E22"),
        name=station_name or "Primary",
        hovertext=hover_text, hoverinfo="text",
    ))

    ref_lines = [
        ("M4<br>6.2h", 6.21), ("S2<br>12h", 12.0),
        ("M2<br>12.4h", 12.42), ("K1<br>24h", 23.93),
        ("O1<br>25.8h", 25.82), ("2d", 48.0),
        ("7d", 168.0), ("30d", 720.0),
    ]
    p_min, p_max = period_h[0], period_h[-1]
    for label, p_ref in ref_lines:
        if p_min <= p_ref <= p_max:
            fig.add_vline(x=p_ref, line_dash="dot",
                          line_color="#bbb", line_width=0.8)
            fig.add_annotation(
                x=np.log10(p_ref), xref="x", y=1.0, yref="paper",
                yshift=6, text=f"<sub>{label}</sub>",
                showarrow=False, font=dict(size=8, color="#888"),
                align="center",
            )

    title = "Normalised Error Amplitude Spectrum"
    if station_name:
        title += f"  —  {station_name}"

    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        xaxis_title="Period [h]",
        yaxis_title="Normalised Amplitude [ε/σ]",
        xaxis=dict(type="log"), yaxis=dict(type="log"),
        template="plotly_white",
        margin=dict(l=60, r=20, t=60, b=40),
        height=340,
        showlegend=bool(compare_data),
        hovermode="x unified",
    )

    if compare_data:
        for i, (cdf, cname) in enumerate(compare_data):
            csig = cdf["error_m"].dropna().values if not cdf.empty else np.array([])
            if len(csig) < 32:
                continue
            c_sigma = np.std(csig) or 1.0
            csig = csig / c_sigma
            c_dt = max(cdf["valid_time"].diff().median().total_seconds() / 3600.0, 0.5)
            c_fs = 1.0 / c_dt
            c_nperseg = min(512, len(csig) // 2)
            c_freqs, c_psd = welch(csig, fs=c_fs, nperseg=c_nperseg)
            c_mask = c_freqs > 0
            c_freqs, c_psd = c_freqs[c_mask], c_psd[c_mask]
            c_df = c_freqs[1] - c_freqs[0] if len(c_freqs) > 1 else c_freqs[0]
            c_amp = np.sqrt(2.0 * c_psd * c_df)
            c_order = np.argsort(c_freqs)
            c_period_h = 1.0 / c_freqs[c_order]
            c_amp = c_amp[c_order]
            clr = _COMPARE_COLORS[i % len(_COMPARE_COLORS)]
            fig.add_trace(go.Scattergl(
                x=c_period_h, y=c_amp,
                mode="lines", line=dict(width=1.2, color=clr),
                name=cname, opacity=0.7,
            ))
        fig.update_layout(showlegend=True)

    return fig


def compute_stats(df: pd.DataFrame) -> dict:
    """Compute error & data quality statistics from a station slice."""
    if df.empty:
        return {"total": 0, "valid": 0, "pct_valid": 0.0,
                "bias": np.nan, "rmse": np.nan, "mae": np.nan, "std": np.nan}

    err = df["error_m"].dropna()
    total, valid = len(df), len(err)
    bias = float(err.mean()) if valid else np.nan
    rmse = float(np.sqrt((err ** 2).mean())) if valid else np.nan
    mae = float(err.abs().mean()) if valid else np.nan
    std = float(err.std()) if valid else np.nan

    return {
        "total": total, "valid": valid,
        "pct_valid": round(100 * valid / total, 1) if total else 0.0,
        "bias": round(bias, 5), "rmse": round(rmse, 5),
        "mae": round(mae, 5), "std": round(std, 5),
    }


# ═════════════════════════════════════════════════════════════════════════
# REGRESSION tab — compute + figures
# ═════════════════════════════════════════════════════════════════════════

_REG_FEATURES = ["SLP", "t2m", "u10", "v10"]
_REG_LABELS   = ["SLP", "T2m", "u10", "v10"]


def compute_regression(
    df: pd.DataFrame,
    method: str = "ols",
    lag: int = 0,
) -> dict:
    """Regression  ε(t) ~ f(SLP, T2m, u10, v10).

    Parameters
    ----------
    df     : DataFrame with valid_time, SLP, t2m, u10, v10, error_m.
    method : ``"ols"`` (instantaneous) or ``"miso"`` (MISO lag).
    lag    : max lag *L* in hours (only when *method* = ``"miso"``).

    Returns dict with: ok, r2, rmse, bias, dw, coefs, n, method, lag,
                        y, y_pred, residuals, time, beta_matrix.
    """
    empty: dict = {
        "ok": False, "r2": np.nan, "rmse": np.nan, "bias": np.nan,
        "dw": np.nan, "coefs": {}, "n": 0, "method": method, "lag": lag,
        "y": None, "y_pred": None, "residuals": None, "time": None,
        "beta_matrix": None,
    }

    needed = _REG_FEATURES + ["error_m"]
    if any(c not in df.columns for c in needed) or df.empty:
        return empty

    if method == "miso" and lag >= 1:
        return _compute_miso(df, lag, empty)
    return _compute_ols(df, empty)


def _compute_ols(df: pd.DataFrame, empty: dict) -> dict:
    cols = _REG_FEATURES + ["error_m"]
    has_time = "valid_time" in df.columns
    if has_time:
        cols = cols + ["valid_time"]

    sub = df[cols].dropna()
    if len(sub) < 30:
        return empty

    scaler = StandardScaler()
    X = scaler.fit_transform(sub[_REG_FEATURES].values)
    y = sub["error_m"].values

    model = LinearRegression().fit(X, y)
    y_pred = model.predict(X)
    residuals = y - y_pred

    return {
        "ok": True,
        "r2": float(r2_score(y, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y, y_pred))),
        "bias": float(y_pred.mean() - y.mean()),
        "dw": float(durbin_watson(residuals)),
        "coefs": {lbl: float(c) for lbl, c in zip(_REG_LABELS, model.coef_)},
        "n": len(sub), "method": "ols", "lag": 0,
        "y": y, "y_pred": y_pred, "residuals": residuals,
        "time": sub["valid_time"].values if has_time else None,
        "beta_matrix": None,
    }


def _compute_miso(df: pd.DataFrame, lag: int, empty: dict) -> dict:
    L = lag
    sub = df.copy()
    has_time = "valid_time" in sub.columns
    if has_time:
        sub = sub.sort_values("valid_time").reset_index(drop=True)

    lag_dict: dict[str, pd.Series] = {}
    for feat in _REG_FEATURES:
        for k in range(L + 1):
            lag_dict[f"{feat}_lag{k:03d}"] = sub[feat].shift(k)

    df_lag = pd.DataFrame(lag_dict)
    keep = ["error_m"] + (["valid_time"] if has_time else [])
    df_model = pd.concat([sub[keep], df_lag], axis=1).dropna().reset_index(drop=True)

    if len(df_model) < 30:
        return empty

    y = df_model["error_m"].values
    X_raw = df_model[df_lag.columns].values

    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X_raw)

    model = LinearRegression().fit(X_sc, y)
    y_pred = model.predict(X_sc)
    residuals = y - y_pred
    beta_matrix = model.coef_.reshape(len(_REG_FEATURES), L + 1)

    return {
        "ok": True,
        "r2": float(r2_score(y, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y, y_pred))),
        "bias": float(y_pred.mean() - y.mean()),
        "dw": float(durbin_watson(residuals)),
        "coefs": {lbl: float(beta_matrix[i, 0])
                  for i, lbl in enumerate(_REG_LABELS)},
        "n": len(df_model), "method": "miso", "lag": L,
        "y": y, "y_pred": y_pred, "residuals": residuals,
        "time": df_model["valid_time"].values if has_time else None,
        "beta_matrix": beta_matrix,
    }


# ── Regression figures ──────────────────────────────────────────────────

def make_regression_plot(reg: dict, station_name: str = "") -> go.Figure:
    """Dispatch to OLS or MISO figure builder."""
    if not reg.get("ok"):
        fig = go.Figure()
        fig.add_annotation(text="Insufficient data for regression",
                           showarrow=False, font=dict(size=14, color="#888"))
        fig.update_layout(height=300, template="plotly_white")
        return fig

    if reg["method"] == "miso":
        return _make_miso_fig(reg, station_name)
    return _make_ols_fig(reg, station_name)


def _make_ols_fig(reg: dict, station_name: str) -> go.Figure:
    """OLS: row-1 time series, row-2 scatter + β-bar."""
    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{"colspan": 2}, None], [{}, {}]],
        row_heights=[0.55, 0.45],
        vertical_spacing=0.14,
        horizontal_spacing=0.15,
        subplot_titles=[
            "ε(t) observed  vs  predicted", "",
            "Scatter: obs vs pred", "Standardised coefficients β",
        ],
    )

    y, yp, t = reg["y"], reg["y_pred"], reg["time"]

    # Row 1 — time series
    if t is not None:
        fig.add_trace(go.Scattergl(
            x=t, y=y, mode="lines",
            line=dict(width=0.7, color="#2980B9"), name="ε observed",
        ), row=1, col=1)
        fig.add_trace(go.Scattergl(
            x=t, y=yp, mode="lines",
            line=dict(width=1.0, color="#E74C3C"), name="ε predicted (OLS)",
        ), row=1, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="grey",
                  line_width=0.5, row=1, col=1)

    # Row 2 left — scatter
    lim = max(abs(y).max(), abs(yp).max()) * 1.05
    fig.add_trace(go.Scattergl(
        x=y, y=yp, mode="markers",
        marker=dict(size=2, color="#8E44AD", opacity=0.20),
        showlegend=False,
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=[-lim, lim], y=[-lim, lim], mode="lines",
        line=dict(color="grey", dash="dash", width=1), showlegend=False,
    ), row=2, col=1)
    fig.add_annotation(
        text=f"R² = {reg['r2']:.3f}", xref="x3", yref="y3",
        x=lim * 0.6, y=-lim * 0.7, showarrow=False,
        font=dict(size=11, color="#555"),
        bgcolor="rgba(255,255,255,0.8)",
    )

    # Row 2 right — β bar
    labels = list(reg["coefs"].keys())
    coefs = list(reg["coefs"].values())
    colors = ["#C0392B" if v < 0 else "#1A6B9A" for v in coefs]
    fig.add_trace(go.Bar(
        y=labels, x=coefs, orientation="h", marker_color=colors,
        text=[f"{v:+.4f}" for v in coefs], textposition="outside",
        textfont=dict(size=10), showlegend=False,
    ), row=2, col=2)
    fig.add_vline(x=0, line_color="grey", line_width=1, row=2, col=2)

    name_tag = f"<b>{station_name}</b>  ·  " if station_name else ""
    fig.update_layout(
        title=dict(
            text=(
                f"{name_tag}<b>OLS Regression</b>"
                f"<br><span style='font-size:11px;color:#666'>"
                f"R² = {reg['r2']:.3f}  ·  "
                f"RMSE = {reg['rmse'] * 100:.2f} cm  ·  "
                f"DW = {reg['dw']:.2f}  ·  n = {reg['n']:,}</span>"
            ),
            font=dict(size=14),
        ),
        template="plotly_white", height=480,
        margin=dict(l=55, r=25, t=80, b=45),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1),
    )
    fig.update_xaxes(title_text="ε observed [m]", row=2, col=1)
    fig.update_yaxes(title_text="ε predicted [m]", row=2, col=1)
    fig.update_xaxes(title_text="β [m/σ]", row=2, col=2)
    fig.update_yaxes(title_text="Error [m]", row=1, col=1)

    return fig


def _make_miso_fig(reg: dict, station_name: str) -> go.Figure:
    """MISO: row-1 time series, row-2 scatter + residual histogram, row-3 IRFs."""
    L = reg["lag"]
    n_feat = len(_REG_FEATURES)

    fig = make_subplots(
        rows=3, cols=n_feat,
        specs=[
            [{"colspan": n_feat}, *([None] * (n_feat - 1))],
            [{"colspan": 2}, None, {"colspan": 2}, None],
            [{} for _ in range(n_feat)],
        ],
        row_heights=[0.25, 0.25, 0.50],
        vertical_spacing=0.09,
        horizontal_spacing=0.06,
        subplot_titles=[
            "ε(t) observed  vs  predicted", "",
            "Scatter: obs vs pred", "Residuals distribution",
        ] + [f"IRF — {lbl}" for lbl in _REG_LABELS],
    )

    y, yp = reg["y"], reg["y_pred"]
    residuals, t = reg["residuals"], reg["time"]
    beta = reg["beta_matrix"]
    lags_x = np.arange(L + 1)

    # Row 1 — time series
    if t is not None:
        fig.add_trace(go.Scattergl(
            x=t, y=y, mode="lines",
            line=dict(width=0.6, color="#2980B9"), name="ε observed",
        ), row=1, col=1)
        fig.add_trace(go.Scattergl(
            x=t, y=yp, mode="lines",
            line=dict(width=0.9, color="#E74C3C"),
            name=f"ε predicted (MISO L={L}h)",
        ), row=1, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="grey",
                  line_width=0.5, row=1, col=1)

    # Row 2 left — scatter
    lim = max(abs(y).max(), abs(yp).max()) * 1.05
    fig.add_trace(go.Scattergl(
        x=y, y=yp, mode="markers",
        marker=dict(size=2, color="#8E44AD", opacity=0.15),
        showlegend=False,
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=[-lim, lim], y=[-lim, lim], mode="lines",
        line=dict(color="grey", dash="dash", width=1), showlegend=False,
    ), row=2, col=1)

    # Row 2 right — residuals histogram
    fig.add_trace(go.Histogram(
        x=residuals, nbinsx=80,
        marker_color="rgba(39,174,96,0.65)", showlegend=False,
    ), row=2, col=3)

    # Row 3 — IRF panels
    for i, (lbl, clr) in enumerate(zip(_REG_LABELS, _IRF_COLORS)):
        fig.add_trace(go.Scatter(
            x=lags_x, y=beta[i], mode="lines",
            line=dict(width=2.0, color=clr),
            fill="tozeroy",
            fillcolor=f"rgba({_hex_to_rgb(clr)},0.15)",
            showlegend=False, name=lbl,
        ), row=3, col=i + 1)
        fig.add_hline(y=0, line_dash="dash", line_color="grey",
                      line_width=0.7, row=3, col=i + 1)
        # peak annotation
        peak_k = int(np.argmax(np.abs(beta[i])))
        peak_v = beta[i, peak_k]
        fig.add_annotation(
            x=peak_k, y=peak_v, text=f"k={peak_k}h",
            showarrow=True, arrowhead=2, arrowsize=0.8,
            font=dict(size=8, color=clr), row=3, col=i + 1,
        )
        fig.update_xaxes(title_text="Lag k [h]", title_font_size=10,
                         row=3, col=i + 1)
        fig.update_yaxes(title_text="β [m/σ]", title_font_size=10,
                         row=3, col=i + 1)

    name_tag = f"<b>{station_name}</b>  ·  " if station_name else ""
    fig.update_layout(
        title=dict(
            text=(
                f"{name_tag}<b>MISO Lag Regression</b>  (L = {L} h)"
                f"<br><span style='font-size:11px;color:#666'>"
                f"R² = {reg['r2']:.3f}  ·  "
                f"RMSE = {reg['rmse'] * 100:.2f} cm  ·  "
                f"DW = {reg['dw']:.2f}  ·  n = {reg['n']:,}  ·  "
                f"{n_feat} × {L + 1} = {n_feat * (L + 1)} predictors</span>"
            ),
            font=dict(size=14),
        ),
        template="plotly_white", height=780,
        margin=dict(l=50, r=25, t=80, b=45),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1),
    )
    fig.update_xaxes(title_text="ε observed [m]", row=2, col=1)
    fig.update_yaxes(title_text="ε predicted [m]", row=2, col=1)
    fig.update_xaxes(title_text="Residual [m]", row=2, col=3)
    fig.update_yaxes(title_text="Count", row=2, col=3)
    fig.update_yaxes(title_text="Error [m]", row=1, col=1)

    return fig


# ── ACF plot ────────────────────────────────────────────────────────────

def make_acf_plot(
    reg: dict | None = None,
    station_name: str = "",
    df: pd.DataFrame | None = None,
) -> go.Figure:
    """ACF of regression residuals (preferred) or raw ε(t)."""
    fig = go.Figure()

    # Choose series
    if reg and reg.get("ok") and reg.get("residuals") is not None:
        series = reg["residuals"]
        meth = reg["method"].upper()
        if reg["method"] == "miso":
            meth += f" L={reg['lag']}h"
        acf_label = f"Regression Residuals ({meth})"
        dw_val = reg["dw"]
    elif df is not None and not df.empty and "error_m" in df.columns:
        series = df["error_m"].dropna().values
        acf_label = "Raw Error ε(t)"
        dw_val = float(durbin_watson(series)) if len(series) > 1 else np.nan
    else:
        fig.add_annotation(text="Insufficient data for ACF",
                           showarrow=False, font=dict(size=14, color="#888"))
        fig.update_layout(height=320, template="plotly_white")
        return fig

    if len(series) < 48:
        fig.add_annotation(text="Insufficient data for ACF",
                           showarrow=False, font=dict(size=14, color="#888"))
        fig.update_layout(height=320, template="plotly_white")
        return fig

    MAX_LAGS = min(168, len(series) // 2 - 1)
    acf_vals, confint = acf(series, nlags=MAX_LAGS, alpha=0.05, fft=True)
    lags = np.arange(MAX_LAGS + 1)
    ci_upper = confint[:, 1] - acf_vals
    ci_lower = acf_vals - confint[:, 0]

    # 95 % CI band
    fig.add_trace(go.Scatter(
        x=np.concatenate([lags, lags[::-1]]),
        y=np.concatenate([acf_vals + ci_upper,
                          (acf_vals - ci_lower)[::-1]]),
        fill="toself", fillcolor="rgba(41,128,185,0.12)",
        line=dict(color="rgba(0,0,0,0)"),
        showlegend=False, hoverinfo="skip",
    ))
    # Stems
    for k, r in zip(lags[1:], acf_vals[1:]):
        color = "#2980B9" if r >= 0 else "#E74C3C"
        fig.add_shape(type="line", x0=k, x1=k, y0=0, y1=r,
                      line=dict(color=color, width=1.0))
    fig.add_trace(go.Scatter(
        x=lags[1:], y=acf_vals[1:], mode="markers",
        marker=dict(size=3, color=[
            "#2980B9" if r >= 0 else "#E74C3C" for r in acf_vals[1:]
        ]),
        showlegend=False,
    ))
    fig.add_hline(y=0, line_color="grey", line_width=0.8)

    # Tidal reference
    tidal = [(6.21, "M4"), (12.0, "S2"), (12.42, "M2"),
             (23.93, "K1"), (25.82, "O1"), (48, "2d"), (168, "7d")]
    for lag_h, label in tidal:
        k = int(round(lag_h))
        if 1 <= k <= MAX_LAGS:
            fig.add_vline(x=k, line_dash="dot", line_color="#ddd",
                          line_width=0.8)
            fig.add_annotation(
                x=k, y=1.0, yref="paper", yshift=4,
                text=f"<sub>{label}</sub>",
                showarrow=False, font=dict(size=8, color="#aaa"),
            )

    title = f"ACF — {acf_label}"
    if station_name:
        title += f"  ·  {station_name}"

    dw_text = f"DW = {dw_val:.2f}" if not np.isnan(dw_val) else ""
    extra_annots = ()
    if dw_text:
        extra_annots = (
            go.layout.Annotation(
                text=dw_text, xref="paper", yref="paper",
                x=0.99, y=0.96, showarrow=False,
                font=dict(size=11, color="#555"),
                bgcolor="rgba(255,255,255,0.85)",
                bordercolor="#ddd", borderwidth=1, borderpad=3,
            ),
        )

    fig.update_layout(
        title=dict(text=title, font=dict(size=13)),
        xaxis_title="Lag [hours]",
        yaxis_title="Autocorrelation",
        xaxis=dict(range=[0, MAX_LAGS]),
        template="plotly_white",
        height=320,
        margin=dict(l=55, r=25, t=55, b=45),
        annotations=fig.layout.annotations + extra_annots,
    )
    return fig

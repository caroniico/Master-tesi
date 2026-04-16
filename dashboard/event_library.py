"""Event Library — detect, save, load and plot storm-surge events per station.

Events are persisted as JSON records in:
    <project_root>/data/event_library/<station_id>/events.json

Each event record (all fields are optional except id/station_id/peak_time):
{
    # ── Auto-filled on save ──────────────────────────────────────────
    "id":           "26473_20131205T1000",
    "station_id":   "26473",
    "station_name": "Sønderborg",
    "peak_time":    "2013-12-05T10:00:00",
    "peak_tg_m":    1.234,
    "duration_h":   18,
    "thresh_m":     0.80,
    "saved_at":     "2026-04-02T14:30:00",

    # ── Regression results (auto-filled by batch save) ───────────────
    "regression": {
        "window_days":  15,
        "lag":          72,
        "ols":          {"r2": 0.45, "rmse": 0.12, "coefs": {...}},
        "miso":         {"r2": 0.60, "rmse": 0.10, "coefs": {...}, "beta_matrix": [...]},
        "ridge":        {"r2": 0.44, "rmse": 0.12, "coefs": {...}, "alpha": 1.0},
        "ridge-miso":   {"r2": 0.62, "rmse": 0.09, "coefs": {...}, "beta_matrix": [...], "alpha": 1.0},
    },

    # ── User-editable fields ─────────────────────────────────────────
    "note":         "",          # free-text note
    "quality":      "",          # e.g. "good" / "poor" / "uncertain"
    "wind_dir":     "",          # dominant wind direction (e.g. "NW")
    "pressure_min_hpa": null,    # minimum SLP during event [hPa]
    "max_error_m":  null,        # max |ε| during event window [m]
    "tags":         [],          # list of free tags, e.g. ["westerly","long"]
    "exclude":      false        # True → exclude from analysis
}

EDITABLE_FIELDS — the set of fields the user can modify after saving.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Storage path ────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_LIB_DIR = _PROJECT_ROOT / "data" / "event_library"
_LIB_DIR.mkdir(parents=True, exist_ok=True)

# ── Schema of user-editable fields (name → {label, type, default}) ──────
EDITABLE_FIELDS: dict[str, dict] = {
    "note":             {"label": "Note",               "type": "text",   "default": ""},
    "quality":          {"label": "Quality",            "type": "select",
                         "options": ["", "good", "uncertain", "poor"],    "default": ""},
    "wind_dir":         {"label": "Wind direction",     "type": "text",   "default": ""},
    "pressure_min_hpa": {"label": "Min SLP [hPa]",     "type": "number", "default": None},
    "max_error_m":      {"label": "Max |ε| [m]",        "type": "number", "default": None},
    "tags":             {"label": "Tags (comma-sep.)",  "type": "tags",   "default": []},
    "exclude":          {"label": "Exclude from analysis", "type": "bool","default": False},
}

# ── Default event record template ───────────────────────────────────────
def _empty_event() -> dict:
    rec: dict[str, Any] = {}
    for key, meta in EDITABLE_FIELDS.items():
        rec[key] = meta["default"]
    return rec


# ═══════════════════════════════════════════════════════════════════════
# Detect events from a station DataFrame
# ═══════════════════════════════════════════════════════════════════════

def detect_events(
    df: pd.DataFrame,
    thresh_m: float = 0.80,
    gap_h: int = 12,
    max_events: int = 20,
) -> pd.DataFrame:
    """Return a DataFrame of storm surge events sorted by peak TG desc.

    Parameters
    ----------
    df        : station DataFrame with columns valid_time, tg_obs_m.
    thresh_m  : surge threshold [m].
    gap_h     : minimum gap in hours between separate events.
    max_events: maximum number of events to return (top by peak).

    Returns DataFrame with columns:
        event_id, peak_time, peak_tg_m, duration_h, start_time, end_time
    """
    needed = {"valid_time", "tg_obs_m"}
    if df.empty or not needed.issubset(df.columns):
        return pd.DataFrame()

    df = df.sort_values("valid_time").reset_index(drop=True)
    exc = df[df["tg_obs_m"] > thresh_m][["valid_time", "tg_obs_m"]].dropna().copy()
    if exc.empty:
        return pd.DataFrame()

    exc = exc.reset_index(drop=True)
    dt_diff = exc["valid_time"].diff().dt.total_seconds().fillna(0) / 3600
    exc["event_group"] = (dt_diff > gap_h).cumsum()

    records = []
    for grp_id, grp in exc.groupby("event_group"):
        peak_row = grp.loc[grp["tg_obs_m"].idxmax()]
        records.append({
            "event_id":   grp_id,
            "peak_time":  peak_row["valid_time"],
            "peak_tg_m":  float(peak_row["tg_obs_m"]),
            "duration_h": len(grp),
            "start_time": grp["valid_time"].min(),
            "end_time":   grp["valid_time"].max(),
        })

    ev_df = (
        pd.DataFrame(records)
        .nlargest(max_events, "peak_tg_m")
        .sort_values("peak_time")
        .reset_index(drop=True)
    )
    ev_df.index = ev_df.index + 1          # 1-based display index
    return ev_df


# ═══════════════════════════════════════════════════════════════════════
# Persistence
# ═══════════════════════════════════════════════════════════════════════

def _station_file(station_id: str) -> Path:
    d = _LIB_DIR / str(station_id)
    d.mkdir(parents=True, exist_ok=True)
    return d / "events.json"


def load_saved_events(station_id: str) -> list[dict]:
    """Load persisted events for a station ([] if none)."""
    fp = _station_file(station_id)
    if not fp.exists():
        return []
    try:
        with fp.open() as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return []


def save_event(
    station_id: str,
    station_name: str,
    peak_time: pd.Timestamp,
    peak_tg_m: float,
    duration_h: int,
    thresh_m: float,
    editable: dict[str, Any] | None = None,
) -> str:
    """Persist one event; return its id string.  Overwrites if already exists.

    Parameters
    ----------
    editable : dict with any subset of EDITABLE_FIELDS keys (+ arbitrary extras
               the user added as "custom" key-value pairs).
               Pass None to use all defaults.
    """
    ev_id = f"{station_id}_{pd.Timestamp(peak_time).strftime('%Y%m%dT%H%M')}"
    events = load_saved_events(station_id)
    # Remove duplicate (same id)
    events = [e for e in events if e.get("id") != ev_id]

    # Build the editable-fields portion
    ed: dict[str, Any] = _empty_event()
    if editable:
        for k, v in editable.items():
            if k == "tags" and isinstance(v, str):
                v = [t.strip() for t in v.split(",") if t.strip()]
            ed[k] = v

    events.append({
        "id":           ev_id,
        "station_id":   station_id,
        "station_name": station_name,
        "peak_time":    pd.Timestamp(peak_time).isoformat(),
        "peak_tg_m":    round(float(peak_tg_m), 4),
        "duration_h":   int(duration_h),
        "thresh_m":     float(thresh_m),
        "saved_at":     datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
        **ed,
    })
    # Sort by peak_time
    events.sort(key=lambda e: e["peak_time"])
    fp = _station_file(station_id)
    with fp.open("w") as f:
        json.dump(events, f, indent=2, ensure_ascii=False)
    return ev_id


def update_event(station_id: str, ev_id: str, editable: dict[str, Any]) -> bool:
    """Update only the editable fields of an existing saved event.

    Returns True on success, False if event not found.
    """
    events = load_saved_events(station_id)
    for ev in events:
        if ev.get("id") == ev_id:
            for k, v in editable.items():
                if k == "tags" and isinstance(v, str):
                    v = [t.strip() for t in v.split(",") if t.strip()]
                ev[k] = v
            fp = _station_file(station_id)
            with fp.open("w") as f:
                json.dump(events, f, indent=2, ensure_ascii=False)
            return True
    return False


def delete_event(station_id: str, ev_id: str) -> None:
    """Remove a saved event by its id."""
    events = load_saved_events(station_id)
    events = [e for e in events if e.get("id") != ev_id]
    fp = _station_file(station_id)
    with fp.open("w") as f:
        json.dump(events, f, indent=2, ensure_ascii=False)


def all_saved_stations() -> list[str]:
    """Return list of station_ids that have at least one saved event."""
    return [d.name for d in _LIB_DIR.iterdir()
            if d.is_dir() and (d / "events.json").exists()]


# ═══════════════════════════════════════════════════════════════════════
# Batch save — run all 4 regression methods per event window
# ═══════════════════════════════════════════════════════════════════════

def _reg_summary(reg: dict) -> dict | None:
    """Extract a JSON-serialisable summary from a compute_regression result."""
    if not reg.get("ok"):
        return None
    out: dict[str, Any] = {
        "r2":   round(float(reg["r2"]),   4),
        "rmse": round(float(reg["rmse"]), 4),
        "bias": round(float(reg["bias"]), 4),
        "dw":   round(float(reg["dw"]),   4) if reg.get("dw") is not None else None,
        "n":    int(reg["n"]),
        "coefs": {k: round(float(v), 6) for k, v in reg["coefs"].items()},
    }
    if reg.get("alpha") is not None:
        out["alpha"] = round(float(reg["alpha"]), 4)
    if reg.get("beta_matrix") is not None:
        out["beta_matrix"] = reg["beta_matrix"].tolist()
    return out


def batch_save_events(
    df: pd.DataFrame,
    station_id: str,
    station_name: str,
    thresh_m: float = 0.80,
    lag: int = 72,
    window_days: int = 15,
    remove_bias: bool = False,
    include_tide: bool = False,
    lat: float | None = None,
    lon: float | None = None,
) -> tuple[int, int]:
    """Detect all events above *thresh_m*, run all 4 regression methods on a
    ±*window_days* window around each peak, and persist to the event library.

    Existing events for the same station are kept; records with the same id
    are overwritten (regression results updated).

    Parameters
    ----------
    df           : full station DataFrame.
    station_id   : station identifier string.
    station_name : human-readable name.
    thresh_m     : surge threshold [m].
    lag          : max lag L for MISO / ridge-MISO [hours].
    window_days  : half-width of the regression window around the peak.
    remove_bias  : whether to subtract the per-station bias before regression.
    include_tide : whether to add DTU10 tidal signal as a regression feature.
    lat, lon     : station coordinates (required when include_tide=True).

    Returns
    -------
    (n_saved, n_skipped) — events saved and events skipped (no valid regression).
    """
    # Lazy import to avoid circular dependency (figures imports nothing from evlib)
    from . import figures as _fig

    events_df = detect_events(df, thresh_m=thresh_m)
    if events_df.empty:
        return 0, 0

    _METHODS = ["ols", "miso", "ridge", "ridge-miso"]

    n_saved = 0
    n_skipped = 0

    for _, row in events_df.iterrows():
        peak_time = pd.Timestamp(row["peak_time"])

        # ── Slice a ±window_days window around the event peak ──────────
        t0 = peak_time - pd.Timedelta(days=window_days)
        t1 = peak_time + pd.Timedelta(days=window_days)
        win_df = df[(df["valid_time"] >= t0) & (df["valid_time"] <= t1)].copy()

        # ── Run all 4 methods ──────────────────────────────────────────
        reg_results: dict[str, Any] = {
            "window_days": window_days,
            "lag": lag,
            "include_tide": include_tide,
        }
        any_ok = False
        for method in _METHODS:
            use_lag = lag if method in ("miso", "ridge-miso") else 0
            reg = _fig.compute_regression(
                win_df, method=method,
                lag=use_lag, remove_bias=remove_bias,
                include_tide=include_tide, lat=lat, lon=lon,
            )
            summary = _reg_summary(reg)
            reg_results[method] = summary
            if summary is not None:
                any_ok = True

        # Auto-compute max |ε| inside the event window
        max_err: float | None = None
        if "error_m" in win_df.columns and not win_df["error_m"].isna().all():
            max_err = round(float(win_df["error_m"].abs().max()), 4)

        editable: dict[str, Any] = {
            "max_error_m": max_err,
        }

        # ── Persist the event with embedded regression results ─────────
        ev_id = f"{station_id}_{peak_time.strftime('%Y%m%dT%H%M')}"
        events = load_saved_events(station_id)
        events = [e for e in events if e.get("id") != ev_id]

        ed = _empty_event()
        ed.update(editable)

        events.append({
            "id":           ev_id,
            "station_id":   station_id,
            "station_name": station_name,
            "peak_time":    peak_time.isoformat(),
            "peak_tg_m":    round(float(row["peak_tg_m"]), 4),
            "duration_h":   int(row["duration_h"]),
            "thresh_m":     float(thresh_m),
            "saved_at":     datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
            "regression":   reg_results,
            **ed,
        })
        events.sort(key=lambda e: e["peak_time"])
        fp = _station_file(station_id)
        with fp.open("w") as f:
            json.dump(events, f, indent=2, ensure_ascii=False)

        if any_ok:
            n_saved += 1
        else:
            n_skipped += 1

    return n_saved, n_skipped


# ═══════════════════════════════════════════════════════════════════════
# Figures
# ═══════════════════════════════════════════════════════════════════════

def make_event_zoom_plot(
    df: pd.DataFrame,
    peak_time: pd.Timestamp,
    station_name: str = "",
    thresh_m: float = 0.80,
    window_days: int = 10,
) -> go.Figure:
    """±window_days zoom around a surge peak: TG obs (top) + error ε (bottom)."""
    peak_time = pd.Timestamp(peak_time)
    t_start = peak_time - pd.Timedelta(days=window_days)
    t_end   = peak_time + pd.Timedelta(days=window_days)

    win = df[
        (df["valid_time"] >= t_start) &
        (df["valid_time"] <= t_end)
    ].copy()

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.6, 0.4],
        subplot_titles=["Sea level TG obs [m]", "Error ε(t) = Model − TG obs [m]"],
    )

    # ── TOP: TG obs ──────────────────────────────────────────────────
    fig.add_trace(go.Scattergl(
        x=win["valid_time"], y=win["tg_obs_m"],
        mode="lines", line=dict(width=1.0, color="#2980B9"), name="TG obs",
    ), row=1, col=1)

    # threshold line + shading
    fig.add_hline(y=thresh_m, line_dash="dash", line_color="#E74C3C",
                  line_width=1.0, row=1, col=1,
                  annotation_text=f"thresh = {thresh_m} m",
                  annotation_position="top right",
                  annotation_font=dict(size=9, color="#E74C3C"))

    surge_mask = win["tg_obs_m"] > thresh_m
    if surge_mask.any():
        w_surge = win[surge_mask]
        fig.add_trace(go.Scatter(
            x=pd.concat([w_surge["valid_time"], w_surge["valid_time"].iloc[::-1]]),
            y=pd.concat([w_surge["tg_obs_m"],
                         pd.Series([thresh_m] * len(w_surge), index=w_surge.index)
                         .iloc[::-1]]),
            fill="toself", fillcolor="rgba(231,76,60,0.15)",
            line=dict(color="rgba(0,0,0,0)"),
            showlegend=False, hoverinfo="skip",
        ), row=1, col=1)

    # peak marker
    peak_tg = float(win.loc[win["valid_time"] == peak_time, "tg_obs_m"].values[0]) \
        if peak_time in win["valid_time"].values else float(win["tg_obs_m"].max())
    fig.add_trace(go.Scatter(
        x=[peak_time], y=[peak_tg],
        mode="markers+text",
        marker=dict(size=8, color="#E74C3C", symbol="diamond"),
        text=[f"  {peak_tg:.3f} m"],
        textposition="top right",
        textfont=dict(size=9),
        showlegend=False,
    ), row=1, col=1)
    fig.add_vline(x=peak_time, line_dash="dot", line_color="grey",
                  line_width=0.9, row=1, col=1)

    # ── BOTTOM: ε ────────────────────────────────────────────────────
    if "error_m" in win.columns:
        fig.add_trace(go.Scattergl(
            x=win["valid_time"], y=win["error_m"],
            mode="lines", line=dict(width=0.8, color="#8E44AD"),
            fill="tozeroy", fillcolor="rgba(142,68,173,0.12)",
            name="ε(t)",
        ), row=2, col=1)
        fig.add_hline(y=0, line_dash="dash", line_color="grey",
                      line_width=0.5, row=2, col=1)
        fig.add_vline(x=peak_time, line_dash="dot", line_color="grey",
                      line_width=0.9, row=2, col=1)

    date_str = peak_time.strftime("%d %b %Y  %H:%M")
    title = f"Storm surge event — {station_name}  ·  Peak: {date_str}  ({peak_tg:.3f} m)"

    fig.update_layout(
        title=dict(text=title, font=dict(size=13)),
        template="plotly_white",
        height=420,
        margin=dict(l=55, r=20, t=55, b=45),
        legend=dict(orientation="h", y=1.08, x=1, xanchor="right"),
    )
    fig.update_yaxes(title_text="TG [m]",    row=1, col=1)
    fig.update_yaxes(title_text="Error [m]", row=2, col=1)
    return fig


def make_events_overview_plot(
    df: pd.DataFrame,
    events_df: pd.DataFrame,
    station_name: str = "",
    thresh_m: float = 0.80,
) -> go.Figure:
    """Full time series with detected surge events highlighted."""
    fig = go.Figure()

    if df.empty:
        fig.add_annotation(text="No data", showarrow=False,
                           font=dict(size=14, color="#888"))
        fig.update_layout(height=260, template="plotly_white")
        return fig

    fig.add_trace(go.Scattergl(
        x=df["valid_time"], y=df["tg_obs_m"],
        mode="lines", line=dict(width=0.6, color="#2980B9"),
        name="TG obs",
    ))
    fig.add_hline(y=thresh_m, line_dash="dash", line_color="#E74C3C",
                  line_width=1.0,
                  annotation_text=f"{thresh_m} m",
                  annotation_position="top right",
                  annotation_font=dict(size=9, color="#E74C3C"))

    if not events_df.empty:
        fig.add_trace(go.Scatter(
            x=events_df["peak_time"],
            y=events_df["peak_tg_m"],
            mode="markers+text",
            marker=dict(size=7, color="#E74C3C", symbol="diamond",
                        line=dict(width=1, color="white")),
            text=["#" + str(i) for i in events_df.index],
            textposition="top center",
            textfont=dict(size=8, color="#E74C3C"),
            name="Events",
        ))

    fig.update_layout(
        title=dict(
            text=f"Detected surge events — {station_name}  "
                 f"(threshold = {thresh_m} m)",
            font=dict(size=12),
        ),
        template="plotly_white",
        height=260,
        margin=dict(l=55, r=20, t=45, b=40),
        yaxis_title="TG [m]",
    )
    return fig

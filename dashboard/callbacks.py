"""Dash callbacks: wire map clicks, dropdown, date picker to plots."""
from __future__ import annotations

import json as _json

import pandas as pd
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
from dash import ALL, MATCH, Input, Output, State, callback, ctx, html, dcc, no_update

from . import data_loader, figures
from . import event_library as evlib


# ── Map click → dropdown sync ────────────────────────────────────────────
@callback(
    Output("station-dropdown", "value"),
    Input({"type": "station-marker", "index": ALL}, "n_clicks"),
    State("station-dropdown", "value"),
    prevent_initial_call=True,
)
def marker_click_to_dropdown(n_clicks_list, current_value):
    """When a map marker is clicked, update the station dropdown."""
    if not ctx.triggered_id or not any(n_clicks_list):
        return current_value
    return ctx.triggered_id["index"]


# ── Show/hide lag slider based on regression method ──────────────────────
@callback(
    Output("lag-control", "style"),
    Input("reg-method", "value"),
)
def toggle_lag_slider(method):
    """Show the lag slider only when a lag-based method is selected."""
    if method in ("miso", "ridge-miso"):
        return {"display": "block"}
    return {"display": "none"}


# ── Station + DateRange → plots + stats ──────────────────────────────────
@callback(
    Output("time-plot", "figure"),
    Output("psd-plot", "figure"),
    Output("stats-card", "children"),
    Input("station-dropdown", "value"),
    Input("compare-dropdown", "value"),
    Input("date-range", "start_date"),
    Input("date-range", "end_date"),
    Input("remove-bias-toggle", "value"),
)
def update_plots(station_id, compare_ids, start_date, end_date, remove_bias):
    """Fetch data for selected station/period and regenerate plots."""
    remove_bias = bool(remove_bias)
    if not station_id:
        empty = figures.make_time_plot(pd.DataFrame())
        return empty, empty, ""

    # Look up station name
    stations = data_loader.get_stations()
    name_map = {s["id"]: s["name"] for s in stations}
    station_name = name_map.get(station_id, station_id)

    # Slice data for primary station
    df = data_loader.get_station_data(station_id, start_date, end_date)

    # Build comparison list: [(df, name), …]
    compare_data = []
    if compare_ids:
        for cid in compare_ids:
            if cid == station_id:
                continue
            cdf = data_loader.get_station_data(cid, start_date, end_date)
            cname = name_map.get(cid, cid)
            compare_data.append((cdf, cname))

    # Build figures
    fig_time = figures.make_time_plot(df, station_name,
                                      compare_data=compare_data,
                                      remove_bias=remove_bias,
                                      start_date=start_date,
                                      end_date=end_date)
    fig_psd = figures.make_psd_plot(df, station_name=station_name,
                                    compare_data=compare_data,
                                    remove_bias=remove_bias,
                                    start_date=start_date,
                                    end_date=end_date)

    # Statistics
    stats = figures.compute_stats(df, remove_bias=remove_bias)
    stats_card = _build_stats_card(stats)

    return fig_time, fig_psd, stats_card


# ── Regression tab → regression-plot + acf-plot ──────────────────────────
@callback(
    Output("regression-plot", "figure"),
    Output("acf-plot", "figure"),
    Output("irf-plot", "figure"),
    Input("station-dropdown", "value"),
    Input("date-range", "start_date"),
    Input("date-range", "end_date"),
    Input("reg-method", "value"),
    Input("lag-slider", "value"),
    Input("remove-bias-toggle", "value"),
    Input("include-tide-toggle", "value"),
)
def update_regression(station_id, start_date, end_date, reg_method, lag_value,
                      remove_bias, include_tide):
    """Compute and render regression figures when the Regression tab is active."""
    remove_bias   = bool(remove_bias)
    include_tide  = bool(include_tide)
    empty_reg = figures.make_regression_plot({}, "")
    empty_acf = figures.make_acf_plot({}, "")
    empty_irf = figures.make_irf_plot({}, "")
    if not station_id:
        return empty_reg, empty_acf, empty_irf

    stations = data_loader.get_stations()
    name_map = {s["id"]: s["name"] for s in stations}
    station_name = name_map.get(station_id, station_id)
    station_meta = next((s for s in stations if s["id"] == station_id), {})
    lat = station_meta.get("lat")
    lon = station_meta.get("lon")

    df = data_loader.get_station_data(station_id, start_date, end_date)
    if df.empty:
        return empty_reg, empty_acf, empty_irf

    lag = lag_value if reg_method in ("miso", "ridge-miso") else 0
    reg = figures.compute_regression(
        df, method=reg_method or "ols",
        lag=lag or 0, remove_bias=remove_bias,
        include_tide=include_tide, lat=lat, lon=lon,
    )

    fig_reg = figures.make_regression_plot(reg, station_name, start_date, end_date)
    fig_acf = figures.make_acf_plot(reg, station_name, remove_bias=remove_bias,
                                     start_date=start_date, end_date=end_date)
    fig_irf = figures.make_irf_plot(reg, station_name, start_date, end_date)
    return fig_reg, fig_acf, fig_irf


# ── Error Statistics tab → error-stats-plot ──────────────────────────────
@callback(
    Output("error-stats-plot", "figure"),
    Input("station-dropdown", "value"),
    Input("date-range", "start_date"),
    Input("date-range", "end_date"),
    Input("remove-bias-toggle", "value"),
)
def update_error_stats(station_id, start_date, end_date, remove_bias):
    """Generate error distribution histogram for the selected station."""
    remove_bias = bool(remove_bias)
    if not station_id:
        return figures.make_error_stats_plot(pd.DataFrame())

    stations = data_loader.get_stations()
    name_map = {s["id"]: s["name"] for s in stations}
    station_name = name_map.get(station_id, station_id)

    df = data_loader.get_station_data(station_id, start_date, end_date)

    return figures.make_error_stats_plot(
        df, station_name,
        remove_bias=remove_bias,
        start_date=start_date,
        end_date=end_date,
    )


def _build_stats_card(stats: dict):
    """Build a Bootstrap card showing error statistics."""
    if stats["total"] == 0:
        return dbc.Alert("No data for this selection.", color="warning")

    rows = [
        ("Total timestamps", f"{stats['total']:,}"),
        ("Valid (matched)", f"{stats['valid']:,}  ({stats['pct_valid']}%)"),
        ("Std (σ)", f"{stats['std']:.5f} m"),
    ]

    return dbc.Card(
        dbc.CardBody([
            html.Table(
                [html.Tr([
                    html.Td(label, className="pe-3 text-muted small"),
                    html.Td(html.Strong(value), className="small"),
                ]) for label, value in rows],
                className="mb-0",
            )
        ]),
        className="shadow-sm",
    )


# ══════════════════════════════════════════════════════════════════════
#  EVENT LIBRARY callbacks
# ══════════════════════════════════════════════════════════════════════

@callback(
    Output("ev-overview-plot",   "figure"),
    Output("ev-detected-table",  "children"),
    Input("station-dropdown",    "value"),
    Input("date-range",          "start_date"),
    Input("date-range",          "end_date"),
    Input("surge-thresh-slider", "value"),
)
def update_event_library_detect(station_id, start_date, end_date, thresh_m):
    """Detect surge events and render the overview plot + detected-events table."""
    thresh_m = float(thresh_m or 0.80)

    empty_fig = evlib.make_events_overview_plot(
        pd.DataFrame(), pd.DataFrame(), thresh_m=thresh_m)

    if not station_id:
        return empty_fig, dbc.Alert("Select a station.", color="secondary",
                                     className="small py-2")

    stations = data_loader.get_stations()
    name_map  = {s["id"]: s["name"] for s in stations}
    station_name = name_map.get(station_id, station_id)

    df = data_loader.get_station_data(station_id, start_date, end_date)
    if df.empty:
        return empty_fig, dbc.Alert("No data for this selection.",
                                     color="warning", className="small py-2")

    events_df = evlib.detect_events(df, thresh_m=thresh_m)
    ov_fig    = evlib.make_events_overview_plot(
        df, events_df, station_name=station_name, thresh_m=thresh_m)

    if events_df.empty:
        return ov_fig, dbc.Alert(
            f"No events above {thresh_m} m found.", color="info",
            className="small py-2")

    # Build table rows
    rows = []
    for i, row in events_df.iterrows():
        peak_str = pd.Timestamp(row["peak_time"]).strftime("%d %b %Y  %H:%M")
        rows.append(html.Tr([
            html.Td(html.Strong(f"#{i}"), className="text-center pe-2",
                    style={"width": "40px"}),
            html.Td(peak_str,                         className="small"),
            html.Td(f"{row['peak_tg_m']:.3f} m",      className="small text-center"),
            html.Td(f"{row['duration_h']} h",          className="small text-center"),
            html.Td(
                dbc.Button(
                    html.I(className="bi bi-search"),
                    id={"type": "ev-view-btn", "index": i},
                    size="sm", color="link", className="p-0",
                    title="Zoom",
                ),
                className="text-center",
            ),
            html.Td(
                dbc.Button(
                    html.I(className="bi bi-bookmark-plus"),
                    id={"type": "ev-add-btn", "index": i},
                    size="sm", color="link", className="p-0 text-success",
                    title="Save to library",
                ),
                className="text-center",
            ),
        ]))

    table = dbc.Table(
        [
            html.Thead(html.Tr([
                html.Th("#",           className="text-center", style={"width": "40px"}),
                html.Th("Peak time"),
                html.Th("Peak TG",     className="text-center"),
                html.Th("Duration",    className="text-center"),
                html.Th("",            style={"width": "36px"}),
                html.Th("",            style={"width": "36px"}),
            ])),
            html.Tbody(rows),
        ],
        bordered=False, hover=True, size="sm",
        className="mb-0",
    )
    return ov_fig, table


# ── Batch-save all detected events with regression results ────────────────
@callback(
    Output("ev-batch-feedback",   "children"),
    Output("ev-batch-feedback",   "is_open"),
    Output("ev-saved-table",      "children", allow_duplicate=True),
    Input("ev-batch-save-btn",    "n_clicks"),
    State("station-dropdown",     "value"),
    State("date-range",           "start_date"),
    State("date-range",           "end_date"),
    State("surge-thresh-slider",  "value"),
    State("lag-slider",           "value"),
    State("remove-bias-toggle",   "value"),
    State("include-tide-toggle",  "value"),
    prevent_initial_call=True,
)
def batch_save_all_events(n_clicks, station_id, start_date, end_date,
                          thresh_m, lag_value, remove_bias, include_tide):
    """Detect all events above threshold, run all 4 regression methods on a
    ±15-day window around each peak and save everything to the library."""
    if not n_clicks or not station_id:
        return no_update, no_update, no_update

    thresh_m     = float(thresh_m or 0.80)
    lag          = int(lag_value or 72)
    remove_bias  = bool(remove_bias)
    include_tide = bool(include_tide)

    stations     = data_loader.get_stations()
    name_map     = {s["id"]: s["name"] for s in stations}
    station_name = name_map.get(station_id, station_id)
    station_meta = next((s for s in stations if s["id"] == station_id), {})
    lat = station_meta.get("lat")
    lon = station_meta.get("lon")

    df = data_loader.get_station_data(station_id, start_date, end_date)
    if df.empty:
        return "⚠️ No data for this selection.", True, no_update

    n_saved, n_skipped = evlib.batch_save_events(
        df,
        station_id=station_id,
        station_name=station_name,
        thresh_m=thresh_m,
        lag=lag,
        window_days=15,
        remove_bias=remove_bias,
        include_tide=include_tide,
        lat=lat,
        lon=lon,
    )

    if n_saved == 0 and n_skipped == 0:
        msg = f"ℹ️ No events found above {thresh_m} m in this period."
    else:
        tide_tag = " + DTU10 tide" if include_tide else ""
        msg = (
            f"⚡ Batch save complete — {n_saved} event(s) saved "
            f"(±15 days, lag={lag}h, thresh={thresh_m} m{tide_tag}). "
            + (f"{n_skipped} skipped (no valid regression)." if n_skipped else "")
        )

    return msg, True, _build_saved_table(station_id)


# ── Open edit/save modal from detected table ──────────────────────────────
@callback(
    Output("ev-edit-store",       "data"),
    Output("ev-modal-title",      "children"),
    Output("ev-modal-peak-time",  "children"),
    Output("ev-modal-peak-tg",    "children"),
    Output("ev-modal-duration",   "children"),
    Output("ev-field-note",       "value"),
    Output("ev-field-quality",    "value"),
    Output("ev-field-wind-dir",   "value"),
    Output("ev-field-pressure",   "value"),
    Output("ev-field-max-error",  "value"),
    Output("ev-field-tags",       "value"),
    Output("ev-field-exclude",    "value"),
    Output("ev-custom-fields-container", "children"),
    # Triggered by:
    Input({"type": "ev-add-btn",       "index": ALL}, "n_clicks"),
    Input({"type": "ev-edit-saved-btn","index": ALL}, "n_clicks"),
    Input("ev-modal-cancel-btn",  "n_clicks"),
    # State
    State("station-dropdown",     "value"),
    State("date-range",           "start_date"),
    State("date-range",           "end_date"),
    State("surge-thresh-slider",  "value"),
    prevent_initial_call=True,
)
def open_edit_modal(add_clicks, edit_saved_clicks, cancel_clicks,
                    station_id, start_date, end_date, thresh_m):
    """Open the save/edit modal with either a detected or a saved event."""
    _none14 = (False, no_update, "", "", "", "", "", "", "", None, None, "", False, [])

    if not ctx.triggered_id:
        return _none14

    triggered = ctx.triggered_id
    thresh_m = float(thresh_m or 0.80)

    # Close modal on cancel only
    if triggered == "ev-modal-cancel-btn":
        return (False,) + (no_update,) * 13

    # ── Open for a NEW event from detected table ──────────────────────
    if isinstance(triggered, dict) and triggered.get("type") == "ev-add-btn":
        ev_index = triggered["index"]
        if not station_id:
            return _none14
        df = data_loader.get_station_data(station_id, start_date, end_date)
        events_df = evlib.detect_events(df, thresh_m=thresh_m)
        if events_df.empty or ev_index not in events_df.index:
            return _none14
        row = events_df.loc[ev_index]
        peak_str = pd.Timestamp(row["peak_time"]).strftime("%d %b %Y  %H:%M")
        store = {
            "mode":        "new",
            "station_id":  station_id,
            "peak_time":   str(row["peak_time"]),
            "peak_tg_m":   float(row["peak_tg_m"]),
            "duration_h":  int(row["duration_h"]),
            "thresh_m":    thresh_m,
        }
        return (True, store,
                f"Save event — {peak_str}",
                peak_str,
                f"{row['peak_tg_m']:.3f} m",
                f"{row['duration_h']} h",
                "", "", "", None, None, "", False, [])

    # ── Open for EDITING an existing saved event ──────────────────────
    if isinstance(triggered, dict) and triggered.get("type") == "ev-edit-saved-btn":
        ev_id = triggered["index"]
        if not station_id:
            return _none14
        saved = evlib.load_saved_events(station_id)
        match = [e for e in saved if e["id"] == ev_id]
        if not match:
            return _none14
        ev = match[0]
        peak_str = pd.Timestamp(ev["peak_time"]).strftime("%d %b %Y  %H:%M")
        store = {
            "mode":        "edit",
            "station_id":  station_id,
            "ev_id":       ev_id,
            "peak_time":   ev["peak_time"],
            "peak_tg_m":   ev["peak_tg_m"],
            "duration_h":  ev["duration_h"],
            "thresh_m":    ev.get("thresh_m", thresh_m),
        }
        # Populate custom fields rows
        custom_rows = _build_custom_field_rows(ev.get("custom_fields", {}))
        tags_str = ", ".join(ev.get("tags", []))
        return (True, store,
                f"Edit event — {peak_str}",
                peak_str,
                f"{ev['peak_tg_m']:.3f} m",
                f"{ev['duration_h']} h",
                ev.get("note", ""),
                ev.get("quality", ""),
                ev.get("wind_dir", ""),
                ev.get("pressure_min_hpa"),
                ev.get("max_error_m"),
                tags_str,
                ev.get("exclude", False),
                custom_rows)

    return _none14


# ── Add a custom field row ────────────────────────────────────────────────
@callback(
    Output("ev-custom-fields-container", "children", allow_duplicate=True),
    Input("ev-add-custom-field-btn",     "n_clicks"),
    State("ev-custom-fields-container",  "children"),
    prevent_initial_call=True,
)
def add_custom_field_row(n_clicks, current_rows):
    """Append a new empty key-value row to the custom fields container."""
    rows = current_rows or []
    idx  = len(rows)
    rows.append(_make_custom_row(idx, "", ""))
    return rows


# ── Remove a custom field row ─────────────────────────────────────────────
@callback(
    Output("ev-custom-fields-container", "children", allow_duplicate=True),
    Input({"type": "ev-rm-custom-row", "index": ALL}, "n_clicks"),
    State("ev-custom-fields-container",  "children"),
    prevent_initial_call=True,
)
def remove_custom_field_row(rm_clicks, current_rows):
    """Remove the row whose remove-button was clicked."""
    if not ctx.triggered_id or not any(c for c in rm_clicks if c):
        return no_update
    rm_idx = ctx.triggered_id["index"]
    rows   = current_rows or []
    # Rebuild without the removed row, re-index
    kept   = [r for i, r in enumerate(rows) if i != rm_idx]
    return [_make_custom_row(i,
                             _extract_row_key(r),
                             _extract_row_val(r))
            for i, r in enumerate(kept)]


# ── Actually save / update event from modal ───────────────────────────────
@callback(
    Output("ev-save-feedback",    "children"),
    Output("ev-save-feedback",    "is_open"),
    Output("ev-saved-table",      "children"),
    Output("ev-edit-modal",       "is_open", allow_duplicate=True),
    Input("ev-modal-save-btn",    "n_clicks"),
    State("ev-edit-store",        "data"),
    State("ev-field-note",        "value"),
    State("ev-field-quality",     "value"),
    State("ev-field-wind-dir",    "value"),
    State("ev-field-pressure",    "value"),
    State("ev-field-max-error",   "value"),
    State("ev-field-tags",        "value"),
    State("ev-field-exclude",     "value"),
    State("ev-custom-fields-container", "children"),
    prevent_initial_call=True,
)
def save_event_from_modal(n_clicks, store, note, quality, wind_dir,
                          pressure, max_error, tags, exclude, custom_rows):
    """Persist a new event or update an existing one with all editable fields."""
    if not store:
        return no_update, no_update, no_update, no_update

    station_id = store.get("station_id")
    if not station_id:
        return no_update, no_update, no_update, no_update

    # Collect standard editable fields
    editable = {
        "note":             (note or "").strip(),
        "quality":          (quality or ""),
        "wind_dir":         (wind_dir or "").strip(),
        "pressure_min_hpa": pressure,
        "max_error_m":      max_error,
        "tags":             (tags or ""),
        "exclude":          bool(exclude),
    }
    # Collect custom key-value fields (skip empty keys)
    custom = {}
    for row in (custom_rows or []):
        k = _extract_row_key(row).strip()
        v = _extract_row_val(row).strip()
        if k:
            custom[k] = v
    if custom:
        editable["custom_fields"] = custom

    # Save / update
    stations     = data_loader.get_stations()
    name_map     = {s["id"]: s["name"] for s in stations}
    station_name = name_map.get(station_id, station_id)

    if store.get("mode") == "edit":
        evlib.update_event(station_id, store["ev_id"], editable)
        peak_str = pd.Timestamp(store["peak_time"]).strftime("%d %b %Y %H:%M")
        msg = f"✏️ Updated: {peak_str}"
    else:
        evlib.save_event(
            station_id=station_id,
            station_name=station_name,
            peak_time=pd.Timestamp(store["peak_time"]),
            peak_tg_m=float(store["peak_tg_m"]),
            duration_h=int(store["duration_h"]),
            thresh_m=float(store["thresh_m"]),
            editable=editable,
        )
        peak_str = pd.Timestamp(store["peak_time"]).strftime("%d %b %Y %H:%M")
        msg = f"✅ Saved: {peak_str}  ({store['peak_tg_m']:.3f} m)"

    return msg, True, _build_saved_table(station_id), False


# ── Delete from saved library ─────────────────────────────────────────────
@callback(
    Output("ev-save-feedback",   "children", allow_duplicate=True),
    Output("ev-save-feedback",   "is_open",  allow_duplicate=True),
    Output("ev-saved-table",     "children", allow_duplicate=True),
    Input({"type": "ev-del-btn", "index": ALL}, "n_clicks"),
    State("station-dropdown",    "value"),
    prevent_initial_call=True,
)
def handle_delete_event(del_clicks, station_id):
    """Delete a saved event and refresh the table."""
    if not ctx.triggered_id or not station_id:
        return no_update, no_update, no_update
    triggered = ctx.triggered_id
    if not isinstance(triggered, dict) or triggered.get("type") != "ev-del-btn":
        return no_update, no_update, no_update
    if not any(c for c in del_clicks if c):
        return no_update, no_update, no_update

    evlib.delete_event(station_id, triggered["index"])
    return "🗑 Event deleted.", True, _build_saved_table(station_id)


# ── Store selected event for zoom plot (from detected table) ──────────────
@callback(
    Output("ev-selected-store", "data"),
    Input({"type": "ev-view-btn", "index": ALL}, "n_clicks"),
    State("station-dropdown",    "value"),
    State("date-range",          "start_date"),
    State("date-range",          "end_date"),
    State("surge-thresh-slider", "value"),
    prevent_initial_call=True,
)
def store_selected_event(view_clicks, station_id,
                         start_date, end_date, thresh_m):
    """Store the detected event data for the zoom plot."""
    if not ctx.triggered_id or not any(c for c in view_clicks if c):
        return no_update
    triggered = ctx.triggered_id
    if not isinstance(triggered, dict) or triggered.get("type") != "ev-view-btn":
        return no_update
    if not station_id:
        return no_update

    thresh_m  = float(thresh_m or 0.80)
    ev_index  = triggered["index"]
    df        = data_loader.get_station_data(station_id, start_date, end_date)
    events_df = evlib.detect_events(df, thresh_m=thresh_m)
    if events_df.empty or ev_index not in events_df.index:
        return no_update

    row = events_df.loc[ev_index]
    return {
        "station_id": station_id,
        "peak_time":  str(row["peak_time"]),
        "peak_tg_m":  float(row["peak_tg_m"]),
        "duration_h": int(row["duration_h"]),
        "thresh_m":   thresh_m,
        "source":     "detected",
    }


# ── Zoom plot ─────────────────────────────────────────────────────────────
@callback(
    Output("ev-zoom-plot", "figure"),
    Input("ev-selected-store",  "data"),
    Input({"type": "ev-view-saved-btn", "index": ALL}, "n_clicks"),
    State("station-dropdown",   "value"),
    State("date-range",         "start_date"),
    State("date-range",         "end_date"),
    State("surge-thresh-slider","value"),
    prevent_initial_call=True,
)
def update_ev_zoom(store_data, view_saved_clicks, station_id,
                   start_date, end_date, thresh_m):
    """Render the event zoom plot from store or from saved-library view button."""
    thresh_m  = float(thresh_m or 0.80)
    triggered = ctx.triggered_id
    peak_time = None

    if isinstance(triggered, dict) and triggered.get("type") == "ev-view-saved-btn":
        ev_id = triggered["index"]
        if station_id:
            saved = evlib.load_saved_events(station_id)
            match = [e for e in saved if e["id"] == ev_id]
            if match:
                peak_time = pd.Timestamp(match[0]["peak_time"])
    elif store_data and store_data.get("peak_time"):
        peak_time = pd.Timestamp(store_data["peak_time"])

    if peak_time is None or not station_id:
        fig = go.Figure()
        fig.add_annotation(text="Click 🔍 on an event to zoom in",
                           showarrow=False, font=dict(size=14, color="#aaa"))
        fig.update_layout(height=420, template="plotly_white",
                          margin=dict(l=20, r=20, t=20, b=20))
        return fig

    df = data_loader.get_station_data(station_id, start_date, end_date)
    stations     = data_loader.get_stations()
    name_map     = {s["id"]: s["name"] for s in stations}
    station_name = name_map.get(station_id, station_id)

    return evlib.make_event_zoom_plot(
        df, peak_time, station_name=station_name, thresh_m=thresh_m)


# ── Export saved events as JSON ───────────────────────────────────────────
@callback(
    Output("ev-download", "data"),
    Input("ev-export-btn",     "n_clicks"),
    State("station-dropdown",  "value"),
    prevent_initial_call=True,
)
def export_events_json(n_clicks, station_id):
    """Download all saved events for the current station as a JSON file."""
    if not station_id or not n_clicks:
        return no_update
    saved = evlib.load_saved_events(station_id)
    filename = f"events_{station_id}.json"
    return dict(content=_json.dumps(saved, indent=2, ensure_ascii=False),
                filename=filename)


# ══════════════════════════════════════════════════════════════════════
#  Private helpers for event library
# ══════════════════════════════════════════════════════════════════════

def _make_custom_row(idx: int, key: str = "", val: str = "") -> html.Div:
    """One editable key-value row for the custom fields section of the modal."""
    return html.Div([
        dbc.Row([
            dbc.Col(
                dbc.Input(
                    id={"type": "ev-custom-key",  "index": idx},
                    placeholder="Field name",
                    value=key,
                    type="text",
                    size="sm",
                ),
                width=5,
            ),
            dbc.Col(
                dbc.Input(
                    id={"type": "ev-custom-val",  "index": idx},
                    placeholder="Value",
                    value=val,
                    type="text",
                    size="sm",
                ),
                width=6,
            ),
            dbc.Col(
                dbc.Button(
                    html.I(className="bi bi-x"),
                    id={"type": "ev-rm-custom-row", "index": idx},
                    size="sm", color="link",
                    className="p-0 text-danger",
                ),
                width=1,
            ),
        ], className="mb-1 g-1"),
    ], id=f"ev-custom-row-{idx}")


def _build_custom_field_rows(custom: dict) -> list:
    """Build the custom-field row list from a saved custom_fields dict."""
    return [_make_custom_row(i, k, v)
            for i, (k, v) in enumerate(custom.items())]


def _extract_row_key(row_div) -> str:
    """Extract the key string from a custom field row Div (best-effort)."""
    try:
        # row_div is a Div → children[0] is a Row → children[0] is Col → children is Input
        return row_div["props"]["children"][0]["props"]["children"][0][
            "props"]["children"]["props"].get("value", "") or ""
    except Exception:
        return ""


def _extract_row_val(row_div) -> str:
    """Extract the value string from a custom field row Div (best-effort)."""
    try:
        return row_div["props"]["children"][0]["props"]["children"][1][
            "props"]["children"]["props"].get("value", "") or ""
    except Exception:
        return ""


def _build_saved_table(station_id: str):
    """Build the saved-library table HTML for the given station."""
    saved = evlib.load_saved_events(station_id) if station_id else []
    if not saved:
        return html.P("No saved events for this station.",
                      className="text-muted small mb-0")

    rows = []
    for ev in saved:
        peak_str  = pd.Timestamp(ev["peak_time"]).strftime("%d %b %Y  %H:%M")
        tags_str  = ", ".join(ev.get("tags", []))
        qual      = ev.get("quality", "")
        qual_badge = dbc.Badge(
            qual, color={
                "good": "success", "uncertain": "warning", "poor": "danger"
            }.get(qual, "secondary"),
            className="small",
        ) if qual else ""
        exclude_badge = dbc.Badge("excl.", color="dark", className="small ms-1") \
            if ev.get("exclude") else ""
        custom_fields = ev.get("custom_fields", {})
        custom_tip = "  ·  ".join(f"{k}: {v}" for k, v in custom_fields.items()) \
            if custom_fields else ""

        rows.append(html.Tr([
            html.Td(peak_str,                                   className="small"),
            html.Td(f"{ev['peak_tg_m']:.3f} m",                className="small text-center"),
            html.Td(f"{ev['duration_h']} h",                   className="small text-center"),
            html.Td([qual_badge, exclude_badge],                className="small text-center"),
            html.Td(ev.get("note", ""),                        className="small text-muted",
                    style={"maxWidth": "160px", "overflow": "hidden",
                           "textOverflow": "ellipsis", "whiteSpace": "nowrap"}),
            html.Td(
                dbc.Button(
                    html.I(className="bi bi-search"),
                    id={"type": "ev-view-saved-btn", "index": ev["id"]},
                    size="sm", color="link", className="p-0", title="Zoom",
                ),
                className="text-center",
            ),
            html.Td(
                dbc.Button(
                    html.I(className="bi bi-pencil"),
                    id={"type": "ev-edit-saved-btn", "index": ev["id"]},
                    size="sm", color="link", className="p-0 text-primary",
                    title="Edit",
                ),
                className="text-center",
            ),
            html.Td(
                dbc.Button(
                    html.I(className="bi bi-trash"),
                    id={"type": "ev-del-btn", "index": ev["id"]},
                    size="sm", color="link", className="p-0 text-danger",
                    title="Delete",
                ),
                className="text-center",
            ),
        ], title=custom_tip))

    return dbc.Table(
        [
            html.Thead(html.Tr([
                html.Th("Peak time"),
                html.Th("Peak TG",  className="text-center"),
                html.Th("Duration", className="text-center"),
                html.Th("Quality",  className="text-center"),
                html.Th("Note"),
                html.Th("",         style={"width": "32px"}),
                html.Th("",         style={"width": "32px"}),
                html.Th("",         style={"width": "32px"}),
            ])),
            html.Tbody(rows),
        ],
        bordered=False, hover=True, size="sm", className="mb-0",
    )

"""Dash callbacks: wire map clicks, dropdown, date picker to plots."""
from __future__ import annotations

import pandas as pd
import dash_bootstrap_components as dbc
from dash import ALL, Input, Output, State, callback, ctx, html

from . import data_loader, figures


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
    """Show the lag slider only when MISO lag is selected."""
    if method == "miso":
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
)
def update_regression(station_id, start_date, end_date, reg_method, lag_value, remove_bias):
    """Compute and render regression figures when the Regression tab is active."""
    remove_bias = bool(remove_bias)
    empty_reg = figures.make_regression_plot({}, "")
    empty_acf = figures.make_acf_plot({}, "")
    empty_irf = figures.make_irf_plot({}, "")
    if not station_id:
        return empty_reg, empty_acf, empty_irf

    stations = data_loader.get_stations()
    name_map = {s["id"]: s["name"] for s in stations}
    station_name = name_map.get(station_id, station_id)

    df = data_loader.get_station_data(station_id, start_date, end_date)
    if df.empty:
        return empty_reg, empty_acf, empty_irf

    lag = lag_value if reg_method == "miso" else 0
    reg = figures.compute_regression(df, method=reg_method or "ols",
                                     lag=lag or 0, remove_bias=remove_bias)

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

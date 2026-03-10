"""Dash callbacks: wire map clicks, dropdown, date picker to plots."""
from __future__ import annotations

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


# ── Station + DateRange → plots + stats ──────────────────────────────────
@callback(
    Output("time-plot", "figure"),
    Output("psd-plot", "figure"),
    Output("stats-card", "children"),
    Input("station-dropdown", "value"),
    Input("compare-dropdown", "value"),
    Input("date-range", "start_date"),
    Input("date-range", "end_date"),
)
def update_plots(station_id, compare_ids, start_date, end_date):
    """Fetch data for selected station/period and regenerate plots."""
    if not station_id:
        empty = figures.make_time_plot(data_loader.get_dataframe().iloc[:0])
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
                continue  # skip if same as primary
            cdf = data_loader.get_station_data(cid, start_date, end_date)
            cname = name_map.get(cid, cid)
            compare_data.append((cdf, cname))

    # Build figures
    fig_time = figures.make_time_plot(df, station_name, compare_data=compare_data)
    fig_psd = figures.make_psd_plot(df, station_name=station_name, compare_data=compare_data)

    # Statistics
    stats = figures.compute_stats(df)
    stats_card = _build_stats_card(stats)

    return fig_time, fig_psd, stats_card


def _build_stats_card(stats: dict):
    """Build a Bootstrap card showing error statistics."""
    if stats["total"] == 0:
        return dbc.Alert("No data for this selection.", color="warning")

    rows = [
        ("Total timestamps", f"{stats['total']:,}"),
        ("Valid (matched)", f"{stats['valid']:,}  ({stats['pct_valid']}%)"),
        ("Bias (mean ε)", f"{stats['bias']:+.5f} m"),
        ("RMSE", f"{stats['rmse']:.5f} m"),
        ("MAE", f"{stats['mae']:.5f} m"),
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

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
    if not ctx.triggered_id or not any(n_clicks_list):
        return current_value
    return ctx.triggered_id["index"]


# ── Show / hide lag slider ───────────────────────────────────────────────
@callback(
    Output("lag-control", "style"),
    Input("reg-method", "value"),
)
def toggle_lag_control(method):
    if method == "miso":
        return {"display": "block"}
    return {"display": "none"}


# ── Station + DateRange + Regression → all plots + stats ─────────────────
@callback(
    Output("time-plot", "figure"),
    Output("psd-plot", "figure"),
    Output("stats-card", "children"),
    Output("regression-card", "children"),
    Output("regression-plot", "figure"),
    Output("acf-plot", "figure"),
    Input("station-dropdown", "value"),
    Input("compare-dropdown", "value"),
    Input("date-range", "start_date"),
    Input("date-range", "end_date"),
    Input("reg-method", "value"),
    Input("lag-slider", "value"),
)
def update_plots(station_id, compare_ids, start_date, end_date,
                 reg_method, lag_value):
    if not station_id:
        empty = figures.make_time_plot(data_loader.get_dataframe().iloc[:0])
        return empty, empty, "", "", empty, empty

    # Station name
    stations = data_loader.get_stations()
    name_map = {s["id"]: s["name"] for s in stations}
    station_name = name_map.get(station_id, station_id)

    # Primary data
    df = data_loader.get_station_data(station_id, start_date, end_date)

    # Comparison data
    compare_data = []
    if compare_ids:
        for cid in compare_ids:
            if cid == station_id:
                continue
            cdf = data_loader.get_station_data(cid, start_date, end_date)
            cname = name_map.get(cid, cid)
            compare_data.append((cdf, cname))

    # ── Overview plots ────────────────────────────────────────────
    fig_time = figures.make_time_plot(
        df, station_name, compare_data=compare_data
    )
    fig_psd = figures.make_psd_plot(
        df, station_name=station_name, compare_data=compare_data
    )
    stats = figures.compute_stats(df)

    # ── Regression ────────────────────────────────────────────────
    reg = figures.compute_regression(
        df,
        method=reg_method or "ols",
        lag=lag_value or 72,
    )
    fig_reg = figures.make_regression_plot(reg, station_name=station_name)
    fig_acf = figures.make_acf_plot(
        reg=reg, station_name=station_name, df=df
    )

    return (
        fig_time,
        fig_psd,
        _build_stats_card(stats),
        _build_regression_card(reg),
        fig_reg,
        fig_acf,
    )


# ═══════════════════════════════════════════════════════════════════════
# Card builders
# ═══════════════════════════════════════════════════════════════════════

def _build_stats_card(stats: dict):
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
        dbc.CardBody(
            [
                html.Table(
                    [
                        html.Tr(
                            [
                                html.Td(label, className="pe-3 text-muted small"),
                                html.Td(html.Strong(value), className="small"),
                            ]
                        )
                        for label, value in rows
                    ],
                    className="mb-0",
                )
            ]
        ),
        className="shadow-sm",
    )


def _build_regression_card(reg: dict):
    if not reg.get("ok"):
        return html.Span(
            "Insufficient data (need atmospheric columns).",
            className="text-muted small",
        )

    method_label = (
        "OLS"
        if reg["method"] == "ols"
        else f"MISO  L = {reg['lag']} h"
    )

    def _color(v):
        return "#C0392B" if v < 0 else "#1A6B9A"

    # Metrics row
    metrics = [
        ("R²", f"{reg['r2']:.3f}", None),
        ("RMSE", f"{reg['rmse'] * 100:.2f} cm", None),
        ("DW", f"{reg['dw']:.2f}", None),
        ("Bias", f"{reg['bias'] * 100:+.2f} cm", _color(reg["bias"])),
    ]
    metrics_row = dbc.Row(
        [
            dbc.Col(
                [
                    html.Div(lbl, className="text-muted text-center",
                             style={"fontSize": "10px"}),
                    html.Div(
                        val,
                        className="fw-bold text-center",
                        style={
                            "fontSize": "14px",
                            **({"color": clr} if clr else {}),
                        },
                    ),
                ],
                width=3,
            )
            for lbl, val, clr in metrics
        ],
        className="mb-2",
    )

    # β table
    coef_rows = [
        html.Tr(
            [
                html.Td(label, className="pe-2 text-muted small"),
                html.Td(
                    html.Span(
                        f"{v:+.4f} m/σ",
                        style={
                            "color": _color(v),
                            "fontWeight": "600",
                            "fontSize": "12px",
                        },
                    )
                ),
            ]
        )
        for label, v in reg["coefs"].items()
    ]

    lag_note = " (lag-0 only)" if reg["method"] == "miso" else ""

    return html.Div(
        [
            html.Div(
                html.Span(method_label, className="badge bg-accent"),
                className="text-center mb-2",
            ),
            metrics_row,
            html.Hr(className="my-1"),
            html.Div(
                f"Standardised β{lag_note}",
                className="text-muted mb-1",
                style={"fontSize": "11px"},
            ),
            html.Table(coef_rows, className="mb-0 w-100"),
            html.Div(
                f"n = {reg['n']:,}",
                className="text-muted mt-1",
                style={"fontSize": "10px", "textAlign": "right"},
            ),
        ]
    )

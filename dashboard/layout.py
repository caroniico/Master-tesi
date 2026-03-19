"""Page layout for the HDM-TG error dashboard."""
from __future__ import annotations

import dash_bootstrap_components as dbc
import dash_leaflet as dl
from dash import dcc, html

from . import data_loader

# ── Constants ────────────────────────────────────────────────────────────
MAP_CENTER = [55.8, 11.5]
MAP_ZOOM = 7

_ACCENT = "#1B9AAA"
_DARK = "#1A1A2E"


# ═══════════════════════════════════════════════════════════════════════
# Sidebar helpers
# ═══════════════════════════════════════════════════════════════════════

def _build_station_markers() -> list:
    stations = data_loader.get_stations()
    df = data_loader.get_dataframe()
    ids_with_data = set(df["station_id"].unique())
    markers = []
    for st in stations:
        has_data = st["id"] in ids_with_data
        markers.append(
            dl.CircleMarker(
                center=[st["lat"], st["lon"]],
                radius=8 if has_data else 5,
                color=_ACCENT if has_data else "#aaa",
                fillColor=_ACCENT if has_data else "#ccc",
                fillOpacity=0.85 if has_data else 0.50,
                weight=2,
                children=[
                    dl.Tooltip(
                        f"{st['name']}" + ("" if has_data else " (no data)")
                    ),
                    dl.Popup(f"{st['name']}  —  ID {st['id']}"),
                ],
                id={"type": "station-marker", "index": st["id"]},
            )
        )
    return markers


def _build_station_dropdown() -> dcc.Dropdown:
    stations = data_loader.get_stations()
    df = data_loader.get_dataframe()
    ids_with_data = set(df["station_id"].unique())
    options = [
        {"label": st["name"], "value": st["id"]}
        for st in stations
        if st["id"] in ids_with_data
    ]
    return dcc.Dropdown(
        id="station-dropdown",
        options=options,
        value=options[0]["value"] if options else None,
        placeholder="Select a tide gauge…",
        clearable=False,
        className="mb-2",
        style={"fontSize": "13px"},
    )


def _build_compare_dropdown() -> dcc.Dropdown:
    stations = data_loader.get_stations()
    df = data_loader.get_dataframe()
    ids_with_data = set(df["station_id"].unique())
    options = [
        {"label": st["name"], "value": st["id"]}
        for st in stations
        if st["id"] in ids_with_data
    ]
    return dcc.Dropdown(
        id="compare-dropdown",
        options=options,
        value=[],
        multi=True,
        placeholder="Add stations to compare…",
        clearable=True,
        className="mb-2",
        style={"fontSize": "13px"},
    )


def _build_date_picker() -> dcc.DatePickerRange:
    t_min, t_max = data_loader.get_time_range()
    return dcc.DatePickerRange(
        id="date-range",
        min_date_allowed=t_min,
        max_date_allowed=t_max,
        start_date=t_min,
        end_date=t_max,
        display_format="YYYY-MM-DD",
        className="mb-2",
        style={"fontSize": "12px"},
    )


def _brand_header() -> dbc.Navbar:
    return dbc.Navbar(
        dbc.Container(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            html.Div(
                                [
                                    html.I(
                                        className="bi bi-water me-2",
                                        style={"fontSize": "1.4rem"},
                                    ),
                                    html.Span(
                                        "HDM Error Analysis",
                                        className="fw-bold fs-5",
                                    ),
                                ],
                                className="d-flex align-items-center",
                            ),
                            width="auto",
                        ),
                        dbc.Col(
                            html.Span(
                                "ε(t) = FORCOAST − TG obs  ·  Danish Waters",
                                className="text-light opacity-75 small",
                            ),
                            className="d-flex align-items-center",
                        ),
                    ],
                    align="center",
                    className="g-0 w-100",
                ),
            ],
            fluid=True,
        ),
        color=_DARK,
        dark=True,
        className="mb-0 shadow-sm",
        style={"minHeight": "48px"},
    )


# ═══════════════════════════════════════════════════════════════════════
# Reusable sidebar card
# ═══════════════════════════════════════════════════════════════════════

def _card(icon: str, title: str, body_children) -> dbc.Card:
    header = dbc.CardHeader(
        html.Span([html.I(className=f"bi bi-{icon} me-2"), title]),
        className="fw-semibold small py-2",
    )
    body = dbc.CardBody(body_children, className="p-2")
    return dbc.Card([header, body], className="shadow-sm mb-3")


# ═══════════════════════════════════════════════════════════════════════
# Main layout
# ═══════════════════════════════════════════════════════════════════════

def build_layout() -> dbc.Container:
    return html.Div(
        [
            _brand_header(),
            dbc.Container(
                fluid=True,
                className="px-3 pt-3 pb-4",
                children=[
                    dbc.Row(
                        [
                            # ─── Left sidebar ────────────────────
                            dbc.Col(
                                [
                                    _card("geo-alt-fill", "Station",
                                          [_build_station_dropdown()]),
                                    _card("layers", "Compare",
                                          [_build_compare_dropdown()]),
                                    _card("calendar3", "Time Period",
                                          [_build_date_picker()]),

                                    # ── Regression settings ──────
                                    _card(
                                        "sliders",
                                        "Regression",
                                        [
                                            html.Label(
                                                "Method",
                                                className="text-muted small mb-1",
                                            ),
                                            dbc.RadioItems(
                                                id="reg-method",
                                                options=[
                                                    {"label": "OLS (instant)",
                                                     "value": "ols"},
                                                    {"label": "MISO Lag",
                                                     "value": "miso"},
                                                ],
                                                value="ols",
                                                inline=True,
                                                className="mb-2",
                                                input_class_name="me-1",
                                                label_class_name="small me-3",
                                            ),
                                            html.Div(
                                                id="lag-control",
                                                children=[
                                                    html.Label(
                                                        "Max lag L [hours]",
                                                        className="text-muted small mb-1",
                                                    ),
                                                    dcc.Slider(
                                                        id="lag-slider",
                                                        min=12,
                                                        max=240,
                                                        step=12,
                                                        value=72,
                                                        marks={
                                                            24: "24h",
                                                            48: "2d",
                                                            72: "3d",
                                                            120: "5d",
                                                            168: "7d",
                                                            240: "10d",
                                                        },
                                                        tooltip={
                                                            "placement": "bottom",
                                                            "always_visible": True,
                                                        },
                                                    ),
                                                ],
                                                style={"display": "none"},
                                            ),
                                        ],
                                    ),

                                    # ── Map ──────────────────────
                                    _card(
                                        "map",
                                        "Station Map",
                                        [
                                            dl.Map(
                                                id="station-map",
                                                center=MAP_CENTER,
                                                zoom=MAP_ZOOM,
                                                children=[
                                                    dl.TileLayer(
                                                        url="https://cartodb-basemaps-{s}.global.ssl.fastly.net/light_all/{z}/{x}/{y}{r}.png",
                                                        attribution='© <a href="https://carto.com/">CARTO</a>',
                                                    ),
                                                    dl.LayerGroup(
                                                        id="station-markers",
                                                        children=_build_station_markers(),
                                                    ),
                                                ],
                                                style={
                                                    "height": "280px",
                                                    "borderRadius": "4px",
                                                },
                                            ),
                                        ],
                                    ),

                                    # ── Stats cards ──────────────
                                    _card("bar-chart-line", "Error Statistics",
                                          [html.Div(id="stats-card")]),
                                    _card("wind", "Regression  ε ~ Atmo",
                                          [html.Div(id="regression-card")]),
                                ],
                                lg=3,
                                md=4,
                                sm=12,
                                className="pe-lg-3",
                            ),

                            # ─── Right: tabbed plots ─────────────
                            dbc.Col(
                                [
                                    dbc.Tabs(
                                        id="plot-tabs",
                                        active_tab="overview",
                                        className="mb-3 nav-tabs-custom",
                                        children=[
                                            # ── Tab 1: Overview ──
                                            dbc.Tab(
                                                label="Overview",
                                                tab_id="overview",
                                                children=[
                                                    dbc.Card(
                                                        dbc.CardBody(
                                                            dcc.Loading(
                                                                type="circle",
                                                                color=_ACCENT,
                                                                children=[
                                                                    dcc.Graph(
                                                                        id="time-plot",
                                                                        config={
                                                                            "displayModeBar": True,
                                                                            "displaylogo": False,
                                                                        },
                                                                        style={"height": "480px"},
                                                                    ),
                                                                ],
                                                            ),
                                                            className="p-2",
                                                        ),
                                                        className="shadow-sm mb-3 mt-3",
                                                    ),
                                                    dbc.Card(
                                                        dbc.CardBody(
                                                            dcc.Loading(
                                                                type="circle",
                                                                color=_ACCENT,
                                                                children=[
                                                                    dcc.Graph(
                                                                        id="psd-plot",
                                                                        config={
                                                                            "displayModeBar": True,
                                                                            "displaylogo": False,
                                                                        },
                                                                        style={"height": "340px"},
                                                                    ),
                                                                ],
                                                            ),
                                                            className="p-2",
                                                        ),
                                                        className="shadow-sm mb-3",
                                                    ),
                                                ],
                                            ),
                                            # ── Tab 2: Regression ─
                                            dbc.Tab(
                                                label="Regression",
                                                tab_id="regression",
                                                children=[
                                                    dbc.Card(
                                                        dbc.CardBody(
                                                            dcc.Loading(
                                                                type="circle",
                                                                color=_ACCENT,
                                                                children=[
                                                                    dcc.Graph(
                                                                        id="regression-plot",
                                                                        config={
                                                                            "displayModeBar": True,
                                                                            "displaylogo": False,
                                                                        },
                                                                    ),
                                                                ],
                                                            ),
                                                            className="p-2",
                                                        ),
                                                        className="shadow-sm mb-3 mt-3",
                                                    ),
                                                    dbc.Card(
                                                        dbc.CardBody(
                                                            dcc.Loading(
                                                                type="circle",
                                                                color=_ACCENT,
                                                                children=[
                                                                    dcc.Graph(
                                                                        id="acf-plot",
                                                                        config={
                                                                            "displayModeBar": True,
                                                                            "displaylogo": False,
                                                                        },
                                                                        style={"height": "320px"},
                                                                    ),
                                                                ],
                                                            ),
                                                            className="p-2",
                                                        ),
                                                        className="shadow-sm",
                                                    ),
                                                ],
                                            ),
                                        ],
                                    ),
                                ],
                                lg=9,
                                md=8,
                                sm=12,
                            ),
                        ]
                    ),
                ],
            ),
        ]
    )

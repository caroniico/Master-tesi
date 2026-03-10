"""Page layout for the HDM-TG error dashboard."""
from __future__ import annotations

import dash_bootstrap_components as dbc
import dash_leaflet as dl
from dash import dcc, html

from . import data_loader

# ── Constants ────────────────────────────────────────────────────────────
MAP_CENTER = [55.8, 11.5]
MAP_ZOOM = 7

# Colour palette
_ACCENT = "#1B9AAA"      # teal accent
_DARK = "#1A1A2E"        # sidebar / header bg
_SIDEBAR_BG = "#F5F7FA"  # light grey sidebar


def _build_station_markers() -> list:
    """Create CircleMarker components for each station (only those with data)."""
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
                        f"{st['name']}" + ("" if has_data else " (no data)"),
                    ),
                    dl.Popup(f"{st['name']}  —  ID {st['id']}"),
                ],
                id={"type": "station-marker", "index": st["id"]},
            )
        )
    return markers


def _build_station_dropdown() -> dcc.Dropdown:
    """Build a dropdown with only stations that have data."""
    stations = data_loader.get_stations()
    df = data_loader.get_dataframe()
    ids_with_data = set(df["station_id"].unique())
    options = [
        {"label": st["name"], "value": st["id"]}
        for st in stations if st["id"] in ids_with_data
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
    """Multi-select dropdown to overlay other stations for comparison."""
    stations = data_loader.get_stations()
    df = data_loader.get_dataframe()
    ids_with_data = set(df["station_id"].unique())
    options = [
        {"label": st["name"], "value": st["id"]}
        for st in stations if st["id"] in ids_with_data
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
    """Build a date range picker spanning the dataset."""
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
    """Top navigation bar."""
    return dbc.Navbar(
        dbc.Container([
            dbc.Row([
                dbc.Col(
                    html.Div([
                        html.I(className="bi bi-water me-2",
                               style={"fontSize": "1.4rem"}),
                        html.Span("HDM Error Analysis",
                                  className="fw-bold fs-5"),
                    ], className="d-flex align-items-center"),
                    width="auto",
                ),
                dbc.Col(
                    html.Span(
                        "ε(t) = DKSS − TG obs  ·  Danish Waters",
                        className="text-light opacity-75 small",
                    ),
                    className="d-flex align-items-center",
                ),
            ], align="center", className="g-0 w-100"),
        ], fluid=True),
        color=_DARK,
        dark=True,
        className="mb-0 shadow-sm",
        style={"minHeight": "48px"},
    )


def build_layout() -> dbc.Container:
    """Assemble the full page layout."""
    return html.Div([
        # ── Top bar ──────────────────────────────────────────────
        _brand_header(),

        # ── Body ─────────────────────────────────────────────────
        dbc.Container(
            fluid=True,
            className="px-3 pt-3 pb-4",
            children=[
                dbc.Row([

                    # ─── Left sidebar ────────────────────────────
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader(
                                html.Span([
                                    html.I(className="bi bi-geo-alt-fill me-2"),
                                    "Station",
                                ]),
                                className="fw-semibold small py-2",
                            ),
                            dbc.CardBody([
                                _build_station_dropdown(),
                            ], className="p-2"),
                        ], className="shadow-sm mb-3"),

                        dbc.Card([
                            dbc.CardHeader(
                                html.Span([
                                    html.I(className="bi bi-layers me-2"),
                                    "Compare",
                                ]),
                                className="fw-semibold small py-2",
                            ),
                            dbc.CardBody([
                                _build_compare_dropdown(),
                            ], className="p-2"),
                        ], className="shadow-sm mb-3"),

                        dbc.Card([
                            dbc.CardHeader(
                                html.Span([
                                    html.I(className="bi bi-calendar3 me-2"),
                                    "Time Period",
                                ]),
                                className="fw-semibold small py-2",
                            ),
                            dbc.CardBody([
                                _build_date_picker(),
                            ], className="p-2"),
                        ], className="shadow-sm mb-3"),

                        dbc.Card([
                            dbc.CardHeader(
                                html.Span([
                                    html.I(className="bi bi-map me-2"),
                                    "Station Map",
                                ]),
                                className="fw-semibold small py-2",
                            ),
                            dbc.CardBody([
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
                                    style={"height": "340px", "borderRadius": "4px"},
                                ),
                            ], className="p-2"),
                        ], className="shadow-sm mb-3"),

                        dbc.Card([
                            dbc.CardHeader(
                                html.Span([
                                    html.I(className="bi bi-bar-chart-line me-2"),
                                    "Error Statistics",
                                ]),
                                className="fw-semibold small py-2",
                            ),
                            dbc.CardBody(
                                html.Div(id="stats-card"),
                                className="p-2",
                            ),
                        ], className="shadow-sm mb-3"),

                    ], lg=3, md=4, sm=12, className="pe-lg-3"),

                    # ─── Right: plots ────────────────────────────
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                dcc.Loading(
                                    type="circle",
                                    color=_ACCENT,
                                    children=[
                                        dcc.Graph(
                                            id="time-plot",
                                            config={"displayModeBar": True,
                                                    "displaylogo": False},
                                            style={"height": "460px"},
                                        ),
                                    ],
                                ),
                            ], className="p-2"),
                        ], className="shadow-sm mb-3"),

                        dbc.Card([
                            dbc.CardBody([
                                dcc.Loading(
                                    type="circle",
                                    color=_ACCENT,
                                    children=[
                                        dcc.Graph(
                                            id="psd-plot",
                                            config={"displayModeBar": True,
                                                    "displaylogo": False},
                                            style={"height": "340px"},
                                        ),
                                    ],
                                ),
                            ], className="p-2"),
                        ], className="shadow-sm"),

                    ], lg=9, md=8, sm=12),
                ]),
            ],
        ),
    ])

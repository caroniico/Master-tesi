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
                                "ε(t) = Model − TG obs  ·  Danish Waters",
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
                                                    {"label": "OLS no lag",
                                                     "value": "ols"},
                                                    {"label": "OLS lag",
                                                     "value": "miso"},
                                                    {"label": "Ridge no lag",
                                                     "value": "ridge"},
                                                    {"label": "Ridge lag",
                                                     "value": "ridge-miso"},
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
                                            html.Hr(className="my-2"),
                                            dbc.Switch(
                                                id="include-tide-toggle",
                                                label="Include DTU10 tides",
                                                value=False,
                                                className="small",
                                            ),
                                            html.P(
                                                "Aggiunge il segnale di marea DTU10/GOT4.7 "
                                                "come feature nella regressione.",
                                                className="text-muted",
                                                style={"fontSize": "11px",
                                                       "lineHeight": "1.3",
                                                       "marginTop": "2px"},
                                            ),
                                        ],
                                    ),

                                    # ── Bias correction toggle ────
                                    _card(
                                        "eraser-fill",
                                        "Bias Correction",
                                        [
                                            html.P(
                                                "Rimuove la media temporale di "
                                                "(Model − TG) calcolata sull'intero "
                                                "record di ogni stazione.",
                                                className="text-muted small mb-2",
                                                style={"lineHeight": "1.35"},
                                            ),
                                            dbc.Switch(
                                                id="remove-bias-toggle",
                                                label="Rimuovi bias",
                                                value=False,
                                                className="small",
                                            ),
                                        ],
                                    ),

                                    # ── Surge threshold (Event Library) ──
                                    _card(
                                        "exclamation-triangle-fill",
                                        "Storm Surge",
                                        [
                                            html.Label(
                                                "Threshold TG [m]",
                                                className="text-muted small mb-1",
                                            ),
                                            dcc.Slider(
                                                id="surge-thresh-slider",
                                                min=0.4, max=1.5, step=0.05,
                                                value=0.80,
                                                marks={
                                                    0.5: "0.5",
                                                    0.8: "0.8",
                                                    1.0: "1.0",
                                                    1.2: "1.2",
                                                    1.5: "1.5",
                                                },
                                                tooltip={
                                                    "placement": "bottom",
                                                    "always_visible": True,
                                                },
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
                                            # ── Tab 2: Error Statistics ─
                                            dbc.Tab(
                                                label="Error Statistics",
                                                tab_id="error-stats",
                                                children=[
                                                    dbc.Card(
                                                        dbc.CardBody(
                                                            dcc.Loading(
                                                                type="circle",
                                                                color=_ACCENT,
                                                                children=[
                                                                    dcc.Graph(
                                                                        id="error-stats-plot",
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
                                                ],
                                            ),
                                            # ── Tab 3: Regression ─
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
                                                        className="shadow-sm mb-3",
                                                    ),
                                                    dbc.Card(
                                                        dbc.CardBody(
                                                            dcc.Loading(
                                                                type="circle",
                                                                color=_ACCENT,
                                                                children=[
                                                                    dcc.Graph(
                                                                        id="irf-plot",
                                                                        config={
                                                                            "displayModeBar": True,
                                                                            "displaylogo": False,
                                                                        },
                                                                        style={"height": "280px"},
                                                                    ),
                                                                ],
                                                            ),
                                                            className="p-2",
                                                        ),
                                                        className="shadow-sm",
                                                    ),
                                                ],
                                            ),
                                            # ── Tab 4: Event Library ─
                                            dbc.Tab(
                                                label="Event Library",
                                                tab_id="event-library",
                                                children=[
                                                    # Overview plot
                                                    dbc.Card(
                                                        dbc.CardBody(
                                                            dcc.Loading(
                                                                type="circle",
                                                                color=_ACCENT,
                                                                children=[dcc.Graph(
                                                                    id="ev-overview-plot",
                                                                    config={"displayModeBar": True, "displaylogo": False},
                                                                    style={"height": "260px"},
                                                                )],
                                                            ),
                                                            className="p-2",
                                                        ),
                                                        className="shadow-sm mb-3 mt-3",
                                                    ),
                                                    # Detected events table
                                                    dbc.Card(
                                                        dbc.CardBody([
                                                            dbc.Row([
                                                                dbc.Col(
                                                                    html.H6(
                                                                        [html.I(className="bi bi-search me-2"),
                                                                         "Detected Events"],
                                                                        className="mb-2",
                                                                    ),
                                                                    width=True,
                                                                ),
                                                                dbc.Col(
                                                                    dbc.Button(
                                                                        [html.I(className="bi bi-lightning-charge-fill me-1"),
                                                                         "Batch Save All"],
                                                                        id="ev-batch-save-btn",
                                                                        size="sm",
                                                                        color="warning",
                                                                        title="Detect all events and run all 4 regression methods (±15 days window) — save to library",
                                                                    ),
                                                                    width="auto",
                                                                ),
                                                            ], align="center", className="mb-2"),
                                                            dbc.Alert(
                                                                id="ev-batch-feedback",
                                                                is_open=False,
                                                                duration=6000,
                                                                color="info",
                                                                className="mb-2 py-2 small",
                                                            ),
                                                            html.Div(id="ev-detected-table"),
                                                            dbc.Alert(
                                                                id="ev-save-feedback",
                                                                is_open=False,
                                                                duration=3000,
                                                                color="success",
                                                                className="mt-2 mb-0 py-2 small",
                                                            ),
                                                        ]),
                                                        className="shadow-sm mb-3",
                                                    ),
                                                    # Saved library
                                                    dbc.Card(
                                                        dbc.CardBody([
                                                            dbc.Row([
                                                                dbc.Col(
                                                                    html.H6(
                                                                        [html.I(className="bi bi-bookmark-check me-2"),
                                                                         "Saved Library"],
                                                                        className="mb-2",
                                                                    ),
                                                                    width=True,
                                                                ),
                                                                dbc.Col(
                                                                    dbc.Button(
                                                                        [html.I(className="bi bi-download me-1"),
                                                                         "Export JSON"],
                                                                        id="ev-export-btn",
                                                                        size="sm",
                                                                        color="secondary",
                                                                        outline=True,
                                                                    ),
                                                                    width="auto",
                                                                ),
                                                            ], align="center", className="mb-2"),
                                                            html.Div(id="ev-saved-table"),
                                                            dcc.Download(id="ev-download"),
                                                        ]),
                                                        className="shadow-sm mb-3",
                                                    ),
                                                    # Event zoom plot
                                                    dbc.Card(
                                                        dbc.CardBody(
                                                            dcc.Loading(
                                                                type="circle",
                                                                color=_ACCENT,
                                                                children=[dcc.Graph(
                                                                    id="ev-zoom-plot",
                                                                    config={"displayModeBar": True, "displaylogo": False},
                                                                    style={"height": "420px"},
                                                                )],
                                                            ),
                                                            className="p-2",
                                                        ),
                                                        className="shadow-sm",
                                                    ),
                                                    # ── Hidden stores ────────────────────
                                                    dcc.Store(id="ev-selected-store"),
                                                    dcc.Store(id="ev-edit-store"),
                                                    # ── Save/Edit modal ──────────────────
                                                    dbc.Modal([
                                                        dbc.ModalHeader(
                                                            dbc.ModalTitle(
                                                                [html.I(className="bi bi-bookmark-plus me-2"),
                                                                 html.Span(id="ev-modal-title")],
                                                            ),
                                                            close_button=True,
                                                        ),
                                                        dbc.ModalBody([
                                                            # ── Read-only summary ─────────
                                                            dbc.Row([
                                                                dbc.Col([
                                                                    dbc.Label("Peak time", className="small text-muted mb-0"),
                                                                    html.Div(id="ev-modal-peak-time",
                                                                             className="fw-semibold"),
                                                                ], width=4),
                                                                dbc.Col([
                                                                    dbc.Label("Peak TG [m]", className="small text-muted mb-0"),
                                                                    html.Div(id="ev-modal-peak-tg",
                                                                             className="fw-semibold"),
                                                                ], width=4),
                                                                dbc.Col([
                                                                    dbc.Label("Duration [h]", className="small text-muted mb-0"),
                                                                    html.Div(id="ev-modal-duration",
                                                                             className="fw-semibold"),
                                                                ], width=4),
                                                            ], className="mb-3 p-2 rounded",
                                                              style={"background": "#f8f9fa"}),
                                                            html.Hr(className="my-2"),
                                                            # ── Editable standard fields ──
                                                            dbc.Row([
                                                                dbc.Col([
                                                                    dbc.Label("Note", className="small"),
                                                                    dbc.Textarea(
                                                                        id="ev-field-note",
                                                                        rows=2,
                                                                        placeholder="Free text note…",
                                                                        className="form-control form-control-sm",
                                                                    ),
                                                                ], width=12),
                                                            ], className="mb-2"),
                                                            dbc.Row([
                                                                dbc.Col([
                                                                    dbc.Label("Quality", className="small"),
                                                                    dbc.Select(
                                                                        id="ev-field-quality",
                                                                        options=[
                                                                            {"label": "—", "value": ""},
                                                                            {"label": "Good", "value": "good"},
                                                                            {"label": "Uncertain", "value": "uncertain"},
                                                                            {"label": "Poor", "value": "poor"},
                                                                        ],
                                                                        value="",
                                                                        className="form-select form-select-sm",
                                                                    ),
                                                                ], width=4),
                                                                dbc.Col([
                                                                    dbc.Label("Wind direction", className="small"),
                                                                    dbc.Input(
                                                                        id="ev-field-wind-dir",
                                                                        placeholder="e.g. NW",
                                                                        type="text",
                                                                        size="sm",
                                                                    ),
                                                                ], width=4),
                                                                dbc.Col([
                                                                    dbc.Label("Min SLP [hPa]", className="small"),
                                                                    dbc.Input(
                                                                        id="ev-field-pressure",
                                                                        placeholder="e.g. 985",
                                                                        type="number",
                                                                        size="sm",
                                                                    ),
                                                                ], width=4),
                                                            ], className="mb-2"),
                                                            dbc.Row([
                                                                dbc.Col([
                                                                    dbc.Label("Max |ε| [m]", className="small"),
                                                                    dbc.Input(
                                                                        id="ev-field-max-error",
                                                                        placeholder="auto or manual",
                                                                        type="number",
                                                                        step=0.001,
                                                                        size="sm",
                                                                    ),
                                                                ], width=4),
                                                                dbc.Col([
                                                                    dbc.Label("Tags (comma-separated)", className="small"),
                                                                    dbc.Input(
                                                                        id="ev-field-tags",
                                                                        placeholder="e.g. westerly, long",
                                                                        type="text",
                                                                        size="sm",
                                                                    ),
                                                                ], width=5),
                                                                dbc.Col([
                                                                    dbc.Label("Exclude from analysis", className="small"),
                                                                    dbc.Checkbox(
                                                                        id="ev-field-exclude",
                                                                        value=False,
                                                                        className="mt-1",
                                                                    ),
                                                                ], width=3),
                                                            ], className="mb-3"),
                                                            html.Hr(className="my-2"),
                                                            # ── Custom key-value fields ───
                                                            dbc.Row([
                                                                dbc.Col(
                                                                    html.H6([
                                                                        html.I(className="bi bi-plus-circle me-1"),
                                                                        "Custom Fields",
                                                                    ], className="small fw-semibold mb-1"),
                                                                    width=True,
                                                                ),
                                                                dbc.Col(
                                                                    dbc.Button(
                                                                        [html.I(className="bi bi-plus me-1"), "Add field"],
                                                                        id="ev-add-custom-field-btn",
                                                                        size="sm",
                                                                        color="secondary",
                                                                        outline=True,
                                                                    ),
                                                                    width="auto",
                                                                ),
                                                            ], align="center", className="mb-1"),
                                                            html.Div(
                                                                id="ev-custom-fields-container",
                                                                className="mb-2",
                                                            ),
                                                        ]),
                                                        dbc.ModalFooter([
                                                            dbc.Button(
                                                                "Cancel",
                                                                id="ev-modal-cancel-btn",
                                                                color="secondary",
                                                                outline=True,
                                                                size="sm",
                                                                className="me-2",
                                                            ),
                                                            dbc.Button(
                                                                [html.I(className="bi bi-floppy me-1"),
                                                                 "Save event"],
                                                                id="ev-modal-save-btn",
                                                                color="primary",
                                                                size="sm",
                                                            ),
                                                        ]),
                                                    ],
                                                        id="ev-edit-modal",
                                                        is_open=False,
                                                        size="lg",
                                                        backdrop="static",
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

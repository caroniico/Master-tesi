"""HDM-TG Error Dashboard — Dash entry point.

Run:  python -m dashboard.app
  or: python dashboard/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import dash_bootstrap_components as dbc
from dash import Dash

# Ensure project root is on sys.path when run as a script
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from dashboard.layout import build_layout
import dashboard.callbacks  # noqa: F401  — registers callbacks on import

app = Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.FLATLY,
        dbc.icons.BOOTSTRAP,  # Bootstrap Icons
    ],
    title="HDM Error Dashboard",
    update_title="Loading…",
    suppress_callback_exceptions=True,
    meta_tags=[{"name": "viewport",
                "content": "width=device-width, initial-scale=1"}],
)

app.layout = build_layout()
server = app.server  # for production WSGI

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=8050)

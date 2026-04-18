"""Quick runtime test of the event library callback logic."""
import sys, traceback
sys.path.insert(0, '/Users/nicolocaron/Documents/GitHub/Master-tesi')

import pandas as pd
from dashboard import data_loader, event_library as evlib

# 1. Load a real station
stations = data_loader.get_stations()
sid = stations[0]["id"]
sname = stations[0]["name"]
print(f"Station: {sname} ({sid})")

# 2. Get data
df = data_loader.get_station_data(sid)
print(f"Data shape: {df.shape}, cols: {list(df.columns)}")

# 3. Detect events
events_df = evlib.detect_events(df, thresh_m=0.80)
print(f"Events detected: {len(events_df)}")
if not events_df.empty:
    print(events_df[["peak_time","peak_tg_m","duration_h"]].head(3))

# 4. Overview plot
fig = evlib.make_events_overview_plot(df, events_df, station_name=sname, thresh_m=0.80)
print(f"Overview fig type: {type(fig).__name__}")

# 5. Zoom plot (first event)
if not events_df.empty:
    pt = events_df.iloc[0]["peak_time"]
    fig2 = evlib.make_event_zoom_plot(df, pt, station_name=sname)
    print(f"Zoom fig type: {type(fig2).__name__}")

print("\nAll tests PASSED")

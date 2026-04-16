import traceback, sys
sys.path.insert(0, '/Users/nicolocaron/Documents/GitHub/Master-tesi')
try:
    import dashboard.callbacks
    print("callbacks OK")
except Exception:
    traceback.print_exc()

try:
    import dashboard.layout
    print("layout OK")
except Exception:
    traceback.print_exc()

try:
    import dashboard.app
    print("app OK")
except Exception:
    traceback.print_exc()

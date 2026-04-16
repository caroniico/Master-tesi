import traceback, sys
sys.path.insert(0, '.')
for mod in [
    'dashboard.event_library',
    'dashboard.data_loader',
    'dashboard.figures',
    'dashboard.layout',
    'dashboard.callbacks',
    'dashboard.app',
]:
    try:
        __import__(mod)
        print(mod, 'OK')
    except Exception:
        print(f'\n===== ERRORE in {mod} =====')
        traceback.print_exc()
        sys.exit(1)
print('\nTutti i moduli OK')

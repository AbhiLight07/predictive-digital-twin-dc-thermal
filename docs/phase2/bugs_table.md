# Bug Fix Summary

| Bug | Location | Before | After | Fixed? |
|-----|----------|--------|-------|--------|
| Split A boundary overlap | `src/ttwin/splits.py` | test%=31.7% | test%=20.0% | ✅ |
| Feature dedup (S4 dim) | feature engineering pipeline | 57 cols (crash) | 56 cols (OK) | ✅ |
| Host trailing spaces | CSV ingestion / host join | mismatch (1=error) | clean (0=ok) | ✅ |
| Conformal calib set | uncertainty calibration | train rows (wrong) | val rows only (correct) | ✅ |

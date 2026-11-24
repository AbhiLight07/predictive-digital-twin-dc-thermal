# Paper Number & Claim Audit (`reports/research_paper_draft.md`)

Audit of every quantitative figure and domain claim in `reports/research_paper_draft.md` against verified workspace files.

---

## 1. Changelog of Corrections Applied

| Location | Stale / Inaccurate Claim in Prior Draft | Corrected Value / Phrasing | Authoritative Source |
|---|---|---|---|
| Abstract, §2, §4.5 | Unverified "Dell R840 (4-socket)" / "Dell C6320 (2-socket high-density)" | Mandatory wording: *"as reported in the dataset paper (Ilager et al., UCC 2023): dual-CPU hosts; capacities as in the trace columns; other hardware specifications are unverified"* | `docs/decisions.md`, `PROJECT_BRIEF.md` |
| Abstract, §4.4, §6 | Unverified "seasonal weather variations" causal attribution | Replaced with **operational drift** (ambient weather/HVAC not recorded; cause untested) | `docs/limitations.md` §2 |
| §2 Quality Gate | `20 nodes` quality gate threshold; `2,748,039` raw D1 rows | `40` servers quality gate (`76` qualifying in D1); `2,748,525` raw D1 rows (`4,029,549` total) | `docs/data_audit_report.md` L15–L26 |
| §3 Diagram | Labels `E5 Conformal Uncertainty` & `E4 Few-Shot Adaptation` | Corrected to `E2 Cold-Start Adaptation` and `E4 Conformal Uncertainty` | `PROJECT_BRIEF.md` §4.5 |
| §4.1 (RQ1) | Old Split A ablation (`S1 0.890°C`, `S4 0.955°C`, `S2 5.280°C`) | v3 gap-corrected ablation (`552,672` identical rows): `S0=0.8408`, `S1=0.8360`, `S4=0.8381°C` + paired bootstrap D1 `+0.59%`, D2 `+2.83%` (server CIs crossing zero) + leakage (`17.60%` S2, `18.54%` S3) | `docs/results_v3.md` §1, §2, §6, §7 |
| §4.2 (RQ2) | Old Split B `0.527°C` / Split C `0.408°C` (16 servers) | v3 5-fold GroupKFold ($N=81$): Split B S0 `0.435/0.918/1.817°C`, S1 `0.436/0.924/1.828°C`; Split C (5 seeds) `0.456/0.940/1.710°C` | `docs/results_v3.md` §3 |
| §4.3 (RQ3) | Old Table 2 (`0.424°C` zero-shot vs `5.811°C` scratch) | v3 fleet planning ($N=81$): Zero-shot `5.03°C` vs Fleet Mean `8.87°C` (`43.2%` reduction), Boosted `4.87°C` (168h), Residual `5.52 → 5.02°C` | `docs/results_v3.md` §4 |
| §4.4 (RQ4) | Old Split B `96.7%` vs Split A `85.9%` conformal table | v2 Split C (`experiments/v2_uncertainty/`): Constant `90.33%` (`4.204°C`), Normalized `86.44%` (`3.502°C`), CQR `91.19%` (`3.679°C`, Spearman `0.470`), conditional + per-server `84.7–96.9%` (`40.0% < 90%`) | `docs/results_v2.md` §8 Tables 7–9 |
| §4.5 (RQ5) | Old C2 S1/monitoring (`0.812` persist, `+0.232°C` penalty) | v3 Cross-Cluster C1→C2 planning: Zero-shot `6.85°C` (`[2.92, 13.47]`), Residual `3.79 → 2.92°C` (72h), Boosted `6.05°C` (168h) | `docs/results_v3.md` §5 |
| §5 Twin Prototype | Claimed only Streamlit dashboard | Added verified `models/twin/` CQR metrics, FastAPI + 3D validation stats, and 25-window synthetic-fault alert results; noted legacy `app/streamlit_app.py` | `models/twin/{d1,d2}_metadata.json`, `docs/twin3d_validation.md`, `docs/twin_alert_evaluation.md` |

---

## 2. Verified Number Inventory

| Claim / Metric | Value in Paper | Source File & Reference |
|---|---|---|
| Raw row counts (D1 / D2) | `2,748,525` / `1,281,024` (~4.03M) | `docs/data_audit_report.md` L15 |
| Server counts & gate | `81` D1 (`76` gate-pass, `40` threshold) / `64` D2 | `docs/data_audit_report.md` L16–L26 |
| Split A D1/D2 1h bootstrap skill | D1 `0.8277` vs `0.8326` (`+0.59%`, CI `[-0.38%, +0.17%]`); D2 `0.7814` vs `0.8042` (`+2.83%`, CI `[-6.85%, +3.38%]`) | `docs/results_v3.md` §1 |
| Sensor ablation (D1 1h, `552,672` rows) | S0 `0.8408`, S4 `0.8381`, S4-no-fans `0.8381`, S4-no-power `0.8383`, S1 `0.8360` | `docs/results_v3.md` §2 |
| Random vs GroupKFold underestimation | `17.60%` (S2), `18.54%` (S3), `0.24%` (S1), `0.54%` (S4) | `docs/results_v3.md` §6 |
| Learning rate study (`0.05/0.01/0.005`) | Trees `5/27/49`; MAE delta `<0.00013°C` | `docs/results_v3.md` §7 |
| Split B (`GroupKFold`, $N=81$) | S0 `0.435 / 0.918 / 1.817°C`; S1 `0.436 / 0.924 / 1.828°C` | `docs/results_v3.md` §3 |
| Split C (5 random seeds) | S0 `0.456 / 0.940 / 1.710°C`; S1 `0.456 / 0.936 / 1.699°C` | `docs/results_v3.md` §3 |
| Fleet planning adaptation ($N=81$) | Zero-shot `5.03` vs Fleet mean `8.87` (`-43.2%`), Boosted 168h `4.87`, Residual 168h `5.02` | `docs/results_v3.md` §4 |
| Conformal Split C (`90%` nominal) | CQR `91.19%` (`3.679°C`, $r=0.470$); Moderate bin `89.09%`; Per-server `84.7–96.9%` | `docs/results_v2.md` §8 |
| Cross-cluster transfer C1→C2 | Zero-shot `6.85°C`; Residual 6h `3.16°C`, 72h `2.92°C`; Boosted 168h `6.05°C` | `docs/results_v3.md` §5 |
| Twin deployed monitoring/planning | D1 Mon `0.4023/0.8371/1.6069` (`89.96/90.00/88.14%`), Plan `5.4844` (`81.09%`); D2 Mon `0.4845/0.7807/1.2696` (`89.79/89.31/86.15%`), Plan `3.8926` (`85.33%`) | `models/twin/{d1,d2}_metadata.json` |
| 3D twin validation & compare stats | `12` draws, D1 snapshot p95 `48.03 ms`, Compare D1 `0.50–0.64°C` (`90.79–97.37%`) & D2 `0.51–0.89°C` (`87.27–90.91%`) | `docs/twin3d_validation.md` |
| Synthetic-fault alert eval (25 wins) | `100.0%` detection on crossed windows; `16.0%` (D1) & `0.0%` (D2) false alerts on `clean_control` | `docs/twin_alert_evaluation.md` |

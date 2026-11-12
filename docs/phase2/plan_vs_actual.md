# Plan vs Actual — Phase 2

> [!IMPORTANT]
> All results are **preliminary**. Statuses sourced from `docs/snapshot/08_prompt_checklist.md`.

| Phase | Item | Status | Deviation / Note |
|-------|------|--------|-----------------|
| 0.1 | Create project structure | ✅ DONE | — |
| 0.2 | venv, requirements | ✅ DONE | — |
| 0.3 | .gitignore | ✅ DONE | — |
| 0.4 | Save brief, README | ✅ DONE | — |
| 0.5 | GitHub remote | ⚠️ PARTIAL | Remote creation pending user auth; local git complete |
| 1.1 | Data audit script | ✅ DONE | — |
| 1.2 | Qualification gate | ✅ DONE | 76 D1, 60 D2 servers passed |
| 2.1 | CSV → Parquet | ✅ DONE | — |
| 2.2 | Core modules (10 mods, 1 489 lines) | ✅ DONE | — |
| 2.3 | 25 unit tests | ✅ DONE | 25/25 pass in 2.58 s |
| 2.4 | Freeze evaluation protocol | ✅ DONE | — |
| 3.1 | E1 ablation across splits A/B/C | ✅ DONE | — |
| 3.2 | Literature review | ✅ DONE | 28 entries |
| 4.1 | E2/E4 adaptation | ✅ DONE | — |
| 4.2 | E3 cross-cluster transfer | ✅ DONE | — |
| 4.3 | E4/E5 uncertainty | ✅ DONE | — |
| 5.1 | Streamlit prototype | ✅ DONE | Contains hard-coded mock (known issue) |
| 5.2 | Fault injection sim | ✅ DONE | Synthetic only (see known issues) |
| 5.3 | M100 scale-up | ⛔ SKIPPED | >1 GB data download not approved |
| 5.4 | Logger stub | ⛔ SKIPPED | College server access not provided |
| 6.1 | Paper draft | ⚠️ PARTIAL | Draft exists with legacy numbers; rewrite scheduled for Phase 3 |
| 6.2 | Tag releases | ✅ DONE | v0.1-audit, v0.9-pre-fix, v1.0-corrected |
| — | Planning regime added | 🔀 DEVIATION | Added after audit revealed monitoring/planning conflation was a critical flaw; correct methodologically |
| — | Budgets in days (some experiments) | 🔀 DEVIATION | summary_table.csv uses hours (0, 6, 24, 72, 168 h) which is correct |

**Summary:** 20/23 planned items DONE · 2 PARTIAL · 1 SKIPPED (server access) · 2 unplanned deviations (both improvements)

# Decisions Log

All project decisions not covered in the PROJECT_BRIEF.md are logged here with dates and rationale.

| Date | Decision | Rationale | Who |
|------|----------|-----------|-----|
| 2026-11-08 | Project initialised with structure from PROJECT_BRIEF.md | Following the brief's Phase 0 requirements | Agent |
| 2026-11-08 | Host column identified as `Series` | Auto-detected and verified in both datasets | Agent (audit) |
| 2026-11-08 | Time column identified as `Time` | Auto-detected and verified; format `YYYY-MM-DD HH:MM:SS` | Agent (audit) |
| 2026-11-08 | No disk/storage columns available | Neither dataset contains disk I/O or storage metrics; S2 feature set adjusted to exclude disk features | Agent (audit) |
| 2026-11-08 | Dataset 2 high missingness noted | D2 has 13-16% global missingness in many columns; only 2 servers pass the strict <10% gate; recommend relaxing to <20% for cross-cluster experiments | Agent (audit) |
| 2026-11-08 | 5 Dataset 1 servers lack inlet/fan data | `rcc196`--`rcc200` have 100% missing inlet temp and fan speeds but valid CPU temps; they can participate in S0--S3 but not S4 | Agent (audit) |
| 2026-11-08 | Inlet temperature anomalies flagged | Several servers show inlet temps of 4.0 C (unusually low); to be investigated in EDA | Agent (audit) |
| 2026-11-08 | Gate verdict: PASS (76 qualifying servers from Dataset 1) | Exceeds the 40-server threshold; Dataset 1 is the primary dataset for all experiments | Agent (audit) |

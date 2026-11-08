# Limitations

This document tracks known limitations of the project. Updated throughout the project lifecycle.

## Scope Limitations
- This is an **offline, data-driven predictive digital twin prototype** ("digital shadow with predictive capability"), not a production real-time bidirectional twin.
- No cooling control, scheduling, or job placement is attempted.
- The primary dataset comes from a single data centre (University of Melbourne research cloud); generalisability beyond this environment is not established.

## Data Limitations
- *(To be populated after the data audit)*

## Methodological Limitations
- Conformal prediction guarantees assume exchangeability, which distribution shift can break.
- Fan speed and power features may be consequences of temperature (circularity); they are tested separately in the ablation.
- The college server case study (if conducted) is one server with short history — not a basis for generalisation claims.

## Reporting Limitations
- All what-if scenarios are predictive and correlational, not causal.
- Synthetic fault injection for alert validation is clearly labelled; detection scores on synthetic faults do not represent real-world accuracy.

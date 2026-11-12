# Methodological, Data, and System Limitations

This document tracks known limitations of the Predictive Digital Twin project. Updated following the comprehensive independent diagnostic audit and re-execution protocol.

---

## 1. Scope and System Limitations
* **Digital Shadow, Not Active Control:** This project implements an offline, telemetry-driven predictive digital shadow. It does not perform active closed-loop cooling actuation, HVAC fan regulation, or real-time VM scheduling.
* **Correlational, Not Causal What-If:** What-if projections in the planning regime evaluate historical statistical correlations between workload metrics (CPU load, RAM, VMs) and steady-state CPU temperatures. They cannot simulate the physical consequences of interventional cooling adjustments.
* **Synthetic Fault Injection Disclaimers:** Alert rule testing utilizes synthetically injected thermal ramps and workload surges. Detection performance under simulated failure modes does not guarantee detection of real-world mechanical faults.

---

## 2. Telemetry and Data Limitations
* **Absence of Ambient Weather Telemetry:** The Melbourne Research Cloud trace contains no outdoor meteorological feeds (ambient dry-bulb/wet-bulb temperatures, relative humidity, or solar irradiance). Any temporal drift across seasons must be described as operational drift; attributing changes to "seasonal weather variations" is **empirically unverified (cause NOT TESTED)**.
* **Absence of Facility BMS Cooling Data:** Telemetry contains no HVAC chiller setpoints, chilled water supply/return temperatures, computer room air handler (CRAH) fan speeds, or underfloor plenum pressures.
* **Structural Telemetry Gaps:**
  * Dataset 1 nodes `qh2-rcc196` through `qh2-rcc200` lack inlet temperature and fan speed telemetry (100% missing), and span only 118 days. They are excluded from full S4 telemetry models.
  * Dataset 2 nodes `qh2-rcc263` and `qh2-rcc264` have broken DTS thermal registers (100% missing temperature).
  * Dataset 2 has structural missingness of 12% to 16% in memory and VM metrics across nodes, requiring relaxation of the data gate from 10% to 20% to retain a viable sample (59 qualifying servers under used telemetry columns, or 53 servers if strict RAM/VM presence is required across all 60 physical nodes).
* **Sensor Asymmetries and Form Factors:** Effective CPU temperature is constructed as $\max(\text{CPU1}, \text{CPU2})$ from on-die DTS registers. Dell PowerEdge R840 (dense 2U 4-socket) and Dell PowerEdge C6320 (2U 4-node modular sled) possess fundamentally different chassis thermal impedances and cooling fan curves; physical sensor comparability across clusters cannot be assumed.

---

## 3. Modeling and Methodological Limitations
* **Autoregressive Dominance & Persistence Parity:** Under short-to-medium forecast horizons (10 min to 1 hour), server temperature is heavily governed by thermal inertia. Even with current state $T(t)$ and delta targets $\Delta T = T(t+h) - T(t)$, gradient-boosted trees and ridge regression achieve approximate parity with persistence (skill scores in `results_split_a.csv` range from $+0.07\%$ at 10 min to $+0.54\%$ at 1 h, and $+0.92\%$ at 6 h; near-parity with persistence), rather than decisive double-digit improvements.
* **Row-Shifted Horizons Ignore Time Gaps:** Row-shifted horizons ignore time gaps (see `docs/phase2/gap_audit.md`): missing timestamps mean row $t+h$ does not equal clock time $t + h \times 10\text{ min}$, causing an estimated 12% of 1h targets and 41%–70% of 6h targets in Dataset 1 to reference incorrect future intervals.
* **Early Stopping at ~5 Trees Makes S1 Nearly Persistence:** With learning rate 0.05, early stopping (patience 30) terminates after ~4–5 boosting iterations, effectively collapsing S1 onto near-zero updates and persistence parity.
* **Dashboard What-If and Replay are Mocks:** The Streamlit What-If Simulator evaluates an illustrative linear formula rather than querying saved models, and historical replay intervals use fixed $\hat{q}=1.80$ with sinusoidal noise.
* **Non-Causal Nature of Fan Speeds:** Fan speeds in modern servers operate under autonomous firmware feedback control (BMC/PID). High fan speeds are an effect of elevated temperatures, not an independent cause; including them in predictive models introduces reverse-causality dependencies.
* **Overfitting in Small-Sample Adaptation:** Continued boosting on small adaptation slices ($N < 40$ samples) fails to improve over zero-shot transfer because tree split constraints (`min_child_samples`) prevent meaningful leaf updates, while unconstrained boosting overfits local sample noise. Scalar residual offset correction is more robust under data starvation.
* **Uncertainty Under-Coverage Under Extreme Regimes:** Standard constant-width split conformal prediction intervals suffer severe conditional under-coverage during active workload regimes ($10\text{--}50\%$ load) and high temperatures ($\ge 60^\circ\text{C}$). Adaptive normalized conformal scoring or Conformalized Quantile Regression (CQR) is mandatory for reliable coverage during thermal stress.

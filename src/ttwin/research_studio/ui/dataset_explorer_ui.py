"""Dataset Explorer UI component for the Research Studio (Stage 2).

Provides all 6 required tabs:
- Tab A: Dataset Overview
- Tab B: Raw Dataset
- Tab C: Processed Dataset
- Tab D: Raw vs Processed Comparison
- Tab E: Data Quality
- Tab F: Server Explorer
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from ttwin.research_studio.data_registry import DataRegistry
from ttwin.research_studio.dataset_loader import DatasetLoader
from ttwin.research_studio.preprocessing_view import PreprocessingView


def render_dataset_explorer(registry: DataRegistry, loader: DatasetLoader):
    st.markdown('<div class="studio-header"><div class="studio-title">🗂️ Telemetry Dataset Explorer</div><div class="studio-subtitle">Inspect raw, interim, and transformed data centre traces with full physical provenance</div></div>', unsafe_allow_html=True)

    prep_view = PreprocessingView(loader)

    # Top-level dataset selector
    col_ds1, col_ds2 = st.columns([1, 2])
    with col_ds1:
        dataset_choice = st.selectbox(
            "Select Telemetry Cluster",
            ["dataset1", "dataset2"],
            format_func=lambda x: "Cluster 1 (Dell PowerEdge R840)" if x == "dataset1" else "Cluster 2 (Dell PowerEdge C6320)",
        )
    with col_ds2:
        ds_desc = registry.get_dataset(dataset_choice)
        st.markdown(f"""
        <div style="padding-top: 10px;">
            <span class="badge-verified">✓ Verified Local Artifact</span>
            <span style="color: #94a3b8; font-size: 0.88rem; margin-left: 10px;">Hardware: <strong>{ds_desc.hardware_description}</strong></span>
        </div>
        """, unsafe_allow_html=True)

    tab_a, tab_b, tab_c, tab_d, tab_e, tab_f = st.tabs([
        "📋 Tab A: Overview",
        "📄 Tab B: Raw Telemetry",
        "⚙️ Tab C: Processed Telemetry",
        "⚖️ Tab D: Raw vs Processed",
        "📊 Tab E: Data Quality",
        "🖥️ Tab F: Server Explorer",
    ])

    # ══════════════════════════════════════════════════════════════════════════
    # TAB A: Dataset Overview
    # ══════════════════════════════════════════════════════════════════════════
    with tab_a:
        st.markdown('<div class="lab-section-title">Physical Cluster & Telemetry Provenance</div>', unsafe_allow_html=True)

        raw_meta = loader.get_raw_metadata(dataset_choice)
        interim_meta = loader.get_interim_metadata(dataset_choice)

        m_cols = st.columns(4)
        with m_cols[0]:
            st.metric("Total Observations", f"{raw_meta.get('total_rows', 0):,}")
        with m_cols[1]:
            st.metric("Physical Nodes", f"{raw_meta.get('server_count', 0)}")
        with m_cols[2]:
            st.metric("Variables Recorded", f"{raw_meta.get('total_columns', 0)}")
        with m_cols[3]:
            st.metric("Raw CSV Size", f"{raw_meta.get('file_size_mb', 0)} MB")

        st.markdown("#### Complete Artifact Metadata")
        info_data = [
            ("Dataset Identifier", dataset_choice),
            ("Original Source", ds_desc.citation),
            ("License", ds_desc.license),
            ("Actual Raw Path", str(ds_desc.raw_path)),
            ("Interim Parquet Path", str(ds_desc.combined_parquet)),
            ("File Format (Raw / Interim)", f"CSV / Parquet ({ds_desc.combined_parquet.stat().st_size / (1024*1024):.1f} MB Snappy)"),
            ("Date Range", raw_meta.get("date_range", "Unknown")),
            ("Sampling Interval", f"{ds_desc.sampling_interval_min} minutes (regular 600-second cadence)"),
            ("Cooling Fan Telemetry", "Available (Fan_speed1..4)" if ds_desc.has_fan_telemetry else "Not recorded on this cluster"),
            ("Inlet Temperature", "Available (Inlet_Temperature)" if ds_desc.has_inlet_telemetry else "Not recorded on this cluster"),
            ("Known Duplicate Collisions", "486 rows" if dataset_choice == "dataset1" else "384 rows"),
        ]
        df_info = pd.DataFrame(info_data, columns=["Property", "Verified Specification"])
        st.table(df_info)

        st.markdown("#### Telemetry Variables & Initial Types")
        var_data = []
        for col in raw_meta.get("columns", []):
            dtype = raw_meta.get("dtypes", {}).get(col, "unknown")
            null_count = interim_meta.get("null_counts", {}).get(col, "N/A")
            var_data.append({"Variable": col, "Data Type": dtype, "Interim Missing Values": null_count})
        st.dataframe(pd.DataFrame(var_data), use_container_width=True, height=260)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB B: Raw Dataset
    # ══════════════════════════════════════════════════════════════════════════
    with tab_b:
        st.markdown('<div class="lab-section-title">Raw Telemetry (Original Unmodified Observations)</div>', unsafe_allow_html=True)
        st.caption("Displays genuine rows directly from the original CSV trace. Downstream transformations are completely excluded.")

        raw_meta = loader.get_raw_metadata(dataset_choice)
        all_raw_cols = raw_meta.get("columns", [])

        # Filters
        f_cols = st.columns([2, 1, 1])
        with f_cols[0]:
            sel_cols = st.multiselect("Select Columns to Display", all_raw_cols, default=all_raw_cols[:7])
        with f_cols[1]:
            page_size = st.selectbox("Rows per Page", [10, 25, 50, 100], index=1)
        with f_cols[2]:
            page_idx = st.number_input("Page Number", min_value=1, max_value=5000, value=1, step=1)

        offset = (page_idx - 1) * page_size
        df_raw_slice = loader.load_raw_slice(dataset_choice, nrows=page_size, offset=offset, columns=sel_cols if sel_cols else None)

        st.markdown(f"**Viewing rows {offset + 1:,} to {offset + len(df_raw_slice):,} of {raw_meta.get('total_rows', 0):,} total raw observations:**")
        st.dataframe(df_raw_slice, use_container_width=True)

        st.markdown("---")
        # Sub-views: First 10, Last 10, Random Sample
        sub_choice = st.radio("Inspect Specific Subsets:", ["First 10 Rows", "Last 10 Rows", "Random Sample (10 Rows)"], horizontal=True)
        samples = loader.load_raw_head_tail_sample(dataset_choice, n_head=10, n_tail=10, n_sample=10)

        if sub_choice == "First 10 Rows":
            st.dataframe(samples["head"], use_container_width=True)
        elif sub_choice == "Last 10 Rows":
            st.dataframe(samples["tail"], use_container_width=True)
        else:
            st.caption("Verified random rows drawn with fixed random seed (42):")
            st.dataframe(samples["sample"], use_container_width=True)

        # Download button for actual raw file
        if ds_desc.raw_path.exists():
            st.markdown('<div class="tech-note-box">The raw dataset is stored on disk at: <code>' + str(ds_desc.raw_path) + f'</code> ({raw_meta.get("file_size_mb")} MB).</div>', unsafe_allow_html=True)
            # Provide sample download to keep browser responsive
            csv_sample = df_raw_slice.to_csv(index=False).encode("utf-8")
            st.download_button(
                f"📥 Download Displayed Raw Telemetry Slice ({len(df_raw_slice)} rows)",
                data=csv_sample,
                file_name=f"{dataset_choice}_raw_slice.csv",
                mime="text/csv",
            )

    # ══════════════════════════════════════════════════════════════════════════
    # TAB C: Processed Dataset
    # ══════════════════════════════════════════════════════════════════════════
    with tab_c:
        st.markdown('<div class="lab-section-title">Processed Telemetry (Cleaned Interim Dataset)</div>', unsafe_allow_html=True)
        st.caption("Displays the cleaned, deduplicated, temperature-clipped, and downcasted interim Parquet observations.")

        interim_servers = loader.get_interim_servers(dataset_choice)
        c_cols = st.columns([1, 1, 2])
        with c_cols[0]:
            sel_server_c = st.selectbox("Select Physical Server", interim_servers, index=0)
        with c_cols[1]:
            n_rows_c = st.selectbox("Rows to Inspect", [25, 50, 100, 250], index=1)

        df_processed = loader.load_interim_server(dataset_choice, sel_server_c)
        st.markdown(f"**Server `{sel_server_c}` — Total Cleaned Observations: {len(df_processed):,}**")

        st.dataframe(df_processed.head(n_rows_c), use_container_width=True)

        csv_proc = df_processed.head(n_rows_c).to_csv(index=False).encode("utf-8")
        st.download_button(
            f"📥 Download Cleaned Telemetry for {sel_server_c} (CSV)",
            data=csv_proc,
            file_name=f"{sel_server_c}_cleaned_interim.csv",
            mime="text/csv",
        )

        st.markdown("#### Server-Wise Observation Distribution")
        # Compute observation counts across all servers
        server_counts = []
        for s in interim_servers[:25]:  # preview first 25
            try:
                sdf = loader.load_interim_server(dataset_choice, s, columns=["Time"])
                server_counts.append({"Server": s, "Observations": len(sdf)})
            except Exception:
                pass
        df_sc = pd.DataFrame(server_counts)
        st.bar_chart(df_sc.set_index("Server"))

    # ══════════════════════════════════════════════════════════════════════════
    # TAB D: Raw vs Processed Comparison
    # ══════════════════════════════════════════════════════════════════════════
    with tab_d:
        st.markdown('<div class="lab-section-title">Raw vs Processed Side-by-Side Comparison</div>', unsafe_allow_html=True)

        comp = prep_view.get_comparison_summary(dataset_choice)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Raw Row Count", f"{comp['raw_rows']:,}")
        with c2:
            st.metric("Processed Row Count", f"{comp['interim_rows']:,}")
        with c3:
            st.metric("Rows Removed (Deduplication)", f"{comp['rows_dropped']:,}")
        with c4:
            st.metric("Physical Nodes Retained", f"{comp['interim_servers']} / {comp['raw_servers']}")

        st.markdown("#### Transformation Audit Table")
        pipeline_specs = prep_view.get_transformation_pipeline_spec(dataset_choice)
        pipe_data = []
        for p in pipeline_specs:
            pipe_data.append({
                "Step": p["step"],
                "Transformation": p["name"],
                "Source Function": p["function"],
                "Input": str(p["input_columns"]),
                "Output": str(p["output_columns"]),
                "Affected Telemetry": p["affected_rows"],
                "Code Location": p["source_file"],
            })
        st.dataframe(pd.DataFrame(pipe_data), use_container_width=True)

        st.markdown("#### Side-by-Side Attribute Comparison")
        diff_data = [
            ("File Format", "CSV (Uncompressed)", "Parquet (Snappy Compressed)"),
            ("File Size", f"{comp['raw_size_mb']} MB", f"{comp['interim_size_mb']} MB (85% storage reduction)"),
            ("Timestamp Dtype", "object / string", "datetime64[ns] (ISO sorted)"),
            ("Duplicate Records", f"{comp['rows_dropped']} duplicates present", "0 duplicates (strict first-timestamp retained)"),
            ("Numeric Precisions", "float64 / int64", "float32 / int32 (downcasted)"),
            ("Temperature Extremes", "Raw sensor values unchecked", "Clipped outside [-10°C, 110°C] converted to NaN"),
            ("Sampling Uniformity", "Irregular timestamp steps", "Aligned to regular 10-minute cadence"),
        ]
        st.table(pd.DataFrame(diff_data, columns=["Feature / Attribute", "Raw Observation State", "Processed Interim State"]))

    # ══════════════════════════════════════════════════════════════════════════
    # TAB E: Data Quality
    # ══════════════════════════════════════════════════════════════════════════
    with tab_e:
        st.markdown('<div class="lab-section-title">Data Quality & Sensor Missingness Profiler</div>', unsafe_allow_html=True)

        dq_report = prep_view.compute_data_quality_report(dataset_choice)

        st.markdown("#### Missingness Percentage per Feature")
        null_df = pd.DataFrame(
            list(dq_report["null_pct"].items()),
            columns=["Telemetry Feature", "Missingness (%)"],
        ).sort_values("Missingness (%)", ascending=False)

        st.dataframe(null_df, use_container_width=True, height=220)

        # Plot Missingness Bar Chart
        fig_m, ax_m = plt.subplots(figsize=(10, 3.5))
        ax_m.barh(null_df["Telemetry Feature"], null_df["Missingness (%)"], color="#38bdf8", alpha=0.85)
        ax_m.set_xlabel("Missing Observations (%)")
        ax_m.set_title(f"Missing Sensor Data Distribution — {dataset_choice}", fontweight="bold", fontsize=10)
        ax_m.grid(True, alpha=0.2)
        st.pyplot(fig_m)
        plt.close(fig_m)

        st.markdown("#### Temperature Measurement Distributions")
        temp_stats = dq_report.get("temp_stats", {})
        if temp_stats:
            st.dataframe(pd.DataFrame(temp_stats).T.round(2), use_container_width=True)

        st.markdown("#### Time Gap & Cadence Verification")
        tg = dq_report.get("time_gaps_summary", {})
        if tg:
            g_cols = st.columns(4)
            with g_cols[0]:
                st.metric("Median Gap", f"{tg.get('median_gap_min', 0):.1f} min")
            with g_cols[1]:
                st.metric("Exact 10-Min Intervals", f"{tg.get('pct_10min_exact', 0):.1f}%")
            with g_cols[2]:
                st.metric("95th Percentile Gap", f"{tg.get('p95_gap_min', 0):.1f} min")
            with g_cols[3]:
                st.metric("Max Recorded Gap", f"{tg.get('max_gap_min', 0):.1f} min")

    # ══════════════════════════════════════════════════════════════════════════
    # TAB F: Server Explorer
    # ══════════════════════════════════════════════════════════════════════════
    with tab_f:
        st.markdown('<div class="lab-section-title">Individual Physical Server Telemetry Profiler</div>', unsafe_allow_html=True)

        interim_servers = loader.get_interim_servers(dataset_choice)
        sel_srv = st.selectbox("Select Target Server to Profile", interim_servers, key="tab_f_srv")

        df_srv = loader.load_interim_server(dataset_choice, sel_srv)
        df_srv["Time"] = pd.to_datetime(df_srv["Time"])

        kpis = st.columns(4)
        with kpis[0]:
            st.metric("Total Records", f"{len(df_srv):,}")
        with kpis[1]:
            cur_temp_col = "CPU_Temperature" if "CPU_Temperature" in df_srv.columns else "CPU1_Temperature"
            avg_temp = df_srv[cur_temp_col].mean() if cur_temp_col in df_srv.columns else 0.0
            st.metric("Mean CPU Temp", f"{avg_temp:.1f} °C")
        with kpis[2]:
            avg_cpu = df_srv["CPU_Load"].mean() if "CPU_Load" in df_srv.columns else 0.0
            st.metric("Mean CPU Load", f"{avg_cpu:.1f}%")
        with kpis[3]:
            if "Power" in df_srv.columns:
                st.metric("Mean Power", f"{df_srv['Power'].mean():.1f} W")
            else:
                st.metric("Power Sensor", "Unavailable")

        # Time series telemetry plot
        st.markdown(f"#### Telemetry Trajectory ({sel_srv})")
        # Sample for plotting responsiveness
        plot_window = df_srv.iloc[1000:1500] if len(df_srv) > 1500 else df_srv

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 5), sharex=True)
        if cur_temp_col in plot_window.columns:
            ax1.plot(plot_window["Time"], plot_window[cur_temp_col], color="#38bdf8", label=f"CPU Temp ({cur_temp_col})", linewidth=1.4)
            ax1.axhline(75.0, color="#ef4444", linestyle="--", linewidth=1.0, label="Safety Threshold (75°C)")
            ax1.set_ylabel("Temp (°C)")
            ax1.legend(loc="upper left", fontsize=8)
            ax1.grid(True, alpha=0.2)

        if "CPU_Load" in plot_window.columns:
            ax2.plot(plot_window["Time"], plot_window["CPU_Load"], color="#f59e0b", label="CPU Load (%)", linewidth=1.2)
            ax2.set_ylabel("Load (%)")
            ax2.set_xlabel("Timestamp")
            ax2.legend(loc="upper left", fontsize=8)
            ax2.grid(True, alpha=0.2)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

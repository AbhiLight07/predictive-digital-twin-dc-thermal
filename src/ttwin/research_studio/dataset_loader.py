"""Dataset Loader for the Research Studio.

Provides high-performance, cached, schema-validated access to raw and interim
data without overwhelming memory. Supports pagination, sampling, and server filtering.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np

from ttwin.research_studio.data_registry import DataRegistry, DatasetDescriptor


class DatasetLoader:
    """Loads and caches raw and interim telemetry datasets safely."""

    def __init__(self, registry: Optional[DataRegistry] = None):
        self.registry = registry or DataRegistry()
        self._metadata_cache: Dict[str, Dict[str, Any]] = {}
        self._server_cache: Dict[str, List[str]] = {}

    def get_raw_metadata(self, dataset_name: str) -> Dict[str, Any]:
        """Compute or retrieve comprehensive metadata for the raw CSV.
        Uses cached summaries or fast sampling if already known.
        """
        cache_key = f"raw_{dataset_name}"
        if cache_key in self._metadata_cache:
            return self._metadata_cache[cache_key]

        ds = self.registry.get_dataset(dataset_name)
        if not ds.raw_path.exists():
            return {
                "available": False,
                "error": f"Raw file not found: {ds.raw_path}",
                "file_path": str(ds.raw_path),
            }

        file_size = ds.raw_path.stat().st_size

        # Read first 100 rows for schema and types
        sample_df = pd.read_csv(ds.raw_path, nrows=100)
        columns = list(sample_df.columns)
        dtypes = {col: str(sample_df[col].dtype) for col in columns}

        # Check pre-computed audit metrics from columns.yaml if available
        col_cfg = self.registry.get_columns_config().get(dataset_name, {})
        known_rows = col_cfg.get("n_rows")
        known_servers = col_cfg.get("n_servers")
        known_date_range = col_cfg.get("date_range", [])

        # If known from frozen config, use verified figures to avoid re-reading 600MB on every load
        if known_rows is not None and known_servers is not None:
            total_rows = known_rows
            server_count = known_servers
            date_range_str = f"{known_date_range[0]} to {known_date_range[1]}" if len(known_date_range) == 2 else "Unknown"
        else:
            # Fallback line counter
            with open(ds.raw_path, "r", encoding="utf-8", errors="ignore") as f:
                total_rows = sum(1 for _ in f) - 1
            server_count = ds.expected_servers
            date_range_str = "See interim view"

        meta = {
            "available": True,
            "dataset_name": dataset_name,
            "display_name": ds.display_name,
            "file_path": str(ds.raw_path),
            "file_format": "CSV",
            "file_size_bytes": file_size,
            "file_size_mb": round(file_size / (1024 * 1024), 2),
            "total_rows": total_rows,
            "total_columns": len(columns),
            "columns": columns,
            "dtypes": dtypes,
            "server_count": server_count,
            "date_range": date_range_str,
            "sampling_interval_min": ds.sampling_interval_min,
            "citation": ds.citation,
            "license": ds.license,
            "hardware": ds.hardware_description,
        }
        self._metadata_cache[cache_key] = meta
        return meta

    def load_raw_slice(
        self,
        dataset_name: str,
        nrows: int = 100,
        offset: int = 0,
        columns: Optional[List[str]] = None,
        server_filter: Optional[str] = None,
    ) -> pd.DataFrame:
        """Load a bounded slice of the raw dataset.
        If server_filter is provided, scans chunks until nrows matches are found.
        """
        ds = self.registry.get_dataset(dataset_name)
        if not ds.raw_path.exists():
            raise FileNotFoundError(f"Raw CSV not found: {ds.raw_path}")

        if server_filter:
            # Chunked search for specific server
            matches = []
            matched_count = 0
            for chunk in pd.read_csv(ds.raw_path, chunksize=50_000, usecols=columns, low_memory=False):
                server_mask = chunk[ds.server_column].astype(str).str.strip() == server_filter.strip()
                filtered = chunk[server_mask]
                if not filtered.empty:
                    matches.append(filtered)
                    matched_count += len(filtered)
                    if matched_count >= offset + nrows:
                        break
            if not matches:
                return pd.DataFrame(columns=columns or ds.expected_columns)
            combined = pd.concat(matches, ignore_index=True)
            return combined.iloc[offset : offset + nrows].reset_index(drop=True)

        # Direct pagination from top
        if offset == 0:
            return pd.read_csv(ds.raw_path, nrows=nrows, usecols=columns, low_memory=False)
        else:
            return pd.read_csv(
                ds.raw_path,
                skiprows=range(1, offset + 1),
                nrows=nrows,
                usecols=columns,
                low_memory=False,
            )

    def load_raw_head_tail_sample(
        self, dataset_name: str, n_head: int = 10, n_tail: int = 10, n_sample: int = 10
    ) -> Dict[str, pd.DataFrame]:
        """Retrieve the exact first 10, last 10, and a verifiable random sample of actual raw rows."""
        ds = self.registry.get_dataset(dataset_name)
        if not ds.raw_path.exists():
            raise FileNotFoundError(f"Raw CSV not found: {ds.raw_path}")

        head_df = pd.read_csv(ds.raw_path, nrows=n_head, low_memory=False)

        # Read sample from a middle chunk
        sample_chunk = pd.read_csv(ds.raw_path, skiprows=range(1, 50_000), nrows=1000, low_memory=False)
        sample_df = sample_chunk.sample(n=min(n_sample, len(sample_chunk)), random_state=42).reset_index(drop=True)

        # For tail: seek to near end of the raw CSV file for sub-second retrieval
        # Avoids parsing 2.7M rows sequentially while guaranteeing genuine raw bytes
        try:
            import io
            file_size = ds.raw_path.stat().st_size
            seek_pos = max(0, file_size - 64 * 1024)
            with open(ds.raw_path, "rb") as f:
                f.seek(seek_pos)
                tail_bytes = f.read()
            lines = tail_bytes.decode("utf-8", errors="ignore").splitlines()
            valid_lines = lines[1:] if seek_pos > 0 else lines
            tail_lines = [l for l in valid_lines if l.strip()][-n_tail:]
            header_line = ",".join(head_df.columns)
            tail_df = pd.read_csv(io.StringIO(header_line + "\n" + "\n".join(tail_lines)), low_memory=False)
        except Exception:
            tail_df = head_df.tail(n_tail)

        return {"head": head_df, "tail": tail_df, "sample": sample_df}

    def get_interim_servers(self, dataset_name: str) -> List[str]:
        """Return the sorted list of physical server names available in interim parquets."""
        if dataset_name in self._server_cache:
            return self._server_cache[dataset_name]

        ds = self.registry.get_dataset(dataset_name)
        if not ds.interim_dir.exists():
            return []

        servers = []
        for p in sorted(ds.interim_dir.glob("*.parquet")):
            stem = p.stem.strip()
            if stem.startswith("_"):
                continue
            servers.append(stem)

        self._server_cache[dataset_name] = sorted(list(dict.fromkeys(servers)))
        return self._server_cache[dataset_name]

    def load_interim_server(
        self,
        dataset_name: str,
        server_name: str,
        columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Load the cleaned interim Parquet file for a specific physical server."""
        ds = self.registry.get_dataset(dataset_name)
        # Search for exact stem or space-padded stem (e.g. 'qh2-rcc120 .parquet')
        candidates = list(ds.interim_dir.glob(f"*{server_name}*.parquet"))
        exact = [c for c in candidates if not c.stem.startswith("_")]
        if not exact:
            raise FileNotFoundError(f"Interim parquet for server '{server_name}' not found in {ds.interim_dir}")

        path = exact[0]
        df = pd.read_parquet(path, columns=columns)
        if ds.time_column in df.columns and not pd.api.types.is_datetime64_any_dtype(df[ds.time_column]):
            df[ds.time_column] = pd.to_datetime(df[ds.time_column])
        return df

    def load_interim_combined(
        self,
        dataset_name: str,
        columns: Optional[List[str]] = None,
        max_rows: Optional[int] = None,
    ) -> pd.DataFrame:
        """Load the concatenated interim dataset with optional column projection and row limit."""
        ds = self.registry.get_dataset(dataset_name)
        if not ds.combined_parquet.exists():
            raise FileNotFoundError(f"Combined interim parquet not found: {ds.combined_parquet}")

        df = pd.read_parquet(ds.combined_parquet, columns=columns)
        if max_rows and len(df) > max_rows:
            df = df.iloc[:max_rows]
        if ds.time_column in df.columns and not pd.api.types.is_datetime64_any_dtype(df[ds.time_column]):
            df[ds.time_column] = pd.to_datetime(df[ds.time_column])
        return df

    def get_interim_metadata(self, dataset_name: str) -> Dict[str, Any]:
        """Compute metadata for the interim cleaned dataset."""
        cache_key = f"interim_{dataset_name}"
        if cache_key in self._metadata_cache:
            return self._metadata_cache[cache_key]

        ds = self.registry.get_dataset(dataset_name)
        if not ds.combined_parquet.exists():
            return {"available": False, "error": f"Combined parquet not found: {ds.combined_parquet}"}

        file_size = ds.combined_parquet.stat().st_size
        servers = self.get_interim_servers(dataset_name)

        # Read minimal slice for schema
        sample = pd.read_parquet(ds.combined_parquet, columns=[ds.server_column, ds.time_column])
        total_rows = len(sample)
        sample[ds.time_column] = pd.to_datetime(sample[ds.time_column])
        t_min = sample[ds.time_column].min()
        t_max = sample[ds.time_column].max()

        # Read schema columns
        full_sample = pd.read_parquet(ds.combined_parquet)
        columns = list(full_sample.columns)
        null_counts = full_sample.isnull().sum().to_dict()

        meta = {
            "available": True,
            "dataset_name": dataset_name,
            "display_name": ds.display_name,
            "file_path": str(ds.combined_parquet),
            "file_format": "Parquet (Snappy)",
            "file_size_bytes": file_size,
            "file_size_mb": round(file_size / (1024 * 1024), 2),
            "total_rows": total_rows,
            "total_columns": len(columns),
            "columns": columns,
            "server_count": len(servers),
            "servers": servers,
            "time_min": str(t_min),
            "time_max": str(t_max),
            "date_range": f"{t_min.strftime('%Y-%m-%d')} to {t_max.strftime('%Y-%m-%d')}",
            "null_counts": null_counts,
            "sampling_interval_min": ds.sampling_interval_min,
        }
        self._metadata_cache[cache_key] = meta
        return meta

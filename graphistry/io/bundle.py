"""
Generic bundle engine for serializing/deserializing data bundles.

Low-level, Plottable-agnostic. Handles file I/O, SHA256 integrity,
parquet read/write, manifest management, and zip/dir format support.
"""
import hashlib
import json
import os
import shutil
import tempfile
import zipfile
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd


def _require_pydantic() -> Any:
    """Import and return pydantic module, raising clear error if missing."""
    try:
        import pydantic
        if int(pydantic.VERSION.split('.')[0]) < 2:
            raise ImportError(
                "graphistry serialization requires pydantic >= 2.0. "
                "Install with: pip install 'graphistry[serialization]'"
            )
        return pydantic
    except ImportError:
        raise ImportError(
            "graphistry serialization requires pydantic >= 2.0. "
            "Install with: pip install 'graphistry[serialization]'"
        )


def sha256_file(path: str) -> str:
    """Compute SHA256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    """Compute SHA256 hex digest of bytes."""
    return hashlib.sha256(data).hexdigest()


@dataclass
class BundleWriteReport:
    """Report from a bundle write operation."""
    warnings: List[str] = field(default_factory=list)
    artifacts_written: List[str] = field(default_factory=list)
    artifacts_skipped: List[str] = field(default_factory=list)

    def __repr__(self) -> str:
        return (
            f"BundleWriteReport(written={len(self.artifacts_written)}, "
            f"skipped={len(self.artifacts_skipped)}, "
            f"warnings={len(self.warnings)})"
        )


@dataclass
class BundleReadReport:
    """Report from a bundle read operation."""
    warnings: List[str] = field(default_factory=list)
    artifacts_loaded: List[str] = field(default_factory=list)
    artifacts_skipped: List[str] = field(default_factory=list)
    integrity_ok: bool = True
    remote_state_skipped: bool = False

    def __repr__(self) -> str:
        return (
            f"BundleReadReport(loaded={len(self.artifacts_loaded)}, "
            f"skipped={len(self.artifacts_skipped)}, "
            f"integrity_ok={self.integrity_ok}, "
            f"warnings={len(self.warnings)})"
        )


def _df_to_pandas(df: Any) -> pd.DataFrame:
    """Convert a DataFrame to pandas if it's a cuDF DataFrame."""
    if hasattr(df, 'to_pandas'):
        try:
            return df.to_pandas()
        except Exception:
            pass
    return df


def write_df_parquet(
    df: Any,
    name: str,
    bundle_dir: str,
    report: BundleWriteReport,
) -> Optional[Dict[str, str]]:
    """Write a DataFrame as parquet to bundle_dir/data/{name}.parquet.

    Returns artifact dict {kind, path, sha256} or None on failure.
    """
    if df is None:
        return None

    data_dir = os.path.join(bundle_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)
    rel_path = os.path.join('data', f'{name}.parquet')
    abs_path = os.path.join(bundle_dir, rel_path)

    try:
        pdf = _df_to_pandas(df)
        if not isinstance(pdf, pd.DataFrame):
            report.warnings.append(f"{name}: not a DataFrame, skipping")
            report.artifacts_skipped.append(name)
            return None
        pdf.to_parquet(abs_path)
    except Exception as e:
        report.warnings.append(f"{name}: failed to write parquet: {e}")
        report.artifacts_skipped.append(name)
        return None

    sha = sha256_file(abs_path)
    report.artifacts_written.append(name)
    return {
        'kind': 'parquet',
        'path': rel_path,
        'sha256': sha,
    }


def read_df_parquet(
    rel_path: str,
    bundle_dir: str,
    expected_sha: Optional[str],
    report: BundleReadReport,
) -> Optional[pd.DataFrame]:
    """Read a parquet file from bundle_dir and verify SHA256.

    Returns DataFrame or None on failure.
    """
    abs_path = os.path.join(bundle_dir, rel_path)
    if not os.path.exists(abs_path):
        report.warnings.append(f"File not found: {rel_path}")
        return None

    if expected_sha is not None:
        actual_sha = sha256_file(abs_path)
        if actual_sha != expected_sha:
            report.warnings.append(
                f"SHA256 mismatch for {rel_path}: "
                f"expected {expected_sha[:16]}..., got {actual_sha[:16]}..."
            )
            report.integrity_ok = False

    try:
        return pd.read_parquet(abs_path)
    except Exception as e:
        report.warnings.append(f"Failed to read parquet {rel_path}: {e}")
        return None


def write_manifest(manifest: Dict[str, Any], bundle_dir: str) -> None:
    """Write manifest.json to bundle_dir."""
    path = os.path.join(bundle_dir, 'manifest.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False, default=str)


def read_manifest(bundle_dir: str) -> Dict[str, Any]:
    """Read manifest.json from bundle_dir."""
    path = os.path.join(bundle_dir, 'manifest.json')
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def dir_to_zip(src_dir: str, zip_path: str) -> None:
    """Create a zip archive from a bundle directory."""
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, _dirs, files in os.walk(src_dir):
            for fname in files:
                abs_path = os.path.join(root, fname)
                rel_path = os.path.relpath(abs_path, src_dir)
                zf.write(abs_path, rel_path)


def zip_to_dir(zip_path: str, dest_dir: str) -> None:
    """Extract a zip archive to dest_dir with zip-slip protection."""
    abs_dest = os.path.realpath(dest_dir)
    with zipfile.ZipFile(zip_path, 'r') as zf:
        for member in zf.namelist():
            member_path = os.path.realpath(os.path.join(dest_dir, member))
            if not member_path.startswith(abs_dest + os.sep) and member_path != abs_dest:
                raise ValueError(
                    f"Zip-slip detected: {member} would extract outside {dest_dir}"
                )
        zf.extractall(dest_dir)


def detect_format(path: str) -> str:
    """Detect whether path is a directory bundle or a zip archive.

    Returns "dir" or "zip".
    Raises FileNotFoundError if path doesn't exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Bundle path does not exist: {path}")
    if os.path.isdir(path):
        return "dir"
    if zipfile.is_zipfile(path):
        return "zip"
    raise ValueError(f"Path is neither a directory nor a zip file: {path}")


def prepare_bundle_dir(path: str, fmt: Optional[str]) -> str:
    """Create and return the working bundle directory.

    If fmt is "zip", creates a temp directory that will later be zipped.
    Otherwise creates the directory at path directly.
    """
    if fmt == "zip":
        return tempfile.mkdtemp(prefix="graphistry_bundle_")
    else:
        os.makedirs(path, exist_ok=True)
        return path


def finalize_bundle(bundle_dir: str, path: str, fmt: Optional[str]) -> None:
    """Finalize the bundle: zip if needed, clean up temp dir."""
    if fmt == "zip":
        try:
            dir_to_zip(bundle_dir, path)
        finally:
            shutil.rmtree(bundle_dir, ignore_errors=True)

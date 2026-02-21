"""Tests for graphistry.io.bundle — generic bundle engine."""
import hashlib
import json
import os
import tempfile
import unittest
import zipfile

import pandas as pd

from graphistry.io.bundle import (
    BundleReadReport,
    BundleWriteReport,
    _require_pydantic,
    detect_format,
    dir_to_zip,
    read_df_parquet,
    read_manifest,
    sha256_bytes,
    sha256_file,
    write_df_parquet,
    write_manifest,
    zip_to_dir,
)


class TestSHA256(unittest.TestCase):
    def test_sha256_bytes_consistency(self):
        data = b"hello world"
        expected = hashlib.sha256(data).hexdigest()
        self.assertEqual(sha256_bytes(data), expected)

    def test_sha256_file_consistency(self):
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"test data for hashing")
            f.flush()
            path = f.name
        try:
            expected = hashlib.sha256(b"test data for hashing").hexdigest()
            self.assertEqual(sha256_file(path), expected)
        finally:
            os.unlink(path)

    def test_sha256_empty(self):
        result = sha256_bytes(b"")
        expected = hashlib.sha256(b"").hexdigest()
        self.assertEqual(result, expected)


class TestParquetRoundtrip(unittest.TestCase):
    def test_write_read_roundtrip(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        with tempfile.TemporaryDirectory() as td:
            report = BundleWriteReport()
            art = write_df_parquet(df, "test_df", td, report)
            self.assertIsNotNone(art)
            self.assertEqual(art["kind"], "parquet")
            self.assertIn("test_df", report.artifacts_written)

            read_report = BundleReadReport()
            result = read_df_parquet(
                art["path"], td, art["sha256"], read_report
            )
            self.assertIsNotNone(result)
            pd.testing.assert_frame_equal(result, df)
            self.assertTrue(read_report.integrity_ok)

    def test_write_none_returns_none(self):
        with tempfile.TemporaryDirectory() as td:
            report = BundleWriteReport()
            art = write_df_parquet(None, "missing", td, report)
            self.assertIsNone(art)

    def test_sha_mismatch_detected(self):
        df = pd.DataFrame({"x": [1]})
        with tempfile.TemporaryDirectory() as td:
            report = BundleWriteReport()
            art = write_df_parquet(df, "test", td, report)

            read_report = BundleReadReport()
            result = read_df_parquet(
                art["path"], td, "bad_sha256_value", read_report
            )
            # Should still return data but flag integrity issue
            self.assertIsNotNone(result)
            self.assertFalse(read_report.integrity_ok)
            self.assertTrue(any("mismatch" in w.lower() for w in read_report.warnings))

    def test_non_dataframe_skipped(self):
        with tempfile.TemporaryDirectory() as td:
            report = BundleWriteReport()
            art = write_df_parquet("not a dataframe", "bad", td, report)
            self.assertIsNone(art)
            self.assertIn("bad", report.artifacts_skipped)


class TestManifest(unittest.TestCase):
    def test_write_read_roundtrip(self):
        manifest = {
            "schema_version": "1.0",
            "artifacts": {"_edges": {"kind": "parquet", "path": "data/_edges.parquet"}},
            "nested": {"key": [1, 2, 3]},
        }
        with tempfile.TemporaryDirectory() as td:
            write_manifest(manifest, td)
            result = read_manifest(td)
            self.assertEqual(result, manifest)


class TestZipRoundtrip(unittest.TestCase):
    def test_dir_to_zip_to_dir(self):
        with tempfile.TemporaryDirectory() as src:
            # Create some files
            os.makedirs(os.path.join(src, "data"))
            with open(os.path.join(src, "manifest.json"), "w") as f:
                json.dump({"test": True}, f)
            with open(os.path.join(src, "data", "file.txt"), "w") as f:
                f.write("hello")

            with tempfile.TemporaryDirectory() as tmp:
                zip_path = os.path.join(tmp, "test.zip")
                dir_to_zip(src, zip_path)
                self.assertTrue(os.path.exists(zip_path))

                dest = os.path.join(tmp, "extracted")
                os.makedirs(dest)
                zip_to_dir(zip_path, dest)

                # Verify contents
                with open(os.path.join(dest, "manifest.json")) as f:
                    self.assertEqual(json.load(f), {"test": True})
                with open(os.path.join(dest, "data", "file.txt")) as f:
                    self.assertEqual(f.read(), "hello")

    def test_zip_slip_protection(self):
        with tempfile.TemporaryDirectory() as tmp:
            zip_path = os.path.join(tmp, "evil.zip")
            with zipfile.ZipFile(zip_path, "w") as zf:
                zf.writestr("../../../etc/evil.txt", "malicious")

            dest = os.path.join(tmp, "safe")
            os.makedirs(dest)
            with self.assertRaises(ValueError) as ctx:
                zip_to_dir(zip_path, dest)
            self.assertIn("Zip-slip", str(ctx.exception))


class TestDetectFormat(unittest.TestCase):
    def test_detect_dir(self):
        with tempfile.TemporaryDirectory() as td:
            self.assertEqual(detect_format(td), "dir")

    def test_detect_zip(self):
        with tempfile.TemporaryDirectory() as tmp:
            zip_path = os.path.join(tmp, "test.zip")
            with zipfile.ZipFile(zip_path, "w") as zf:
                zf.writestr("test.txt", "hello")
            self.assertEqual(detect_format(zip_path), "zip")

    def test_nonexistent_raises(self):
        with self.assertRaises(FileNotFoundError):
            detect_format("/nonexistent/path/xyz")

    def test_non_zip_file_raises(self):
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"not a zip")
            path = f.name
        try:
            with self.assertRaises(ValueError):
                detect_format(path)
        finally:
            os.unlink(path)


class TestRequirePydantic(unittest.TestCase):
    def test_pydantic_importable(self):
        """If pydantic is installed, _require_pydantic should succeed."""
        try:
            pydantic = _require_pydantic()
            self.assertTrue(hasattr(pydantic, 'BaseModel'))
        except ImportError:
            # If pydantic not installed, verify error message
            try:
                _require_pydantic()
            except ImportError as e:
                self.assertIn("pydantic", str(e))
                self.assertIn("serialization", str(e))


class TestReports(unittest.TestCase):
    def test_write_report_repr(self):
        r = BundleWriteReport()
        r.artifacts_written.append("a")
        r.artifacts_skipped.append("b")
        r.warnings.append("w")
        s = repr(r)
        self.assertIn("written=1", s)
        self.assertIn("skipped=1", s)

    def test_read_report_repr(self):
        r = BundleReadReport()
        r.artifacts_loaded.append("a")
        s = repr(r)
        self.assertIn("loaded=1", s)
        self.assertIn("integrity_ok=True", s)


if __name__ == "__main__":
    unittest.main()

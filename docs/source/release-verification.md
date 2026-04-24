# Release Verification Guide

This guide explains what release evidence is published by `pygraphistry`, where it is stored, and how consumers can verify it.

## What Is Published Where

- **PyPI/TestPyPI distribution files**: `dist/*.whl` and `dist/*.tar.gz`
- **PyPI/TestPyPI attestations/provenance**: uploaded during publish (`attestations: true`)
- **SBOM (CycloneDX JSON)**: generated in CI as `evidence/sbom-cyclonedx.json` and uploaded as a GitHub Actions artifact (`release-evidence-<run_id>`)

Important: SBOM is currently workflow evidence, not a separate project-level artifact shown in the PyPI/TestPyPI package UI.

## Verify PyPI Attestations/Provenance (Consumer Path)

Use the official PyPI attestation verifier (`pypi-attestations`) with a direct distribution-file URL from the release you want to verify.

```bash
python -m pip install --upgrade pypi-attestations

# Example: wheel URL from the target graphistry release on PyPI
WHEEL_DIRECT_URL=https://files.pythonhosted.org/path/to/graphistry-X.Y.Z-py3-none-any.whl

pypi-attestations verify pypi \
  --repository https://github.com/graphistry/pygraphistry \
  "$WHEEL_DIRECT_URL"
```

This checks that:
- the provenance object is valid for the release file
- the signing identity maps to the expected GitHub repository/workflow publisher
- cryptographic verification succeeds against the attested file digest

Reference: PyPI docs on consuming attestations and the Integrity API.

## Retrieve and Inspect SBOM Evidence

Download release evidence from the corresponding GitHub Actions run:

```bash
RUN_ID=<publish_workflow_run_id>
gh run download "$RUN_ID" \
  --repo graphistry/pygraphistry \
  -n "release-evidence-${RUN_ID}" \
  -D ./release-evidence

jq -r ".bomFormat, .specVersion" ./release-evidence/sbom-cyclonedx.json
```

Expected output starts with:
- `CycloneDX`
- a supported CycloneDX spec version (for example `1.6`)

## Operational Notes

- TestPyPI runs are useful for publish-path smoke testing.
- Consumer verification policy should be anchored on PyPI release artifacts and attestation verification.
- Keep Trusted Publisher settings aligned with `.github/workflows/publish-pypi.yml` and environment `pypi-release`.

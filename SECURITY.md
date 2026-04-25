# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| head    | :white_check_mark: |

## Reporting a Bug

Please (1) submit issue with preferred contact information and label [security], but no sensitive information and (2) send a corresponding email to security@graphistry.com

We will respond within 48 hours.

## Disclosure policy

We will confirm the issue and severity, and as appropriate, prepare a fix and release plan, and if desired, public acknowledgement.

## Verifying Releases

PyPI distributions ship with [attestations and provenance](https://docs.pypi.org/attestations/consuming-attestations/); SBOMs (CycloneDX) are uploaded as a workflow artifact (`release-evidence-<run_id>/sbom-cyclonedx.json`).

```bash
pip install pypi-attestations
pypi-attestations verify pypi \
  --repository https://github.com/graphistry/pygraphistry \
  https://files.pythonhosted.org/.../graphistry-X.Y.Z-py3-none-any.whl

gh run download <run_id> --repo graphistry/pygraphistry \
  -n release-evidence-<run_id> -D ./release-evidence
```

A dedicated docs Security section is tracked in #1208.

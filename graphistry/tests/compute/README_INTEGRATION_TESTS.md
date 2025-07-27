# Integration Test Configuration

This directory contains both unit tests (always run) and integration tests (opt-in).

## Environment Variables for Integration Tests

### GPU Tests
```bash
# Enable CUDF/GPU tests
TEST_CUDF=1 pytest test_chain_dag_gpu.py
```

### Remote Graph Integration Tests
```bash
# Enable remote Graphistry server tests
TEST_REMOTE_INTEGRATION=1 pytest test_chain_dag_remote_integration.py

# Additional configuration for remote tests:
GRAPHISTRY_USERNAME=myuser         # Username for auth
GRAPHISTRY_PASSWORD=mypass         # Password for auth
GRAPHISTRY_API_KEY=key-123         # Alternative to username/password
GRAPHISTRY_SERVER=hub.graphistry.com  # Server URL (optional)
GRAPHISTRY_TEST_DATASET_ID=abc123  # Known dataset for testing (optional)
```

## Running All Tests

```bash
# Unit tests only (fast, no external dependencies)
pytest

# All tests including integration
TEST_CUDF=1 TEST_REMOTE_INTEGRATION=1 pytest
```

## Writing New Integration Tests

1. **Use environment variable guards:**
   ```python
   import os
   import pytest
   
   REMOTE_INTEGRATION_ENABLED = os.environ.get("TEST_REMOTE_INTEGRATION") == "1"
   skip_remote = pytest.mark.skipif(
       not REMOTE_INTEGRATION_ENABLED,
       reason="Remote integration tests need TEST_REMOTE_INTEGRATION=1"
   )
   
   @skip_remote
   def test_my_remote_feature():
       # This only runs when TEST_REMOTE_INTEGRATION=1
       pass
   ```

2. **Always provide mocked versions:**
   - Integration tests verify real behavior
   - Unit tests with mocks ensure CI/CD still validates core logic
   
3. **Document requirements:**
   - What env vars are needed
   - What external services must be running
   - Expected test data setup

## CI/CD Configuration

The CI/CD pipeline runs only unit tests by default. Integration tests can be enabled
in specific CI jobs by setting the appropriate environment variables.
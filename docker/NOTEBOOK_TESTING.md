# Notebook Testing in Docker

This directory contains Docker-based infrastructure for testing Jupyter notebooks in the PyGraphistry project.

## Overview

The notebook testing setup provides three different testing backends:
- **nbval**: Validates notebook execution and optionally checks outputs
- **nbmake**: Executes notebooks and reports failures
- **papermill**: Parameterized notebook execution with output capture

## Quick Start

### Run all notebook tests (default: nbval)
```bash
./docker/test-notebooks.sh
```

### Run specific notebook or directory
```bash
./docker/test-notebooks.sh --path demos/more_examples/graphistry_features
```

### Use a different testing backend
```bash
./docker/test-notebooks.sh --type nbmake
./docker/test-notebooks.sh --type papermill
```

### Run with GPU support
```bash
./docker/test-notebooks.sh --gpu
```

## Docker Compose Usage

```bash
# Run notebook tests using docker-compose
docker-compose run --rm test-notebooks

# With custom parameters
docker-compose run --rm \
  -e TEST_TYPE=nbmake \
  -e NOTEBOOK_PATH=demos/ai \
  -e TIMEOUT=600 \
  test-notebooks
```

## Configuration Options

### Command Line Options

- `--type TYPE`: Testing backend (nbval, nbmake, papermill)
- `--path PATH`: Path to notebooks (default: demos)
- `--timeout SECONDS`: Timeout per notebook (default: 300)
- `--parallel`: Enable parallel testing (default for nbval/nbmake)
- `--no-parallel`: Disable parallel testing
- `--build`: Force rebuild of Docker image
- `--gpu`: Use GPU-enabled image

### Environment Variables

- `TEST_TYPE`: Testing backend to use
- `NOTEBOOK_PATH`: Path to notebooks to test
- `TIMEOUT`: Timeout in seconds per notebook
- `PARALLEL`: Enable/disable parallel execution

## Test Groups

The `notebook-test-config.yml` file defines test groups for targeted testing:

```bash
# Run quick smoke tests
./docker/test-notebooks.sh --path demos/more_examples/graphistry_features --timeout 120

# Run core functionality tests
./docker/test-notebooks.sh --path demos/more_examples/graphistry_features --timeout 300

# Run AI/ML tests with longer timeout
./docker/test-notebooks.sh --path demos/ai --timeout 600
```

## Excluding Notebooks

Notebooks can be excluded from testing by:
1. Adding them to `skip_notebooks` in `notebook-test-config.yml`
2. Using naming conventions (e.g., `private_*.ipynb`, `manual_*.ipynb`)
3. Using pytest markers in the notebook metadata

## Integration with CI/CD

### GitHub Actions Example
```yaml
- name: Test Notebooks
  run: |
    ./docker/test-notebooks.sh --type nbval --parallel
```

### Local Development
```bash
# Test a specific notebook during development
./docker/test-notebooks.sh --path demos/my_notebook.ipynb --no-parallel

# Test with verbose output
./docker/test-notebooks.sh -- -vv

# Test and keep outputs
./docker/test-notebooks.sh --type papermill
```

## Troubleshooting

### Common Issues

1. **Timeout errors**: Increase timeout with `--timeout 600`
2. **Memory issues**: Disable parallel execution with `--no-parallel`
3. **GPU notebooks on CPU**: These will be automatically skipped
4. **Missing dependencies**: Rebuild image with `--build`

### Debugging Failed Notebooks

```bash
# Run with verbose output
./docker/test-notebooks.sh -- -vv

# Run a single notebook
./docker/test-notebooks.sh --path demos/specific_notebook.ipynb

# Keep intermediate outputs (papermill)
./docker/test-notebooks.sh --type papermill
# Check outputs in container at /tmp/papermill_output/
```

## Adding New Test Configurations

1. Edit `notebook-test-config.yml` to add new test groups or exclusions
2. Update timeout values for long-running notebooks
3. Add GPU-only notebooks to the appropriate section

## Maintenance

### Updating Testing Tools
Edit the version numbers in the Dockerfiles:
- `nbval==0.10.0`
- `nbmake==1.4.6`
- `papermill==2.4.0`

### Adding New Testing Backends
1. Update `test-notebooks-entrypoint.sh` with new case
2. Install required packages in Dockerfiles
3. Update documentation

## Best Practices

1. **Keep notebooks testable**: Avoid hardcoded paths and credentials
2. **Use cell tags**: Mark cells to skip during testing with `nbval-skip`
3. **Set reasonable timeouts**: Balance thoroughness with CI time
4. **Test incrementally**: Use test groups during development
5. **Document requirements**: Note special setup needs in notebooks
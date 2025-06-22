#!/bin/bash
set -e

# Activate virtual environment
source /opt/pygraphistry/venv/bin/activate

echo "=== Python Environment ==="
python --version
pip list | grep -E "nbval|nbmake|papermill|jupyter|graphistry"

# Default test parameters
TEST_TYPE=${TEST_TYPE:-"nbval"}
NOTEBOOK_PATH=${NOTEBOOK_PATH:-"demos"}
TIMEOUT=${TIMEOUT:-300}
PARALLEL=${PARALLEL:-"auto"}

echo "=== Notebook Testing Configuration ==="
echo "Test Type: $TEST_TYPE"
echo "Notebook Path: $NOTEBOOK_PATH"
echo "Timeout: $TIMEOUT seconds"
echo "Parallel: $PARALLEL"

case "$TEST_TYPE" in
    "nbval")
        echo "=== Running notebook validation with nbval ==="
        if [ "$PARALLEL" == "auto" ]; then
            pytest --nbval "$NOTEBOOK_PATH" \
                --nbval-lax \
                --nbval-timeout="$TIMEOUT" \
                -n auto \
                $@
        else
            pytest --nbval "$NOTEBOOK_PATH" \
                --nbval-lax \
                --nbval-timeout="$TIMEOUT" \
                $@
        fi
        ;;
    
    "nbmake")
        echo "=== Running notebook execution with nbmake ==="
        if [ "$PARALLEL" == "auto" ]; then
            pytest --nbmake "$NOTEBOOK_PATH" \
                --nbmake-timeout="$TIMEOUT" \
                --overwrite \
                -n auto \
                $@
        else
            pytest --nbmake "$NOTEBOOK_PATH" \
                --nbmake-timeout="$TIMEOUT" \
                --overwrite \
                $@
        fi
        ;;
    
    "papermill")
        echo "=== Running notebooks with papermill ==="
        # Find all notebooks and execute them with papermill
        find "$NOTEBOOK_PATH" -name "*.ipynb" -type f | while read notebook; do
            echo "Executing: $notebook"
            output_dir="/tmp/papermill_output/$(dirname "$notebook")"
            mkdir -p "$output_dir"
            output_file="$output_dir/$(basename "$notebook")"
            
            papermill "$notebook" "$output_file" \
                --cwd "$(dirname "$notebook")" \
                --execution-timeout "$TIMEOUT" \
                || echo "Failed: $notebook"
        done
        ;;
    
    *)
        echo "Unknown test type: $TEST_TYPE"
        echo "Valid options: nbval, nbmake, papermill"
        exit 1
        ;;
esac
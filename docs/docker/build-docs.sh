#!/bin/bash
set -ex
set -o pipefail

# Debug: Check if graphistry is importable
python3 -c "import sys; print('Python path:', sys.path); import graphistry; print('Graphistry imported successfully from:', graphistry.__file__)" || echo "WARNING: Cannot import graphistry"

# Validate RST syntax before building docs
if [ -x "/docs/validate-docs.sh" ]; then
    (cd /docs && ./validate-docs.sh)
else
    echo "ERROR: validate-docs.sh not found or not executable"
    exit 1
fi

# Validate GFQL doc code examples
echo "Running GFQL doc example audit..."
PYGRAPHISTRY_ROOT=/docs python3 -m pytest /docs/test_doc_examples.py -v --tb=short || {
    echo "WARNING: GFQL doc example audit found failures (non-blocking for now)"
}

build_html() {
    sphinx-build -b html -d /docs/doctrees . /docs/_build/html
}

build_epub() {
    sphinx-build -b epub -d /docs/doctrees . /docs/_build/epub
}

build_pdf() {
    sphinx-build -b latex -d /docs/doctrees . /docs/_build/latexpdf
    cd /docs/_build/latexpdf
    # Sphinx occasionally emits the same label twice in a row; collapse those before pdflatex.
    python3 - <<'PY'
import pathlib
import re

path = pathlib.Path("PyGraphistry.tex")
text = path.read_text()
text = re.sub(
    r'(\\label\{\\detokenize\{([^}]*)\}\})(?:\\label\{\\detokenize\{\2\}\})+',
    r'\1',
    text,
)
path.write_text(text)
PY
    rm -f PyGraphistry.aux PyGraphistry.out PyGraphistry.toc PyGraphistry.log

    # Keep rerunning pdflatex until references settle or a real LaTeX error stops the build.
    local max_passes=6
    local pass=1
    local log_file=/tmp/pygraphistry-pdflatex.log
    while [ "$pass" -le "$max_passes" ]; do
        echo "pdflatex pass $pass/$max_passes"
        local pdflatex_rc
        if pdflatex -file-line-error -interaction=nonstopmode PyGraphistry.tex | tee "$log_file"; then
            pdflatex_rc=0
        else
            pdflatex_rc=${PIPESTATUS[0]}
        fi

        if grep -q "undefined references" "$log_file"; then
            pass=$((pass + 1))
            continue
        fi

        if [ "$pdflatex_rc" -ne 0 ]; then
            echo "ERROR: pdflatex failed on pass $pass without an unresolved-reference retry condition"
            return "$pdflatex_rc"
        fi

        if ! grep -q "multiply-defined labels" "$log_file"; then
            return 0
        fi

        echo "ERROR: pdflatex reported multiply-defined labels"
        return 1
    done

    echo "ERROR: pdflatex still reports unresolved references after $max_passes passes"
    return 1
}

# Build docs first
case "$DOCS_FORMAT" in
    html)
        build_html
        ;;
    epub)
        build_epub
        ;;
    pdf)
        build_pdf
        ;;
    all)
        build_html
        build_epub
        build_pdf
        ;;
    *)
        echo "Invalid DOCS_FORMAT value: $DOCS_FORMAT"
        exit 1
        ;;
esac

# Validate notebooks after building docs
NOTEBOOKS_TO_VALIDATE=(
    "/docs/test_notebooks/test_graphistry_import.ipynb"
    "/docs/source/demos/gfql/temporal_predicates.ipynb"
    "/docs/source/gfql/hop_bounds.ipynb"
)

for notebook in "${NOTEBOOKS_TO_VALIDATE[@]}"; do
if [ -f "$notebook" ]; then
    echo "Validating $(basename $notebook) structure..."
    python3 -c "
import json
import sys

notebook_path = '$notebook'
with open(notebook_path, 'r') as f:
    nb = json.load(f)

# Check for missing execution_count in code cells
errors = []
for i, cell in enumerate(nb['cells']):
    if cell.get('cell_type') == 'code' and 'execution_count' not in cell:
        errors.append(f'Cell {i} missing execution_count')

if errors:
    print('Notebook validation errors:')
    for err in errors:
        print(f'  {err}')
    sys.exit(1)
else:
    print('Notebook structure validation passed')
"
    
    # Optionally execute notebook to verify it runs without errors
    if [ "${VALIDATE_NOTEBOOK_EXECUTION:-0}" = "1" ]; then
        echo "Executing $(basename $notebook) to verify it runs..."
        if ! python3 -m nbconvert --to notebook --execute \
            --ExecutePreprocessor.timeout=600 \
            --output /tmp/executed_notebook.ipynb \
            "$notebook"; then
            echo "ERROR: $(basename "$notebook") execution failed"
            exit 1
        fi
        rm -f /tmp/executed_notebook.ipynb
        echo "$(basename "$notebook") execution completed successfully"
    fi
fi
done

#!/bin/bash
set -ex

# Debug: Check if graphistry is importable
python3 -c "import sys; print('Python path:', sys.path); import graphistry; print('Graphistry imported successfully from:', graphistry.__file__)" || echo "WARNING: Cannot import graphistry"

build_html() {
    sphinx-build -b html -d /docs/doctrees . /docs/_build/html
}

build_epub() {
    sphinx-build -b epub -d /docs/doctrees . /docs/_build/epub
}

build_pdf() {
    sphinx-build -b latex -d /docs/doctrees . /docs/_build/latexpdf
    cd /docs/_build/latexpdf
    
    # Temporarily disable exit on error for pdflatex
    # LaTeX may exit with code 1 due to warnings (e.g., multiply-defined labels)
    # but still generate a valid PDF
    set +e
    
    pdflatex -file-line-error -interaction=nonstopmode PyGraphistry.tex
    FIRST_EXIT=$?
    
    pdflatex -file-line-error -interaction=nonstopmode PyGraphistry.tex
    SECOND_EXIT=$?
    
    # Re-enable exit on error
    set -e
    
    # If PDF was generated, consider it a success even if there were warnings
    if [ -f PyGraphistry.pdf ]; then
        if [ $SECOND_EXIT -ne 0 ]; then
            echo "WARNING: pdflatex exited with code $SECOND_EXIT but PDF was generated successfully"
        fi
        return 0
    else
        echo "ERROR: PDF generation failed"
        return 1
    fi
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
        python3 -m nbconvert --to notebook --execute \
            --ExecutePreprocessor.timeout=600 \
            --output /tmp/executed_notebook.ipynb \
            "$notebook" && \
        rm -f /tmp/executed_notebook.ipynb && \
        echo "$(basename $notebook) execution completed successfully"
    fi
fi
done
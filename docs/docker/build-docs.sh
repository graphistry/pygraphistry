#!/bin/sh
set -ex

# Build the HTML documentation incrementally
sphinx-build -b html -d /docs/doctrees . /docs/_build/html

# Build the EPUB documentation incrementally
sphinx-build -b epub -d /docs/doctrees . /docs/_build/epub

# Build the PDF documentation incrementally
sphinx-build -b latex -d /docs/doctrees . /docs/_build/latexpdf
cd /docs/_build/latexpdf

# Run pdflatex twice to resolve cross-references, using batchmode for non-interactive build
pdflatex -file-line-error -interaction=nonstopmode PyGraphistry.tex
pdflatex -file-line-error -interaction=nonstopmode PyGraphistry.tex

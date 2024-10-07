#!/bin/sh

# Build the HTML documentation incrementally
sphinx-build -b html -d /docs/doctrees . /docs/_build/html

# Build the EPUB documentation incrementally
sphinx-build -b epub -d /docs/doctrees . /docs/_build/epub

# Build the PDF documentation incrementally
sphinx-build -b latex -d /docs/doctrees . /docs/_build/latexpdf

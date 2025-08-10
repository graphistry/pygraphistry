#!/bin/bash
# validate-docs.sh - AI-friendly documentation validation script
# 
# Usage:
#   ./bin/validate-docs.sh           # Validate all docs
#   ./bin/validate-docs.sh path/to/file.rst  # Validate specific file
#   ./bin/validate-docs.sh --changed # Validate only changed files in git
#
# This script is designed to be easily used by AI assistants and developers
# during documentation development, before pushing to CI.

set -e

# Colors for output (disabled if not terminal)
if [ -t 1 ]; then
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[1;33m'
    NC='\033[0m' # No Color
else
    RED=''
    GREEN=''
    YELLOW=''
    NC=''
fi

echo_success() {
    echo -e "${GREEN}✓${NC} $1"
}

echo_error() {
    echo -e "${RED}✗${NC} $1"
}

echo_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

echo_info() {
    echo -e "ℹ $1"
}

# Check if rstcheck is installed
if ! command -v rstcheck &> /dev/null; then
    echo_error "rstcheck is not installed"
    echo_info "Install it with: pip install rstcheck[sphinx]"
    exit 1
fi

# Check if config file exists
CONFIG_FILE=".rstcheck.cfg"
if [ ! -f "$CONFIG_FILE" ]; then
    echo_warning "No .rstcheck.cfg found, creating default config..."
    cat > "$CONFIG_FILE" << 'EOF'
[rstcheck]
# Ignore Sphinx-specific roles that are not part of standard RST
ignore_roles = 
    meth,
    class,
    ref,
    doc,
    attr,
    mod,
    func,
    data,
    const,
    exc,
    obj,
    any

# Ignore Sphinx-specific directives
ignore_directives = 
    automodule,
    autoclass,
    autofunction,
    toctree,
    literalinclude,
    code-block,
    note,
    warning,
    versionadded,
    versionchanged,
    deprecated,
    seealso,
    rubric,
    centered,
    hlist,
    glossary,
    productionlist

# Ignore common informational messages
ignore_messages = (Hyperlink target "[^"]*" is not referenced\.$)

# Report level: ERROR, WARNING, INFO
report_level = WARNING
EOF
    echo_success "Created .rstcheck.cfg"
fi

# Determine what files to check
if [ "$1" == "--changed" ]; then
    # Check only changed RST files
    echo_info "Checking changed RST files..."
    FILES=$(git diff --name-only HEAD -- '*.rst' 2>/dev/null || true)
    if [ -z "$FILES" ]; then
        FILES=$(git diff --cached --name-only -- '*.rst' 2>/dev/null || true)
    fi
    if [ -z "$FILES" ]; then
        echo_info "No changed RST files to check"
        exit 0
    fi
elif [ -n "$1" ]; then
    # Check specific file(s) provided as arguments
    FILES="$@"
else
    # Check all RST files in docs/
    echo_info "Checking all RST files in docs/..."
    FILES=$(find docs -name "*.rst" 2>/dev/null || true)
    if [ -z "$FILES" ]; then
        echo_warning "No RST files found in docs/"
        exit 0
    fi
fi

# Count files
FILE_COUNT=$(echo "$FILES" | wc -w)
echo_info "Validating $FILE_COUNT RST file(s)..."

# Run rstcheck on each file
ERRORS_FOUND=0
WARNINGS_FOUND=0

for file in $FILES; do
    if [ ! -f "$file" ]; then
        echo_warning "File not found: $file"
        continue
    fi
    
    # Run rstcheck and capture output
    if OUTPUT=$(rstcheck --config "$CONFIG_FILE" "$file" 2>&1); then
        echo_success "$file"
    else
        # Parse output for errors vs warnings
        if echo "$OUTPUT" | grep -q "ERROR"; then
            echo_error "$file"
            echo "$OUTPUT" | grep "ERROR" | sed 's/^/  /'
            ERRORS_FOUND=$((ERRORS_FOUND + 1))
        elif echo "$OUTPUT" | grep -q "WARNING"; then
            echo_warning "$file"
            if [ "${VERBOSE:-0}" == "1" ]; then
                echo "$OUTPUT" | grep "WARNING" | head -5 | sed 's/^/  /'
            fi
            WARNINGS_FOUND=$((WARNINGS_FOUND + 1))
        else
            echo_error "$file (unknown issue)"
            echo "$OUTPUT" | head -5 | sed 's/^/  /'
            ERRORS_FOUND=$((ERRORS_FOUND + 1))
        fi
    fi
done

# Summary
echo ""
if [ $ERRORS_FOUND -eq 0 ] && [ $WARNINGS_FOUND -eq 0 ]; then
    echo_success "All RST files passed validation!"
    exit 0
elif [ $ERRORS_FOUND -eq 0 ]; then
    echo_warning "Validation completed with $WARNINGS_FOUND warning(s)"
    echo_info "Run with VERBOSE=1 to see warning details"
    exit 0
else
    echo_error "Validation failed with $ERRORS_FOUND error(s) and $WARNINGS_FOUND warning(s)"
    echo_info "Fix the errors above and run again"
    exit 1
fi
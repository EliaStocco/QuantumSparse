#!/usr/bin/env bash
set -e

# Remove existing pyproject.toml if it exists (even if read-only)
if [ -f pyproject.toml ]; then
    chmod u+w pyproject.toml || true
    rm pyproject.toml
fi

# Generate scripts section
python tools/generate_cli.py > tools/scripts.toml

# Rebuild pyproject.toml
cat tools/template.toml tools/scripts.toml > pyproject.toml

# Lock it down again
chmod 444 pyproject.toml

# Cleanup
rm tools/scripts.toml

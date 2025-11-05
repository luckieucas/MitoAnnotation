#!/bin/bash
# Setup script for just runtime directory
# This ensures just can run without permission errors

# Try to use /run/user/$UID/just first
if [ -n "$UID" ] && [ -d "/run/user/$UID" ]; then
    JUST_RUNTIME_DIR="/run/user/$UID/just"
    mkdir -p "$JUST_RUNTIME_DIR" 2>/dev/null && chmod 700 "$JUST_RUNTIME_DIR" 2>/dev/null
    if [ -w "$JUST_RUNTIME_DIR" ]; then
        export JUST_RUNTIME_DIR
        return 0
    fi
fi

# Fallback to project-local directory
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
JUST_RUNTIME_DIR="$PROJECT_DIR/.just_runtime"
mkdir -p "$JUST_RUNTIME_DIR" 2>/dev/null
chmod 700 "$JUST_RUNTIME_DIR" 2>/dev/null
export JUST_RUNTIME_DIR

echo "JUST_RUNTIME_DIR set to: $JUST_RUNTIME_DIR"




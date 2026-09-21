#!/bin/bash
set -euo pipefail

# Maki installer for macOS and Linux
# Usage: curl -fsSL https://raw.githubusercontent.com/BowlOfData/maki/main/install.sh | bash

REPO_URL="https://github.com/BowlOfData/maki.git"
REPO_REF="${MAKI_REF:-main}"
INSTALL_DIR="${MAKI_INSTALL_DIR:-$HOME/.maki}"
BIN_DIR="${MAKI_BIN_DIR:-$HOME/.local/bin}"
MIN_PYTHON_MAJOR=3
MIN_PYTHON_MINOR=10

info()    { printf '==> %s\n' "$1"; }
success() { printf '\033[32m==> %s\033[0m\n' "$1"; }
error()   { printf '\033[31mError: %s\033[0m\n' "$1" >&2; }

fail() {
    error "$1"
    exit 1
}

find_python() {
    local candidate
    for candidate in python3.13 python3.12 python3.11 python3.10 python3; do
        if command -v "$candidate" >/dev/null 2>&1; then
            local version
            version="$("$candidate" -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")' 2>/dev/null || true)"
            local major="${version%%.*}"
            local minor="${version##*.}"
            if [[ -n "$major" && -n "$minor" ]] \
                && [[ "$major" -eq "$MIN_PYTHON_MAJOR" ]] \
                && [[ "$minor" -ge "$MIN_PYTHON_MINOR" ]]; then
                echo "$candidate"
                return 0
            fi
        fi
    done
    return 1
}

command -v git >/dev/null 2>&1 || fail "git is required but was not found on PATH."

PYTHON_BIN="$(find_python)" \
    || fail "Python ${MIN_PYTHON_MAJOR}.${MIN_PYTHON_MINOR}+ is required but was not found on PATH."
info "Using $($PYTHON_BIN --version) at $(command -v "$PYTHON_BIN")"

if [[ -d "$INSTALL_DIR/repo/.git" ]]; then
    info "Updating existing checkout in $INSTALL_DIR/repo"
    git -C "$INSTALL_DIR/repo" fetch --depth 1 origin "$REPO_REF"
    git -C "$INSTALL_DIR/repo" checkout -q FETCH_HEAD
else
    info "Cloning $REPO_URL"
    rm -rf "$INSTALL_DIR/repo"
    mkdir -p "$INSTALL_DIR"
    git clone --depth 1 --branch "$REPO_REF" "$REPO_URL" "$INSTALL_DIR/repo"
fi

info "Creating virtual environment in $INSTALL_DIR/venv"
"$PYTHON_BIN" -m venv "$INSTALL_DIR/venv"
VENV_PIP="$INSTALL_DIR/venv/bin/pip"

info "Installing maki"
"$VENV_PIP" install --quiet --upgrade pip
"$VENV_PIP" install --quiet "$INSTALL_DIR/repo"

mkdir -p "$BIN_DIR"
ln -sf "$INSTALL_DIR/venv/bin/maki" "$BIN_DIR/maki"

success "maki installed to $INSTALL_DIR"

if [[ ":$PATH:" != *":$BIN_DIR:"* ]]; then
    info "Add $BIN_DIR to your PATH to use the 'maki' command, e.g.:"
    printf '    export PATH="%s:$PATH"\n' "$BIN_DIR"
else
    info "Run 'maki --help' to get started."
fi

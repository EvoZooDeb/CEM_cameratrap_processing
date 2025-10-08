#!/bin/sh
set -e

APP_USER="appuser"
APP_HOME="/home/${APP_USER}"
DEFAULT_CMD="felis"

# If container invoked without args, show CLI help
if [ "$#" -eq 0 ]; then
    set -- ${DEFAULT_CMD} --help
fi

# Allow shorthand subcommands (e.g. `docker run image predict ...`)
case "$1" in
    ${DEFAULT_CMD}|python|bash|sh|/bin/sh|/bin/bash|-*)
        ;;
    *)
        set -- ${DEFAULT_CMD} "$@"
        ;;
esac

TARGET_UID="${FELIS_UID:-}"
TARGET_GID="${FELIS_GID:-}"

resolve_from_path() {
    path="$1"
    [ -z "$path" ] && return
    if [ -z "$TARGET_UID" ] && [ -e "$path" ]; then
        uid=$(stat -c %u "$path" 2>/dev/null || true)
        gid=$(stat -c %g "$path" 2>/dev/null || true)
        if [ -n "$uid" ]; then
            TARGET_UID="$uid"
        fi
        if [ -n "$gid" ]; then
            TARGET_GID="$gid"
        fi
    fi
}

if [ -z "$TARGET_UID" ] || [ -z "$TARGET_GID" ]; then
    for candidate in "/work/results" "/work" "/app/results" "/app"; do
        resolve_from_path "$candidate"
        [ -n "$TARGET_UID" ] && [ -n "$TARGET_GID" ] && break
    done
fi

# Fallback to existing appuser identity when we cannot detect UID/GID
if [ -z "$TARGET_UID" ]; then
    TARGET_UID=$(id -u "$APP_USER")
fi
if [ -z "$TARGET_GID" ]; then
    TARGET_GID=$(id -g "$APP_USER")
fi

FORCE_ROOT=0
if [ "$TARGET_UID" -eq 0 ]; then
    FORCE_ROOT=1
fi
if [ "${FELIS_FORCE_ROOT:-}" = "1" ] || [ "${FELIS_RUN_AS:-}" = "root" ]; then
    FORCE_ROOT=1
fi

CURRENT_UID=$(id -u "$APP_USER")
CURRENT_GID=$(id -g "$APP_USER")

if [ "$FORCE_ROOT" -ne 1 ]; then
    if [ "$CURRENT_GID" -ne "$TARGET_GID" ]; then
        if groupmod -g "$TARGET_GID" "$APP_USER" 2>/dev/null; then
            :
        else
            EXISTING_GROUP=$(getent group "$TARGET_GID" | cut -d: -f1 || true)
            if [ -n "$EXISTING_GROUP" ]; then
                usermod -g "$EXISTING_GROUP" "$APP_USER"
            else
                groupadd -g "$TARGET_GID" felis
                usermod -g felis "$APP_USER"
            fi
        fi
    fi

    if [ "$CURRENT_UID" -ne "$TARGET_UID" ]; then
        usermod -u "$TARGET_UID" "$APP_USER"
    fi

    # Ensure home directory matches new IDs when home exists
    if [ -d "$APP_HOME" ]; then
        chown -R "$TARGET_UID":"$TARGET_GID" "$APP_HOME"
    fi

    if [ -d "/work/raw" ]; then
        if ! gosu "$APP_USER" find /work/raw -maxdepth 5 -mindepth 0 >/dev/null 2>&1; then
            FORCE_ROOT=1
        fi
    fi
fi

if [ "$FORCE_ROOT" -eq 1 ]; then
    "$@"
    status=$?
    if [ "$TARGET_UID" -ne 0 ] && [ -d "/work/results" ]; then
        chown -R "$TARGET_UID":"$TARGET_GID" /work/results 2>/dev/null || true
    fi
    exit $status
fi

exec gosu "$APP_USER" "$@"

#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_DIR="${MARIADB_RUNTIME_DIR:-${PROJECT_DIR}/../hackathon-tools/runtime/mariadb-12.3.3-linux-systemd-x86_64}"
COMPAT_DIR="${PROJECT_DIR}/../hackathon-tools/runtime/compat/usr/lib/x86_64-linux-gnu"
if [[ ! -x "${RUNTIME_DIR}/bin/mariadbd" ]]; then
  echo "Set MARIADB_RUNTIME_DIR to an extracted MariaDB 11.7+ binary installation." >&2
  exit 1
fi
if [[ ! -d "${PROJECT_DIR}/data/mariadb/mysql" ]]; then
  echo "Initialize data/mariadb with mariadb-install-db before starting the server." >&2
  exit 1
fi
export LD_LIBRARY_PATH="${COMPAT_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
exec "${RUNTIME_DIR}/bin/mariadbd" --no-defaults \
  --basedir="${RUNTIME_DIR}" --datadir="${PROJECT_DIR}/data/mariadb" \
  --socket=/tmp/junction-mariadb.sock \
  --pid-file="${PROJECT_DIR}/data/mariadb/server.pid" \
  --port=3307 --bind-address=127.0.0.1 --innodb-buffer-pool-size=256M \
  --max-allowed-packet=64M --log-error="${PROJECT_DIR}/data/mariadb/server.log"

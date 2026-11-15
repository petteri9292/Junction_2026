"""Set up a project-local MariaDB in the disposable Hugging Face workspace."""

import hashlib
import os
import secrets
import subprocess
import tarfile
import time
from pathlib import Path

import pymysql
import requests
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]
VERSION = "12.3.3"
NAME = f"mariadb-{VERSION}-linux-systemd-x86_64"
SHA256 = "a12bf7bce08a896cdd01f174f04b4ce1d5da43ec250a5e7fceb9972119514329"
RUNTIME = ROOT / "data" / "runtime"
BASE = RUNTIME / NAME
DATA = ROOT / "data" / "mariadb"
SOCKET = "/tmp/junction-mariadb.sock"


def main():
    if not str(ROOT).startswith("/workspace/"):
        raise RuntimeError("This installer is for the remote /workspace project only")
    RUNTIME.mkdir(parents=True, exist_ok=True)
    archive = RUNTIME / f"{NAME}.tar.gz"
    if not archive.exists():
        url = f"https://archive.mariadb.org/mariadb-{VERSION}/bintar-linux-systemd-x86_64/{archive.name}"
        temporary = archive.with_suffix(".part")
        with requests.get(url, stream=True, timeout=120) as response:
            response.raise_for_status()
            with temporary.open("wb") as output:
                for chunk in response.iter_content(4 * 1024 * 1024):
                    output.write(chunk)
        temporary.replace(archive)
    with archive.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != SHA256:
            raise ValueError("MariaDB archive checksum mismatch")
    print("MariaDB archive SHA-256 verified", flush=True)
    if not (BASE / "bin/mariadbd").exists():
        with tarfile.open(archive) as source:
            members = [
                m
                for m in source.getmembers()
                if any(
                    "/" + part in m.name
                    for part in ("bin/", "scripts/", "share/", "lib/")
                )
                and "/mysql-test/" not in m.name
            ]
            source.extractall(RUNTIME, members=members, filter="data")
    compat = RUNTIME / "compat"
    compat.mkdir(exist_ok=True)
    # Ubuntu 24.04 names this ABI-compatible library libaio.so.1t64.
    aio = Path("/usr/lib/x86_64-linux-gnu/libaio.so.1t64")
    if aio.exists() and not (compat / "libaio.so.1").exists():
        (compat / "libaio.so.1").symlink_to(aio)
    env = dict(os.environ, LD_LIBRARY_PATH=str(compat))
    DATA.mkdir(parents=True, exist_ok=True)
    if not (DATA / "mysql").exists():
        subprocess.run(
            [
                str(BASE / "scripts/mariadb-install-db"),
                "--no-defaults",
                f"--basedir={BASE}",
                f"--datadir={DATA}",
                "--user=root",
                "--auth-root-authentication-method=socket",
                "--skip-test-db",
            ],
            env=env,
            check=True,
            stdout=subprocess.DEVNULL,
        )
    config = DATA / "remote.cnf"
    config.write_text(f"""[mariadbd]
user=root
basedir={BASE}
datadir={DATA}
socket={SOCKET}
pid-file={DATA / "server.pid"}
port=3307
bind-address=127.0.0.1
innodb-buffer-pool-size=512M
max-allowed-packet=64M
log-error={DATA / "server.log"}
""")
    try:
        admin = pymysql.connect(unix_socket=SOCKET, user="root", connect_timeout=2)
    except pymysql.Error:
        with (DATA / "launcher.log").open("a") as log:
            subprocess.Popen(
                [str(BASE / "bin/mariadbd"), f"--defaults-file={config}"],
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=log,
                start_new_session=True,
            )
        for _ in range(60):
            try:
                admin = pymysql.connect(
                    unix_socket=SOCKET, user="root", connect_timeout=2
                )
                break
            except pymysql.Error:
                time.sleep(1)
        else:
            raise RuntimeError("MariaDB did not start; inspect data/mariadb/server.log")
    env_file = ROOT / ".env"
    settings = dotenv_values(env_file) if env_file.exists() else {}
    password = settings.get("DB_PASSWORD") or secrets.token_urlsafe(32)
    with admin, admin.cursor() as cursor:
        cursor.execute(
            "CREATE DATABASE IF NOT EXISTS wikilens CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
        )
        cursor.execute(
            "CREATE USER IF NOT EXISTS %s@%s IDENTIFIED BY %s",
            ("wikilens", "127.0.0.1", password),
        )
        cursor.execute(
            "ALTER USER %s@%s IDENTIFIED BY %s", ("wikilens", "127.0.0.1", password)
        )
        cursor.execute("GRANT ALL PRIVILEGES ON wikilens.* TO 'wikilens'@'127.0.0.1'")
        admin.commit()
    if not env_file.exists():
        descriptor = os.open(env_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w") as output:
            output.write(
                "DB_HOST=127.0.0.1\nDB_PORT=3307\nDB_NAME=wikilens\nDB_USER=wikilens\nDB_PASSWORD="
                + password
                + "\n"
            )
    from wikilens.database import connect, initialize

    initialize()
    with connect() as connection, connection.cursor() as cursor:
        cursor.execute(
            "SELECT VERSION() AS version, VEC_DISTANCE_COSINE(VEC_FromText('[1,0,0]'), VEC_FromText('[1,0,0]')) AS self_distance"
        )
        print("Native vector check:", cursor.fetchone(), flush=True)
        cursor.execute("SHOW CREATE TABLE article_vector")
        definition = str(cursor.fetchone()).upper()
        assert "VECTOR KEY" in definition or "VECTOR INDEX" in definition
        for table in ("article", "article_vector"):
            cursor.execute(f"SELECT COUNT(*) AS rows_present FROM {table}")
            print(table, cursor.fetchone(), flush=True)
    print(
        "Ready: dedicated database, native cosine index, credentials in ignored .env. No dataset import performed.",
        flush=True,
    )


if __name__ == "__main__":
    main()

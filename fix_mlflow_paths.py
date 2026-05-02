"""
fix_mlflow_paths.py
--------------------
Rewrites all Windows-style absolute paths stored in mlflow.db to the
container-internal path (/app/mlruns).

This is needed because the DB was created on Windows (where artifact paths
look like  file:///C:/Users/.../mlruns/...)  but runs inside a Linux Docker
container where the mlruns/ folder is mounted at /app/mlruns.

The script is fully idempotent – safe to run multiple times.
"""

import sqlite3
import re
import sys
import os

DB_PATH = os.environ.get("MLFLOW_DB_PATH", "/app/mlflow.db")
CONTAINER_MLRUNS = "/app/mlruns"

# Pattern that matches Windows-style file URIs for mlruns, e.g.:
#   file:///C:/Users/tb619/.../mlruns
# We capture everything after /mlruns so we can rebuild the path correctly.
WINDOWS_URI_PATTERN = re.compile(
    r"file:///[A-Za-z]:[/\\].*?/mlruns(.*)", re.IGNORECASE
)

def fix_uri(value: str) -> str:
    """Replace a Windows file URI with the container-local equivalent."""
    if value is None:
        return value
    m = WINDOWS_URI_PATTERN.match(value)
    if m:
        suffix = m.group(1).replace("\\", "/")
        return f"file://{CONTAINER_MLRUNS}{suffix}"
    return value

def migrate(conn: sqlite3.Connection):
    cur = conn.cursor()
    changed = 0

    # 1. experiments.artifact_location
    cur.execute("SELECT experiment_id, artifact_location FROM experiments")
    for exp_id, loc in cur.fetchall():
        new_loc = fix_uri(loc)
        if new_loc != loc:
            cur.execute(
                "UPDATE experiments SET artifact_location=? WHERE experiment_id=?",
                (new_loc, exp_id)
            )
            print(f"  [experiment {exp_id}] {loc!r}\n    -> {new_loc!r}")
            changed += 1

    # 2. runs.artifact_uri
    cur.execute("SELECT run_uuid, artifact_uri FROM runs")
    for run_id, uri in cur.fetchall():
        new_uri = fix_uri(uri)
        if new_uri != uri:
            cur.execute(
                "UPDATE runs SET artifact_uri=? WHERE run_uuid=?",
                (new_uri, run_id)
            )
            changed += 1

    # 3. logged_models.artifact_location
    cur.execute("SELECT model_id, artifact_location FROM logged_models")
    for model_id, loc in cur.fetchall():
        new_loc = fix_uri(loc)
        if new_loc != loc:
            cur.execute(
                "UPDATE logged_models SET artifact_location=? WHERE model_id=?",
                (new_loc, model_id)
            )
            changed += 1
            
    # 4. model_versions.storage_location and source
    cur.execute("SELECT name, version, storage_location, source FROM model_versions")
    for name, version, storage, source in cur.fetchall():
        new_storage = fix_uri(storage)
        new_source = fix_uri(source)
        if new_storage != storage or new_source != source:
            cur.execute(
                "UPDATE model_versions SET storage_location=?, source=? WHERE name=? AND version=?",
                (new_storage, new_source, name, version)
            )
            print(f"  [model_version {name} v{version}] storage updated")
            changed += 1

    conn.commit()
    return changed


if __name__ == "__main__":
    if not os.path.exists(DB_PATH):
        print(f"[fix_mlflow_paths] DB not found at {DB_PATH}, skipping.")
        sys.exit(0)

    print(f"[fix_mlflow_paths] Patching Windows paths in {DB_PATH} ...")
    conn = sqlite3.connect(DB_PATH)
    try:
        n = migrate(conn)
        print(f"[fix_mlflow_paths] Done. {n} row(s) updated.")
    finally:
        conn.close()

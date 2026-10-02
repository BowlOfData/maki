"""
Tests for the PostgresStore plugin.

Unit tests need no server. Integration tests run only when MAKI_PG_HOST is set
(they use a throwaway schema that is dropped afterwards).
"""

import os
import uuid

import pytest

from maki.plugins.postgres_store import PostgresStore, PostgresStoreError

CA = os.environ.get("MAKI_PG_SSLROOTCERT")


def _cfg(**over):
    base = dict(host="db.example.com", dbname="d", user="u", password="p", sslmode="verify-full")
    base.update(over)
    return base


def test_no_method_is_llm_callable():
    assert PostgresStore.ALLOWED_METHODS == []


@pytest.mark.parametrize("mode", ["disable", "allow", "prefer", "require"])
def test_weak_sslmode_refused_for_remote_host(mode):
    with pytest.raises(PostgresStoreError):
        PostgresStore(**_cfg(sslmode=mode))


def test_weak_sslmode_allowed_on_loopback():
    PostgresStore(**_cfg(host="localhost", sslmode="disable"))


def test_missing_config_refused(monkeypatch):
    for k in ("HOST", "DB", "USER"):
        monkeypatch.delenv(f"MAKI_PG_{k}", raising=False)
    with pytest.raises(PostgresStoreError):
        PostgresStore()


def test_missing_ca_file_refused():
    with pytest.raises(PostgresStoreError):
        PostgresStore(**_cfg(sslrootcert="/nonexistent/ca.pem"))


def test_pool_is_capped():
    assert PostgresStore(**_cfg(max_connections=99))._max_size == 10


def test_password_not_in_repr():
    assert "p-secret" not in repr(PostgresStore(**_cfg(password="p-secret")))


def test_bad_identifier_refused():
    with pytest.raises(PostgresStoreError):
        PostgresStore(**_cfg(schema="x; DROP TABLE y"))


requires_db = pytest.mark.skipif(not os.environ.get("MAKI_PG_HOST"), reason="MAKI_PG_HOST not set")


@pytest.fixture
def store():
    schema = f"maki_test_{uuid.uuid4().hex[:10]}"
    admin = PostgresStore(max_connections=1)
    admin.execute(f"CREATE SCHEMA {schema}")
    s = PostgresStore(max_connections=2, schema=schema)
    try:
        yield s
    finally:
        s.close()
        admin.execute(f"DROP SCHEMA {schema} CASCADE")
        admin.close()


@requires_db
def test_ping_and_tls():
    with PostgresStore(max_connections=1) as s:
        assert s.ping()
        assert s.query_one("SELECT ssl FROM pg_stat_ssl WHERE pid = pg_backend_pid()")["ssl"] is True


@requires_db
def test_params_are_bound_not_interpolated(store):
    store.execute("CREATE TABLE t (id int PRIMARY KEY, v text)")
    evil = "x'); DROP TABLE t; --"
    store.execute("INSERT INTO t VALUES (%s, %s)", (1, evil))
    assert store.query("SELECT v FROM t") == [{"v": evil}]


@requires_db
def test_transaction_rolls_back_on_error(store):
    store.execute("CREATE TABLE t (id int PRIMARY KEY)")
    with pytest.raises(PostgresStoreError):
        with store.transaction() as conn:
            conn.execute("INSERT INTO t VALUES (1)")
            raise RuntimeError("boom")
    assert store.query("SELECT * FROM t") == []


@requires_db
def test_executemany_and_upsert_idempotent(store):
    store.execute("CREATE TABLE t (k text PRIMARY KEY, n int)")
    sql = "INSERT INTO t VALUES (%s, %s) ON CONFLICT (k) DO UPDATE SET n = EXCLUDED.n"
    store.executemany(sql, [("a", 1), ("b", 2)])
    store.executemany(sql, [("a", 1), ("b", 3)])
    assert store.query("SELECT k, n FROM t ORDER BY k") == [{"k": "a", "n": 1}, {"k": "b", "n": 3}]


@requires_db
def test_migrate_is_idempotent_and_atomic(store):
    migs = [(1, "CREATE TABLE a (id int)"), (2, "CREATE TABLE b (id int)")]
    assert store.migrate(migs) == [1, 2]
    assert store.migrate(migs) == []
    bad = migs + [(3, "CREATE TABLE c (id int); SELECT nope FROM missing")]
    with pytest.raises(PostgresStoreError):
        store.migrate(bad)
    assert [r["version"] for r in store.query("SELECT version FROM schema_version ORDER BY 1")] == [1, 2]
    assert store.query("SELECT to_regclass('c') AS t")[0]["t"] is None


@requires_db
def test_statement_timeout(store):
    s = PostgresStore(max_connections=1, schema=store._schema, statement_timeout_ms=200)
    try:
        with pytest.raises(PostgresStoreError):
            s.query("SELECT pg_sleep(2)")
    finally:
        s.close()

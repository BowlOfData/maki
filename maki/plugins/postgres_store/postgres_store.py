"""
Postgres Store Plugin for Maki Framework

A generic PostgreSQL client: pooled connections, enforced certificate
verification, parameterised queries, transactions and a migration runner.
Domain schemas belong to the calling application, not to this plugin.

Security model
--------------
* ``ALLOWED_METHODS`` exposes only the harmless ``ping`` health check to
  LLM ``TOOL:`` directives. SQL (``query``/``execute``/``migrate``) runs
  only from application code.
* The host comes from operator configuration (arguments or ``MAKI_PG_*``
  environment variables), never from model or feed content.
* TLS must verify the server certificate (``verify-ca`` or ``verify-full``).
  Weaker modes are accepted only for a loopback host (local development).
* The password is never logged and never appears in ``repr()``.
* Values are always passed as bind parameters (``%s`` placeholders).

Requires: ``pip install "psycopg[binary]" psycopg_pool``
"""

import logging
import os
import re
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

from maki.exceptions import MakiError

_VERIFYING_SSLMODES = ("verify-ca", "verify-full")
_LOOPBACK_HOSTS = ("localhost", "127.0.0.1", "::1")
_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,62}$")
# Aiven-style managed servers cap connections low; never let a caller starve them.
_MAX_POOL_CEILING = 10
# Arbitrary constant used as the transaction-level advisory lock id for migrations.
_MIGRATION_LOCK_ID = 0x4D414B49  # "MAKI"


class PostgresStoreError(MakiError):
    """Raised for configuration, connection and query failures."""


def _require_driver():
    try:
        import psycopg  # noqa: F401
        from psycopg.rows import dict_row  # noqa: F401
        from psycopg_pool import ConnectionPool  # noqa: F401
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise PostgresStoreError(
            'PostgresStore needs the drivers: pip install "psycopg[binary]" psycopg_pool'
        ) from exc


def _ident(name: str) -> str:
    if not isinstance(name, str) or not _IDENT.match(name):
        raise PostgresStoreError(f"Invalid SQL identifier: {name!r}")
    return name


class PostgresStore:
    """Pooled PostgreSQL client with enforced TLS verification."""

    # Only the read-only health check is exposed to LLM tool calls.
    ALLOWED_METHODS: List[str] = ["ping"]

    def __init__(
        self,
        maki_instance=None,
        host: Optional[str] = None,
        port: Optional[int] = None,
        dbname: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        sslmode: Optional[str] = None,
        sslrootcert: Optional[str] = None,
        max_connections: Optional[int] = None,
        connect_timeout: float = 10.0,
        statement_timeout_ms: int = 30000,
        schema: Optional[str] = None,
        env_prefix: str = "MAKI_PG_",
    ):
        self.maki = maki_instance
        self.logger = logging.getLogger(__name__)

        def _cfg(value, key, default=None):
            if value is not None:
                return value
            return os.environ.get(f"{env_prefix}{key}", default)

        self._host = _cfg(host, "HOST")
        self._port = int(_cfg(port, "PORT", 5432))
        self._dbname = _cfg(dbname, "DB")
        self._user = _cfg(user, "USER")
        self._password = _cfg(password, "PASSWORD")
        self._sslmode = _cfg(sslmode, "SSLMODE", "verify-full")
        self._sslrootcert = _cfg(sslrootcert, "SSLROOTCERT")
        requested = int(_cfg(max_connections, "MAX_CONNECTIONS", 3))
        self._max_size = max(1, min(requested, _MAX_POOL_CEILING))
        self._connect_timeout = connect_timeout
        self._statement_timeout_ms = int(statement_timeout_ms)
        self._schema = _ident(schema) if schema else None
        self._pool = None

        missing = [
            n for n, v in (("host", self._host), ("dbname", self._dbname),
                           ("user", self._user)) if not v
        ]
        if missing:
            raise PostgresStoreError(f"Missing Postgres configuration: {', '.join(missing)}")

        loopback = self._host in _LOOPBACK_HOSTS
        if self._sslmode not in _VERIFYING_SSLMODES and not loopback:
            raise PostgresStoreError(
                f"sslmode={self._sslmode!r} is refused for a remote host; "
                f"use one of {_VERIFYING_SSLMODES}"
            )
        if self._sslmode in _VERIFYING_SSLMODES and not self._sslrootcert:
            self.logger.info("No CA file given; relying on the system trust store")
        if self._sslrootcert and not os.path.isfile(self._sslrootcert):
            raise PostgresStoreError(f"CA certificate not found: {self._sslrootcert}")

        self.logger.info(
            "PostgresStore configured (host=%s db=%s user=%s sslmode=%s pool<=%d)",
            self._host, self._dbname, self._user, self._sslmode, self._max_size,
        )

    def __repr__(self) -> str:
        return (f"PostgresStore(host={self._host!r}, dbname={self._dbname!r}, "
                f"user={self._user!r}, sslmode={self._sslmode!r})")

    # -- pool -------------------------------------------------------------

    def _conninfo_kwargs(self) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "host": self._host,
            "port": self._port,
            "dbname": self._dbname,
            "user": self._user,
            "sslmode": self._sslmode,
            "connect_timeout": int(self._connect_timeout),
            "options": f"-c statement_timeout={self._statement_timeout_ms}"
            + (f" -c search_path={self._schema}" if self._schema else ""),
        }
        if self._password:
            kwargs["password"] = self._password
        if self._sslrootcert:
            kwargs["sslrootcert"] = self._sslrootcert
        return kwargs

    def _get_pool(self):
        if self._pool is None:
            _require_driver()
            from psycopg.rows import dict_row
            from psycopg_pool import ConnectionPool

            self._pool = ConnectionPool(
                conninfo="",
                kwargs={**self._conninfo_kwargs(), "row_factory": dict_row},
                min_size=1,
                max_size=self._max_size,
                open=False,
                timeout=self._connect_timeout,
            )
            try:
                self._pool.open(wait=True, timeout=self._connect_timeout)
            except Exception as exc:
                self._pool = None
                raise PostgresStoreError(f"Could not connect to Postgres: {self._describe(exc)}") from exc
        return self._pool

    @staticmethod
    def _describe(exc: Exception) -> str:
        # Class name plus the driver message; the password is never part of either.
        return f"{type(exc).__name__}: {exc}"

    def close(self) -> None:
        pool, self._pool = self._pool, None
        if pool is not None:
            pool.close()

    def __enter__(self) -> "PostgresStore":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    # -- queries ----------------------------------------------------------

    @contextmanager
    def transaction(self) -> Iterator[Any]:
        """Yield a connection inside one transaction (commit, or rollback on error)."""
        pool = self._get_pool()
        try:
            with pool.connection() as conn:
                yield conn
        except PostgresStoreError:
            raise
        except Exception as exc:
            raise PostgresStoreError(self._describe(exc)) from exc

    def query(self, sql: str, params: Optional[Sequence[Any]] = None) -> List[Dict[str, Any]]:
        """Run a statement and return all rows as dicts (empty list if none)."""
        with self.transaction() as conn:
            cur = conn.execute(sql, params)
            return cur.fetchall() if cur.description else []

    def query_one(self, sql: str, params: Optional[Sequence[Any]] = None) -> Optional[Dict[str, Any]]:
        rows = self.query(sql, params)
        return rows[0] if rows else None

    def execute(self, sql: str, params: Optional[Sequence[Any]] = None) -> int:
        """Run a statement and return the affected row count."""
        with self.transaction() as conn:
            return conn.execute(sql, params).rowcount

    def executemany(self, sql: str, rows: Sequence[Sequence[Any]]) -> int:
        """Run one statement for many parameter sets in a single transaction."""
        if not rows:
            return 0
        with self.transaction() as conn:
            with conn.cursor() as cur:
                cur.executemany(sql, rows)
                return cur.rowcount

    def ping(self) -> bool:
        try:
            return self.query_one("SELECT 1 AS ok") == {"ok": 1}
        except PostgresStoreError:
            return False

    # -- migrations -------------------------------------------------------

    def migrate(
        self,
        migrations: Sequence[Tuple[int, str]],
        table: str = "schema_version",
    ) -> List[int]:
        """Apply pending ``(version, sql)`` migrations in order, once each.

        A transaction-scoped advisory lock serialises concurrent runners.
        Each migration and its version row commit together. Returns the
        versions applied by this call (empty when already up to date).
        """
        _ident(table)
        ordered = sorted(migrations, key=lambda m: m[0])
        versions = [v for v, _ in ordered]
        if len(set(versions)) != len(versions):
            raise PostgresStoreError("Duplicate migration versions")
        applied: List[int] = []
        with self.transaction() as conn:
            conn.execute("SELECT pg_advisory_xact_lock(%s)", (_MIGRATION_LOCK_ID,))
            conn.execute(
                f"CREATE TABLE IF NOT EXISTS {table} ("
                "version integer PRIMARY KEY, applied_at timestamptz NOT NULL DEFAULT now())"
            )
            done = {r["version"] for r in conn.execute(f"SELECT version FROM {table}").fetchall()}
            for version, sql in ordered:
                if version in done:
                    continue
                conn.execute(sql)
                conn.execute(f"INSERT INTO {table} (version) VALUES (%s)", (version,))
                applied.append(version)
        if applied:
            self.logger.info("Applied migrations: %s", applied)
        return applied


def register_plugin(maki_instance=None, **kwargs):
    """Create a PostgresStore from keyword arguments / MAKI_PG_* environment."""
    return PostgresStore(maki_instance, **kwargs)

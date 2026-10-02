"""
Postgres Store Plugin for Maki Framework

Generic, hardened PostgreSQL access (TLS-verified connection, small pool,
parameterised queries, transactions, migrations). Contains no domain schema.
"""

from .postgres_store import PostgresStore, PostgresStoreError, register_plugin

__all__ = ["PostgresStore", "PostgresStoreError", "register_plugin"]

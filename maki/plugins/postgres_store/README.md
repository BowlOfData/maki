# postgres_store

Generic PostgreSQL client for Maki: TLS-verified pooled connections, parameterised
queries, transactions and a migration runner. No domain schema lives here.

```
pip install "maki-framework[postgres]"
```

Configuration comes from arguments or `MAKI_PG_*` environment variables
(`HOST PORT DB USER PASSWORD SSLMODE SSLROOTCERT MAX_CONNECTIONS`).

```python
from maki.plugins.postgres_store import PostgresStore

with PostgresStore() as db:
    db.migrate([(1, "CREATE TABLE IF NOT EXISTS t (id int primary key)")])
    db.execute("INSERT INTO t VALUES (%s) ON CONFLICT DO NOTHING", (1,))
    rows = db.query("SELECT * FROM t")
```

## Guarantees

- `ALLOWED_METHODS` is empty: nothing is callable from an LLM `TOOL:` directive.
- `sslmode` must be `verify-ca` or `verify-full` unless the host is loopback.
- Pool size is capped at 10 whatever is requested; managed servers allow few connections.
- Values are bind parameters only. Identifiers (schema, migration table) are validated.
- The password is never logged or shown in `repr()`.
- `migrate()` takes an advisory lock; each migration and its version row commit atomically.

## Tests

`pytest maki/plugins/postgres_store`. Integration tests need `MAKI_PG_HOST` etc. and
create and drop a throwaway schema.

"""
postgres_store plugin — usage examples.

Run with:  python -m maki.plugins.postgres_store.example_usage

Requires a reachable PostgreSQL server and:  pip install "psycopg[binary]" psycopg_pool
Connection settings come from MAKI_PG_HOST, MAKI_PG_DB, MAKI_PG_USER,
MAKI_PG_PASSWORD, MAKI_PG_SSLMODE and MAKI_PG_SSLROOTCERT.
"""

from maki.plugins.postgres_store import PostgresStore, PostgresStoreError


def main():
    try:
        store = PostgresStore()  # reads MAKI_PG_* environment variables
    except PostgresStoreError as exc:
        print(f"Configuration error: {exc}")
        return

    try:
        print("Reachable:", store.ping())

        store.execute("CREATE TABLE IF NOT EXISTS demo_notes (id serial PRIMARY KEY, body text)")
        store.execute("INSERT INTO demo_notes (body) VALUES (%s)", ["hello"])
        print(store.query("SELECT id, body FROM demo_notes ORDER BY id DESC LIMIT 5"))

        with store.transaction():
            store.execute("DELETE FROM demo_notes WHERE body = %s", ["hello"])
    finally:
        store.close()


if __name__ == "__main__":
    main()

from pathlib import Path
from typing import Literal
import sqlite3
import pandas as pd

#
# Functions for merging databases
#

def quote_identifier(name):
    """Safely quote an SQLite table or column identifier."""
    return '"' + name.replace('"', '""') + '"'


def merge_individual_databases(source_paths, target_path):
    """
    Merge `individual` tables from several SQLite databases into one database.

    A `source_db` column is added to the merged table to preserve the original
    run/database name. Integer primary-key values are not copied, so SQLite
    assigns unique new IDs in the combined database.

    Parameters
    ----------
    source_paths : list[str | Path]
        Paths to the input .db files.
    target_path : str | Path
        Path for the newly created combined database.
    """
    source_paths = [Path(path) for path in source_paths]
    target_path = Path(target_path)

    if not source_paths:
        raise ValueError("No source databases were supplied.")

    if target_path.exists():
        print(f"File {target_path} already exists")
        return

    # Obtain the individual-table schema from the first database.
    with sqlite3.connect(source_paths[0]) as source_connection:
        schema_row = source_connection.execute("""
            SELECT sql
            FROM sqlite_master
            WHERE type = 'table' AND name = 'individual'
        """).fetchone()

        if schema_row is None:
            raise ValueError(
                f"No table named 'individual' was found in {source_paths[0]}"
            )

        create_individual_sql = schema_row[0]

        # Columns: cid, name, type, notnull, default_value, pk
        table_info = source_connection.execute(
            "PRAGMA table_info(individual)"
        ).fetchall()

    # Do not copy an INTEGER PRIMARY KEY. This prevents ID collisions between
    # separate source databases; SQLite creates a fresh ID on insertion.
    columns_to_copy = []

    for _, column_name, column_type, _, _, is_primary_key in table_info:
        is_integer_primary_key = (
            is_primary_key > 0
            and "INT" in (column_type or "").upper()
        )

        if not is_integer_primary_key:
            columns_to_copy.append(column_name)

    if not columns_to_copy:
        raise ValueError("No copyable columns were found in `individual`.")

    quoted_columns = ", ".join(
        quote_identifier(column)
        for column in columns_to_copy
    )

    with sqlite3.connect(target_path) as target_connection:
        # Wait for up to five seconds if a temporary SQLite lock occurs.
        target_connection.execute("PRAGMA busy_timeout = 5000")

        # Set up the target database.
        target_connection.execute(create_individual_sql)

        target_connection.execute("""
            ALTER TABLE individual
            ADD COLUMN source_db TEXT
        """)

        # Commit schema changes before attaching source databases.
        target_connection.commit()

        for index, source_path in enumerate(source_paths):
            alias = f"source_{index}"

            target_connection.execute(
                f"ATTACH DATABASE ? AS {quote_identifier(alias)}",
                (str(source_path),)
            )

            try:
                # Verify that this source has all expected columns.
                source_columns = {
                    row[1]
                    for row in target_connection.execute(
                        f"PRAGMA {quote_identifier(alias)}.table_info(individual)"
                    )
                }

                missing_columns = set(columns_to_copy) - source_columns

                if missing_columns:
                    raise ValueError(
                        f"{source_path} is missing columns: "
                        f"{sorted(missing_columns)}"
                    )

                # Copy individuals. Their original integer primary-key IDs are
                # deliberately omitted, preventing collisions across databases.
                target_connection.execute(f"""
                    INSERT INTO individual ({quoted_columns}, source_db)
                    SELECT {quoted_columns}, ?
                    FROM {quote_identifier(alias)}.individual
                """, (source_path.name,))

                # A commit is required before DETACH DATABASE.
                target_connection.commit()

            finally:
                target_connection.execute(
                    f"DETACH DATABASE {quote_identifier(alias)}"
                )

        target_connection.commit()

    print(f"Merged {len(source_paths)} databases into: {target_path}")
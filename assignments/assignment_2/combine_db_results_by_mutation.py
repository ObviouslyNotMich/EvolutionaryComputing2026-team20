"""Combine db_<mutation>_<seed>.db (or .db.zip) runs by mutation."""

import argparse
import os
import re
import shutil
import sqlite3
import tempfile
import zipfile
from pathlib import Path


def combine_db_results_by_mutation(
    results_dir: str | Path = Path(__file__).parent / "results",
    output_dir: str | Path | None = None,
) -> dict[str, Path]:
    """Merge individual rows into one database per mutation and return its path.

    Outputs default to results_dir/combined/db_<mutation>.db. All individual
    columns are preserved, with new unique IDs, plus seed and source_id columns
    identifying the original row. Existing outputs and duplicate seeds are
    rejected. Source databases are never modified.
    """
    results_dir = Path(results_dir)
    output_dir = Path(output_dir) if output_dir is not None else results_dir / "combined"
    groups: dict[str, dict[int, Path]] = {}
    for path in sorted(results_dir.iterdir()):
        match = re.fullmatch(r"db_(.+)_(-?\d+)\.db(?:\.zip)?", path.name)
        if not path.is_file() or match is None:
            continue
        mutation, seed_text = match.groups()
        seed = int(seed_text)
        runs = groups.setdefault(mutation, {})
        if seed in runs:
            raise ValueError(f"Duplicate seed {seed} for mutation {mutation}")
        runs[seed] = path
    if not groups:
        raise FileNotFoundError(f"No db_<mutation>_<seed>.db[.zip] files in {results_dir}")

    outputs = {mutation: output_dir / f"db_{mutation}.db" for mutation in groups}
    for output in outputs.values():
        if output.exists():
            raise FileExistsError(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    for mutation, runs in groups.items():
        # Build in a temporary directory so a failed merge leaves no partial DB.
        with tempfile.TemporaryDirectory(dir=output_dir) as temporary:
            temporary = Path(temporary)
            merged = temporary / "merged.db"
            with sqlite3.connect(merged, uri=True) as db:
                expected_schema = None
                for seed, source in runs.items():
                    if source.suffix == ".zip":
                        extracted = temporary / "source.db"
                        with zipfile.ZipFile(source) as archive:
                            members = [name for name in archive.namelist() if name.endswith(".db")]
                            if len(members) != 1:
                                raise ValueError(f"Expected exactly one database in {source}")
                            with archive.open(members[0]) as reader, extracted.open("wb") as writer:
                                shutil.copyfileobj(reader, writer)
                        source = extracted
                    db.execute("ATTACH DATABASE ? AS run", (source.resolve().as_uri() + "?mode=ro",))
                    schema = db.execute("PRAGMA run.table_info(individual)").fetchall()
                    if not schema or not any(column[1] == "id" and column[5] == 1 for column in schema):
                        raise ValueError(f"Missing individual table with an id primary key: {source}")
                    if expected_schema is None:
                        expected_schema = schema
                        if {column[1] for column in schema} & {"seed", "source_id"}:
                            raise ValueError(f"Source already contains seed or source_id: {source}")
                        create_sql = db.execute(
                            "SELECT sql FROM run.sqlite_master WHERE type='table' AND name='individual'"
                        ).fetchone()[0]
                        db.execute(create_sql)
                        db.execute("ALTER TABLE individual ADD COLUMN seed INTEGER NOT NULL DEFAULT 0")
                        db.execute("ALTER TABLE individual ADD COLUMN source_id INTEGER")
                    elif schema != expected_schema:
                        raise ValueError(f"Incompatible individual schema: {source}")
                    columns = ", ".join(
                        '"' + column[1].replace('"', '""') + '"'
                        for column in schema if column[1] != "id"
                    )
                    prefix = f"{columns}, " if columns else ""
                    db.execute(
                        f"INSERT INTO main.individual ({prefix}seed, source_id) "
                        f"SELECT {prefix}?, id FROM run.individual", (seed,)
                    )
                    db.commit()
                    db.execute("DETACH DATABASE run")
            # Publish atomically without overwriting another concurrent merge.
            os.link(merged, outputs[mutation])
    return outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", nargs="?", type=Path, default=Path(__file__).parent / "results")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    for mutation, path in combine_db_results_by_mutation(args.results_dir, args.output_dir).items():
        print(f"{mutation}: {path}")

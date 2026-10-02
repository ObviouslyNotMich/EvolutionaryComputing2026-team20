# Assignment 2: ES step-size experiment

Research question: does a self-adaptive (μ,λ)-ES evolve better controllers with one step size shared by all layers, `[W1,W2,W3]·σ`, or with one step size per layer, `W1·σ1, W2·σ2, W3·σ3`?
Baseline: random search with the same evaluation budget.

Setup: `spider_8` body (John Set), `OlympicArena` world with the terrain seed fixed (`TERRAIN_SEED`), 15 s per simulation, fitness = XY distance to the target (lower is better).

Run all commands from the repo root.

## Single run

```bash
uv run assignments/assignment_2/ai_gen_ea_test.py --mode single --seed 0   # [W1,W2,W3]·σ
uv run assignments/assignment_2/ai_gen_ea_test.py --mode layer  --seed 0   # Wi·σi
uv run assignments/assignment_2/ai_gen_ea_test.py --mode random --seed 0   # random-search baseline
```

Options (defaults shown): `--gens 50 --mu 10 --lam 70 --workers 4`

Quick smoke test:

```bash
uv run assignments/assignment_2/ai_gen_ea_test.py --mode layer --gens 2 --mu 4 --lam 8
```

Output: `__data__/ea_test/<mode>_seed<seed>/database.db`

## Full experiment (3 modes × 5 seeds)

```bash
for m in single layer random; do for s in 0 1 2 3 4; do
  uv run assignments/assignment_2/ai_gen_ea_test.py --mode $m --seed $s --gens 50 --workers 14
done; done 2>&1 | tee __data__/ea_test/run.log
```

Or open it in a new Konsole window (`ai_gen_run_all.sh` runs the loop above, then plots):

```bash
konsole --separate -e bash assignments/assignment_2/ai_gen_run_all.sh
tail -f __data__/ea_test/run.log   # follow progress
```

## Plot

```bash
uv run assignments/assignment_2/ai_gen_plot_ea_test.py
```

Output: `__data__/ea_test/fitness.png` (best so far and offspring mean per generation, mean ± std over seeds).

## Watch the best robot walk

```bash
uv run assignments/assignment_2/ai_gen_view_best.py __data__/ea_test/single_seed0/database.db           # 3D viewer
uv run assignments/assignment_2/ai_gen_view_best.py __data__/ea_test/single_seed0/database.db --video   # mp4 in <run>/videos/
```

Videos of the best robot from every run:

```bash
for db in __data__/ea_test/*_seed*/database.db; do
  uv run assignments/assignment_2/ai_gen_view_best.py "$db" --video
done
```

## Inspect a database

Each run is one SQLite file with a single table, `individual`. Its columns are `id, alive, time_of_birth, time_of_death, requires_eval, fitness_, requires_init, genotype_, tags_`. `genotype_` is JSON: `{"w": [W1, W2, W3], "sigma": [...]}`.

```bash
DB=__data__/ea_test/single_seed0/database.db

# best 5 individuals
sqlite3 $DB "SELECT id, time_of_birth, fitness_ FROM individual ORDER BY fitness_ LIMIT 5;"

# best / mean fitness per generation
sqlite3 $DB "SELECT time_of_birth, MIN(fitness_), AVG(fitness_) FROM individual GROUP BY time_of_birth;"

# step sizes of the current (alive) population
sqlite3 $DB "SELECT fitness_, json_extract(genotype_, '$.sigma') FROM individual WHERE alive = 1;"

# number of evaluations so far
sqlite3 $DB "SELECT COUNT(*) FROM individual;"

# best fitness of every run
for db in __data__/ea_test/*_seed*/database.db; do
  echo "$db $(sqlite3 $db 'SELECT MIN(fitness_) FROM individual;')"
done
```

## Dashboard (one database at a time)

```bash
uv run panel serve src/ariel/visualisation/dashboard/dashboard_new.py --show
```

Open http://localhost:5006/dashboard_new, then go to **Load Database** and pick a `.db` file.

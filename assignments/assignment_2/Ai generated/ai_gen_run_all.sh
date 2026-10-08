#!/usr/bin/env bash
cd "$(dirname "$0")/../.."  # repo root
mkdir -p __data__/ea_test
GENS=50
for m in single layer random; do for s in 0 1 2 3 4; do
  db=__data__/ea_test/${m}_seed${s}/database.db
  if [ "$(sqlite3 "$db" 'SELECT MAX(time_of_birth) FROM individual' 2>/dev/null)" = "$GENS" ]; then
    echo "=== $m seed $s already done, skipping ==="; continue
  fi
  echo "=== $m seed $s ==="
  uv run assignments/assignment_2/ai_gen_ea_test.py --mode $m --seed $s --gens $GENS --workers 14
done; done 2>&1 | tee -a __data__/ea_test/run.log
uv run assignments/assignment_2/ai_gen_plot_ea_test.py
exec bash

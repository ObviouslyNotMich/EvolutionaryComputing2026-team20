"""Watch the best controller from an ea_test.py run walk in the OlympicArena.

    uv run assignments/assignment_2/ai_gen_view_best.py __data__/ea_test/single_seed0/database.db
    uv run assignments/assignment_2/ai_gen_view_best.py <db> --video   # save mp4 instead
"""

import argparse
import json
import sqlite3
from pathlib import Path

import mujoco as mj
import numpy as np
from mujoco import viewer

from ai_gen_ea_test import SIM_DURATION, build_model, fitness_function, nn_controller
from ariel.utils.renderers import video_renderer
from ariel.utils.video_recorder import VideoRecorder


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("db", type=Path)
    p.add_argument("--video", action="store_true")
    args = p.parse_args()

    fit, geno = sqlite3.connect(args.db).execute(
        "SELECT fitness_, genotype_ FROM individual WHERE fitness_ IS NOT NULL "
        "ORDER BY fitness_ LIMIT 1"
    ).fetchone()
    weights = [np.asarray(w) for w in json.loads(geno)["w"]]
    print(f"best stored fitness: {fit:.4f}")

    model, data = build_model()
    mj.set_mjcb_control(lambda m, d: d.ctrl.__setitem__(slice(None), nn_controller(d, weights)))

    if args.video:
        video_renderer(model, data, duration=SIM_DURATION,
                       video_recorder=VideoRecorder(output_folder=str(args.db.parent / "videos")))
        print(f"video saved in {args.db.parent / 'videos'}, replay fitness {fitness_function(data.qpos[:3]):.4f}")
    else:
        viewer.launch(model=model, data=data)  # sim time runs on; fitness is at t=15s
    mj.set_mjcb_control(None)


if __name__ == "__main__":
    main()

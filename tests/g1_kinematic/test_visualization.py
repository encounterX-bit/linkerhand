"""G1 test 5 — visualization artifact (human eyeball).

Until perception's real sequences land, we still produce the target-vs-achieved
artifact the ticket asks for, by replaying the G0 synthetic fixtures through the
harness. Writes a per-frame CSV (target vs achieved direction, thumb included) and
an overlay PNG/GIF sequence to out/. These fixtures include genuinely unreachable
thumb poses, so the overlay shows real nonzero residuals — which is the point.
"""
import glob
import json
import os

import pytest

from src.sim import L20Kinematics, FINGER_ORDER
from src.sim.pipeline import track_frame, error_rows
from src.sim import viz

HERE = os.path.dirname(os.path.abspath(__file__))
G0_FIXTURES = os.path.join(os.path.dirname(HERE), "g0_unit", "fixtures")
OUT_DIR = os.path.join(HERE, "out")


def _fixtures(side):
    files = []
    for path in sorted(glob.glob(os.path.join(G0_FIXTURES, f"*_{side}.json"))):
        if "oracle_cache" not in os.path.basename(path):
            files.append(path)
    return files


@pytest.mark.parametrize("side", ["right", "left"])
def test_write_visualization(side):
    files = _fixtures(side)
    assert files, f"no G0 fixtures for {side}"
    kin = L20Kinematics(side)
    try:
        records, labels, rows = [], [], []
        for i, path in enumerate(files):
            lm = json.load(open(path))["landmarks"]
            rec = track_frame(kin, lm, side)
            records.append(rec)
            labels.append(os.path.basename(path).replace(".json", ""))
            rows.extend(error_rows(rec, frame=i))
        csv_path = viz.write_csv(rows, os.path.join(OUT_DIR, f"synthetic_{side}.csv"))
        seq_dir = viz.render_sequence(records, os.path.join(OUT_DIR, f"synthetic_{side}"),
                                      labels)
        assert os.path.exists(csv_path)
        assert os.path.exists(os.path.join(seq_dir, "sequence.gif"))
        # sanity: every row has finite error and all fingers represented
        assert {r["finger"] for r in rows} == set(FINGER_ORDER)
    finally:
        kin.close()

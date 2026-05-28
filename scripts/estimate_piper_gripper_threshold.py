#!/usr/bin/env python3
import json, os, sys
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
THRESHOLD_JSON = Path(__file__).resolve().parents[1] / "datasets/domain_handler/slai_piper_gripper_thresholds.json"

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
sys.path.insert(0, str(WORKSPACE_ROOT / "Motus/scripts"))
from estimate_gripper_threshold import ARMS, episode_rows, grasp_value, read_grippers

MOTUS_KEY = "piper_multi_tasks_dual_14d_gripper01"


def estimate_threshold(root):
    values = []
    for row in episode_rows(root):
        grippers = read_grippers(root, row["episode_index"])
        found = [grasp_value(grippers[source][arm], 0.20, 0.12, 0.03) for source in ("state", "action") for arm in ARMS]
        values.extend(value for value in found if value is not None)
    if not values:
        raise ValueError(f"Could not estimate gripper threshold for {root}")
    return float(max(values))


def write_threshold(root, threshold):
    thresholds = json.loads(THRESHOLD_JSON.read_text()) if THRESHOLD_JSON.exists() else {}
    thresholds[root.name] = float(threshold)
    THRESHOLD_JSON.write_text(json.dumps(thresholds, indent=4, ensure_ascii=False) + "\n")


def main():
    data = Path(sys.argv[1]) if len(sys.argv) > 1 else WORKSPACE_ROOT / "data/ZhaoRunyi"
    roots = [data] if (data / "meta/info.json").exists() else [path.parent.parent for path in sorted(data.glob("Piper_*/meta/info.json"))]
    roots = [root for root in roots if "real2sim" not in root.name.lower()]
    stat_path = WORKSPACE_ROOT / "Motus/data/utils/stat.json"
    tasks = json.loads(stat_path.read_text()).get(MOTUS_KEY, {}).get("gripper_grasp_values", {}).get("task", {})
    for root in roots:
        value = tasks.get("Motus_" + root.name)
        threshold = float(value) if value is not None else estimate_threshold(root)
        write_threshold(root, threshold)
        print(f"{root.name}: {threshold:.6f}")


if __name__ == "__main__":
    main()

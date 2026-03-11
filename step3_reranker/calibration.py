"""
Step 3 – User Calibration / Learning Engine
=============================================
Persists user corrections so the system can learn from them over time.

Example: if the user always says "a portion of noodles" and corrects
the system to 500 g, the next time we'll use 500 g directly.


Storage format  (data/calibrations/user_prefs.json):
{
  "user_001": {
    "noodles": {
      "preferred_grams": 500,
      "preferred_unit": "g",
      "corrections": 3,
      "last_corrected": "2026-03-05T14:00:00"
    }
  }
}
"""

import json
from datetime import datetime
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402


def _load_calibrations() -> dict:
    if config.CALIBRATION_FILE.exists():
        with open(config.CALIBRATION_FILE) as f:
            content = f.read().strip()
            if not content:
                return {}
            return json.loads(content)
    return {}


def _save_calibrations(data: dict):
    config.CALIBRATION_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(config.CALIBRATION_FILE, "w") as f:
        json.dump(data, f, indent=2)


def get_user_preference(uid: str, item_name: str) -> dict | None:
    """
    Look up whether this user has a stored preference for the given item.

    Returns
    -------
    dict with "preferred_grams" and "preferred_unit", or None.
    """
    cal = _load_calibrations()
    user = cal.get(uid, {})
    key = item_name.lower().strip()
    return user.get(key)


def save_user_correction(uid: str, item_name: str, grams: float, unit: str = "g"):
    """
    Record a user correction to the calibration database.

    If the user has corrected the same item before, we increment the counter.
    """
    cal = _load_calibrations()
    if uid not in cal:
        cal[uid] = {}

    key = item_name.lower().strip()
    existing = cal[uid].get(key, {})

    cal[uid][key] = {
        "preferred_grams": grams,
        "preferred_unit": unit,
        "corrections": existing.get("corrections", 0) + 1,
        "last_corrected": datetime.now().isoformat(),
    }

    _save_calibrations(cal)
    print(f"   📝 Saved calibration: {uid} → \"{item_name}\" = {grams}{unit} "
          f"(correction #{cal[uid][key]['corrections']})")

"""
SayFit Alpha – Database Management
===================================
SQLite-basierte persistente Speicherung von Mahlzeiten mit:
  • Eindeutige Identifikatoren (UUIDs)
  • Tages-Gruppierung für Stabilitätt
  • Transaktionale Integrität
  • Soft Deletes für Audit-Trail
  • Kalibrierungen (Benutzer-Korrektionen)
"""

import sqlite3
import uuid
from datetime import datetime, date
from pathlib import Path
from contextlib import contextmanager
from typing import Optional, List, Dict, Any

import config


class SayFitDB:
    """Database abstraction layer for SayFit."""

    def __init__(self, db_path: Path = None):
        self.db_path = db_path or config.DB_PATH
        self._ensure_db()

    def _ensure_db(self):
        """Create database and schema if not exists."""
        with self.get_connection() as conn:
            conn.executescript("""
                -- Users
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                );

                -- Meals (grouped by date for stability)
                CREATE TABLE IF NOT EXISTS meals (
                    meal_id TEXT PRIMARY KEY,
                    meal_name TEXT DEFAULT '',
                    user_id TEXT NOT NULL,
                    meal_date DATE NOT NULL,
                    logged_at DATETIME NOT NULL,
                    input_text TEXT NOT NULL,
                    
                    total_calories REAL DEFAULT 0,
                    total_protein REAL DEFAULT 0,
                    total_fat REAL DEFAULT 0,
                    total_carbs REAL DEFAULT 0,
                    
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    is_deleted BOOLEAN DEFAULT 0,
                    
                    FOREIGN KEY (user_id) REFERENCES users(user_id),
                    UNIQUE(meal_id)
                );

                -- Individual meal items
                CREATE TABLE IF NOT EXISTS meal_items (
                    item_id TEXT PRIMARY KEY,
                    meal_id TEXT NOT NULL,
                    item_name TEXT NOT NULL,
                    matched_name TEXT NOT NULL,
                    amount_grams REAL NOT NULL,
                    unit TEXT DEFAULT 'g',
                    
                    calories REAL DEFAULT 0,
                    protein REAL DEFAULT 0,
                    fat REAL DEFAULT 0,
                    carbs REAL DEFAULT 0,
                    
                    confidence TEXT DEFAULT 'medium',
                    confidence_note TEXT,
                    
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    is_deleted BOOLEAN DEFAULT 0,
                    
                    FOREIGN KEY (meal_id) REFERENCES meals(meal_id) ON DELETE CASCADE
                );

                -- User calibrations (learned preferences)
                CREATE TABLE IF NOT EXISTS calibrations (
                    calibration_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    item_name TEXT NOT NULL,
                    preferred_grams REAL NOT NULL,
                    preferred_unit TEXT DEFAULT 'g',
                    corrections_count INTEGER DEFAULT 1,
                    last_corrected DATETIME NOT NULL,
                    
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    
                    FOREIGN KEY (user_id) REFERENCES users(user_id),
                    UNIQUE(user_id, item_name)
                );

                -- Indices for fast queries
                CREATE INDEX IF NOT EXISTS idx_meals_user_date 
                    ON meals(user_id, meal_date) WHERE is_deleted = 0;
                CREATE INDEX IF NOT EXISTS idx_meals_logged_at 
                    ON meals(logged_at);
                CREATE INDEX IF NOT EXISTS idx_meal_items_meal 
                    ON meal_items(meal_id) WHERE is_deleted = 0;
                CREATE INDEX IF NOT EXISTS idx_calibrations_user 
                    ON calibrations(user_id);

                -- User profiles (TDEE + daily macro targets)
                CREATE TABLE IF NOT EXISTS user_profiles (
                    user_id TEXT PRIMARY KEY,
                    weight_kg REAL NOT NULL,
                    age_years INTEGER NOT NULL,
                    pal REAL NOT NULL,
                    training_met REAL DEFAULT 0,
                    training_hours_per_week REAL DEFAULT 0,
                    kcal_daily REAL NOT NULL,
                    protein_daily REAL NOT NULL,
                    fat_daily REAL NOT NULL,
                    carbs_daily REAL NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(user_id)
                );
            """)
            # Add daily-target columns to users if they don't exist yet
            # (SQLite has no ADD COLUMN IF NOT EXISTS, so we suppress the error)
            for col, typ in [
                ("kcal_daily",    "REAL"),
                ("protein_daily", "REAL"),
                ("fat_daily",     "REAL"),
                ("carbs_daily",   "REAL"),
            ]:
                try:
                    conn.execute(f"ALTER TABLE users ADD COLUMN {col} {typ}")
                    conn.commit()
                except Exception:
                    pass  # column already exists

    @contextmanager
    def get_connection(self, timeout: int = 30):
        """Context manager for database connections."""
        conn = sqlite3.connect(str(self.db_path), timeout=timeout)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _ensure_user(self, conn, user_id: str):
        """Create user record if not exists (idempotent)."""
        conn.execute("""
            INSERT OR IGNORE INTO users (user_id, created_at, updated_at)
            VALUES (?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
        """, (user_id,))

    def save_meal(
        self,
        user_id: str,
        items: List[Dict[str, Any]],
        input_text: str,
        meal_date: Optional[str] = None,
        meal_id: Optional[str] = None,
        meal_name: str = "",
    ) -> str:
        """
        Save a meal with all its items atomically.

        Parameters
        ----------
        user_id : str
            User ID (will be created if not exists)
        items : list
            List of dicts with keys: item_name, matched_name, amount_grams,
            unit, calories, protein, fat, carbs, confidence, confidence_note
        input_text : str
            Original user input (e.g., "i ate a banana and coffee")
        meal_date : str, optional
            Date the meal belongs to (YYYY-MM-DD). Default: today.
        meal_id : str, optional
            UUID (auto-generated if not provided)

        Returns
        -------
        str : The meal_id

        Raises
        ------
        sqlite3.Error
            If atomicity is violated (transaction rolled back)
        """
        if not meal_id:
            meal_id = str(uuid.uuid4())

        if not meal_date:
            meal_date = datetime.now().strftime("%Y-%m-%d")

        logged_at = datetime.now().isoformat()

        with self.get_connection() as conn:
            try:
                conn.execute("BEGIN TRANSACTION")

                # 1. Ensure user exists
                self._ensure_user(conn, user_id)

                # 2. Insert meal
                conn.execute(
                    """
                    INSERT INTO meals
                    (meal_id, meal_name, user_id, meal_date, logged_at, input_text, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                    """,
                    (meal_id, meal_name, user_id, meal_date, logged_at, input_text),
                )

                # 3. Insert items and calculate totals
                total_cal = total_prot = total_fat = total_carbs = 0.0

                for item in items:
                    item_id = str(uuid.uuid4())
                    cal = item.get("calories", 0) or 0
                    prot = item.get("protein", 0) or 0
                    fat = item.get("fat", 0) or 0
                    carbs = item.get("carbs", 0) or 0

                    total_cal += cal
                    total_prot += prot
                    total_fat += fat
                    total_carbs += carbs

                    conn.execute(
                        """
                        INSERT INTO meal_items
                        (item_id, meal_id, item_name, matched_name, amount_grams, unit,
                         calories, protein, fat, carbs, confidence, confidence_note,
                         created_at, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 
                                CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                        """,
                        (
                            item_id,
                            meal_id,
                            item.get("item_name", ""),
                            item.get("matched_name", ""),
                            item.get("amount_grams", 0),
                            item.get("unit", "g"),
                            cal,
                            prot,
                            fat,
                            carbs,
                            item.get("confidence", "medium"),
                            item.get("confidence_note", ""),
                        ),
                    )

                # 4. Update meal with aggregated totals
                conn.execute(
                    """
                    UPDATE meals
                    SET total_calories = ?,
                        total_protein = ?,
                        total_fat = ?,
                        total_carbs = ?,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE meal_id = ?
                    """,
                    (total_cal, total_prot, total_fat, total_carbs, meal_id),
                )

                conn.commit()
                print(f"   ✅ Meal saved to DB: {meal_id} (date: {meal_date})")

            except sqlite3.Error as e:
                conn.rollback()
                print(f"   ❌ DB error: {e}")
                raise

        return meal_id

    def get_daily_totals(self, user_id: str, meal_date: str) -> Dict[str, float]:
        """
        Get aggregated nutrition for a specific day.

        Parameters
        ----------
        user_id : str
        meal_date : str (YYYY-MM-DD)

        Returns
        -------
        dict with keys: calories, protein, fat, carbs, meal_count
        """
        with self.get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT
                    COALESCE(SUM(total_calories), 0) as calories,
                    COALESCE(SUM(total_protein), 0) as protein,
                    COALESCE(SUM(total_fat), 0) as fat,
                    COALESCE(SUM(total_carbs), 0) as carbs,
                    COUNT(*) as meal_count
                FROM meals
                WHERE user_id = ? AND meal_date = ? AND is_deleted = 0
                """,
                (user_id, meal_date),
            )
            result = cursor.fetchone()

            return {
                "calories": result[0],
                "protein": result[1],
                "fat": result[2],
                "carbs": result[3],
                "meal_count": result[4],
            }

    def get_meals_for_day(self, user_id: str, meal_date: str) -> List[Dict]:
        """Get all meals (with items) for a specific day."""
        with self.get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT meal_id, meal_name, meal_date, logged_at, input_text,
                       total_calories, total_protein, total_fat, total_carbs
                FROM meals
                WHERE user_id = ? AND meal_date = ? AND is_deleted = 0
                ORDER BY logged_at
                """,
                (user_id, meal_date),
            )
            meals = [dict(row) for row in cursor.fetchall()]

            # Fetch items for each meal
            for meal in meals:
                cursor = conn.execute(
                    """
                    SELECT item_id, item_name, matched_name, amount_grams, unit,
                           calories, protein, fat, carbs, confidence, confidence_note
                    FROM meal_items
                    WHERE meal_id = ? AND is_deleted = 0
                    """,
                    (meal["meal_id"],),
                )
                meal["items"] = [dict(row) for row in cursor.fetchall()]

            return meals

    def add_calibration(self, user_id: str, item_name: str, grams: float):
        """Record a user correction (learning)."""
        calib_id = str(uuid.uuid4())
        now = datetime.now().isoformat()

        with self.get_connection() as conn:
            self._ensure_user(conn, user_id)

            cursor = conn.execute(
                """
                SELECT corrections_count FROM calibrations
                WHERE user_id = ? AND item_name = ?
                """,
                (user_id, item_name),
            )
            existing = cursor.fetchone()

            if existing:
                # Update existing
                conn.execute(
                    """
                    UPDATE calibrations
                    SET preferred_grams = ?,
                        corrections_count = corrections_count + 1,
                        last_corrected = ?,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE user_id = ? AND item_name = ?
                    """,
                    (grams, now, user_id, item_name),
                )
            else:
                # Insert new
                conn.execute(
                    """
                    INSERT INTO calibrations
                    (calibration_id, user_id, item_name, preferred_grams,
                     last_corrected, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                    """,
                    (calib_id, user_id, item_name, grams, now),
                )

            conn.commit()
            print(f"   ✅ Calibration saved: {item_name} → {grams}g")

    def get_calibration(self, user_id: str, item_name: str) -> Optional[Dict]:
        """Look up user's preferred amount for an item."""
        with self.get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT preferred_grams, corrections_count, last_corrected
                FROM calibrations
                WHERE user_id = ? AND item_name = ?
                """,
                (user_id, item_name),
            )
            result = cursor.fetchone()
            return dict(result) if result else None

    def save_pipeline_result(
        self,
        reranked: dict,
        uid: str = "default_user",
        input_text: str = "",
        meal_date: str = None,
        meal_name: str = "",
    ) -> str:
        """
        Convert Step 3 reranker output → database format and save atomically.

        Parameters
        ----------
        reranked : dict
            Output from rerank_all() with a "results" list.
        uid : str
            User ID (auto-created if not exists).
        input_text : str
            Original user input text.
        meal_date : str, optional
            Date the meal belongs to (YYYY-MM-DD). Default: today.

        Returns
        -------
        str : The saved meal_id.
        """
        items = []
        for result in reranked.get("results", []):
            nutr = result.get("nutrition", {})
            items.append({
                "item_name": result.get("item_name", ""),
                "matched_name": result.get("matched_name", ""),
                "amount_grams": result.get("amount_grams", 0),
                "unit": result.get("unit", "g"),
                "confidence": result.get("confidence", "medium"),
                "confidence_note": result.get("confidence_note", ""),
                "calories": nutr.get("calories", 0) or 0,
                "protein": nutr.get("protein", 0) or 0,
                "fat": nutr.get("fat", 0) or 0,
                "carbs": nutr.get("carbs", 0) or 0,
            })
        return self.save_meal(
            user_id=uid,
            items=items,
            input_text=input_text,
            meal_date=meal_date,
            meal_name=meal_name,
        )

    def save_user_profile(
        self,
        uid: str,
        weight_kg: float,
        age_years: int,
        pal: float,
        training_met: float,
        training_hours_per_week: float,
        kcal_daily: float,
        protein_daily: float,
        fat_daily: float,
        carbs_daily: float,
    ) -> None:
        """Save (or update) the user's physical profile and daily macro targets."""
        with self.get_connection() as conn:
            self._ensure_user(conn, uid)
            conn.execute(
                """
                INSERT INTO user_profiles
                    (user_id, weight_kg, age_years, pal,
                     training_met, training_hours_per_week,
                     kcal_daily, protein_daily, fat_daily, carbs_daily,
                     created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                        CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                ON CONFLICT(user_id) DO UPDATE SET
                    weight_kg               = excluded.weight_kg,
                    age_years               = excluded.age_years,
                    pal                     = excluded.pal,
                    training_met            = excluded.training_met,
                    training_hours_per_week = excluded.training_hours_per_week,
                    kcal_daily              = excluded.kcal_daily,
                    protein_daily           = excluded.protein_daily,
                    fat_daily               = excluded.fat_daily,
                    carbs_daily             = excluded.carbs_daily,
                    updated_at              = CURRENT_TIMESTAMP
                """,
                (uid, weight_kg, age_years, pal,
                 training_met, training_hours_per_week,
                 kcal_daily, protein_daily, fat_daily, carbs_daily),
            )
            # Mirror the 4 daily targets into users for easy access
            conn.execute(
                """
                UPDATE users
                SET kcal_daily    = ?,
                    protein_daily = ?,
                    fat_daily     = ?,
                    carbs_daily   = ?,
                    updated_at    = CURRENT_TIMESTAMP
                WHERE user_id = ?
                """,
                (kcal_daily, protein_daily, fat_daily, carbs_daily, uid),
            )
            conn.commit()

    def get_user_profile(self, uid: str) -> Optional[Dict]:
        """Return the stored user profile, or None if not yet set up."""
        with self.get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT weight_kg, age_years, pal, training_met,
                       training_hours_per_week,
                       kcal_daily, protein_daily, fat_daily, carbs_daily,
                       updated_at
                FROM user_profiles
                WHERE user_id = ?
                """,
                (uid,),
            )
            row = cursor.fetchone()
            return dict(row) if row else None

    def print_daily_summary(self, user_id: str, meal_date: str = None):
        """Print aggregated nutrition for a day, with remaining targets if profile exists."""
        if not meal_date:
            meal_date = datetime.now().strftime("%Y-%m-%d")
        totals = self.get_daily_totals(user_id, meal_date)
        profile = self.get_user_profile(user_id)
        print("\n" + "=" * 60)
        print(f"  Daily Summary – {meal_date}  (all meals today)")
        print("=" * 60)
        if profile:
            def _row(label: str, consumed: float, target: float, unit: str) -> None:
                rem = target - consumed
                tag = "over" if rem < 0 else "left"
                print(f"  {label:<10}: {consumed:6.1f} / {target:.0f} {unit}"
                      f"   ({abs(rem):.1f} {tag})")
            _row("Calories", totals["calories"], profile["kcal_daily"],    "kcal")
            _row("Protein",  totals["protein"],  profile["protein_daily"], "g")
            _row("Fat",      totals["fat"],       profile["fat_daily"],     "g")
            _row("Carbs",    totals["carbs"],     profile["carbs_daily"],   "g")
        else:
            print(f"  Calories : {totals['calories']:.1f} kcal")
            print(f"  Protein  : {totals['protein']:.1f} g")
            print(f"  Fat      : {totals['fat']:.1f} g")
            print(f"  Carbs    : {totals['carbs']:.1f} g")
        print(f"  Meals    : {totals['meal_count']}")
        print("=" * 60)

    def delete_meal(self, meal_id: str) -> None:
        """Soft-delete a meal and all its items."""
        with self.get_connection() as conn:
            conn.execute(
                "UPDATE meals SET is_deleted = 1, updated_at = CURRENT_TIMESTAMP WHERE meal_id = ?",
                (meal_id,),
            )
            conn.execute(
                "UPDATE meal_items SET is_deleted = 1, updated_at = CURRENT_TIMESTAMP WHERE meal_id = ?",
                (meal_id,),
            )
            conn.commit()

    def delete_meal_item(self, item_id: str, meal_id: str) -> None:
        """Soft-delete an item and recalculate the parent meal totals."""
        with self.get_connection() as conn:
            conn.execute(
                "UPDATE meal_items SET is_deleted = 1, updated_at = CURRENT_TIMESTAMP WHERE item_id = ?",
                (item_id,),
            )
            conn.execute("""
                UPDATE meals SET
                    total_calories = (SELECT COALESCE(SUM(calories), 0) FROM meal_items WHERE meal_id = ? AND is_deleted = 0),
                    total_protein  = (SELECT COALESCE(SUM(protein),  0) FROM meal_items WHERE meal_id = ? AND is_deleted = 0),
                    total_fat      = (SELECT COALESCE(SUM(fat),      0) FROM meal_items WHERE meal_id = ? AND is_deleted = 0),
                    total_carbs    = (SELECT COALESCE(SUM(carbs),    0) FROM meal_items WHERE meal_id = ? AND is_deleted = 0),
                    updated_at     = CURRENT_TIMESTAMP
                WHERE meal_id = ?
            """, (meal_id, meal_id, meal_id, meal_id, meal_id))
            conn.commit()

    def update_meal_item_grams(self, item_id: str, meal_id: str, new_grams: float) -> None:
        """Scale an item's macros proportionally to new_grams and refresh meal totals."""
        with self.get_connection() as conn:
            cursor = conn.execute(
                "SELECT amount_grams, calories, protein, fat, carbs FROM meal_items WHERE item_id = ?",
                (item_id,),
            )
            row = cursor.fetchone()
            if not row:
                return
            old_grams = float(row[0]) or 100.0
            scale = new_grams / old_grams
            conn.execute("""
                UPDATE meal_items SET
                    amount_grams = ?,
                    calories     = ROUND(calories * ?, 1),
                    protein      = ROUND(protein  * ?, 1),
                    fat          = ROUND(fat       * ?, 1),
                    carbs        = ROUND(carbs     * ?, 1),
                    updated_at   = CURRENT_TIMESTAMP
                WHERE item_id = ?
            """, (new_grams, scale, scale, scale, scale, item_id))
            conn.execute("""
                UPDATE meals SET
                    total_calories = (SELECT COALESCE(SUM(calories), 0) FROM meal_items WHERE meal_id = ? AND is_deleted = 0),
                    total_protein  = (SELECT COALESCE(SUM(protein),  0) FROM meal_items WHERE meal_id = ? AND is_deleted = 0),
                    total_fat      = (SELECT COALESCE(SUM(fat),      0) FROM meal_items WHERE meal_id = ? AND is_deleted = 0),
                    total_carbs    = (SELECT COALESCE(SUM(carbs),    0) FROM meal_items WHERE meal_id = ? AND is_deleted = 0),
                    updated_at     = CURRENT_TIMESTAMP
                WHERE meal_id = ?
            """, (meal_id, meal_id, meal_id, meal_id, meal_id))
            conn.commit()

    def get_stats(self, user_id: str, days: int = 7) -> Dict[str, Any]:
        """Get nutrition stats for the last N days."""
        with self.get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT meal_date,
                       SUM(total_calories) as calories,
                       SUM(total_protein) as protein,
                       SUM(total_fat) as fat,
                       SUM(total_carbs) as carbs,
                       COUNT(*) as meal_count
                FROM meals
                WHERE user_id = ? AND meal_date >= DATE('now', '-' || ? || ' days') 
                      AND is_deleted = 0
                GROUP BY meal_date
                ORDER BY meal_date DESC
                """,
                (user_id, days),
            )
            daily = [dict(row) for row in cursor.fetchall()]

            # Calculate averages and totals
            if daily:
                avg_cal = sum(d["calories"] for d in daily) / len(daily)
                avg_prot = sum(d["protein"] for d in daily) / len(daily)
            else:
                avg_cal = avg_prot = 0

            return {
                "user_id": user_id,
                "days": days,
                "daily_breakdown": daily,
                "average_calories": avg_cal,
                "average_protein": avg_prot,
            }


# Global DB instance
_db_instance = None


def get_db() -> SayFitDB:
    """Get or create the global database instance."""
    global _db_instance
    if _db_instance is None:
        _db_instance = SayFitDB()
    return _db_instance

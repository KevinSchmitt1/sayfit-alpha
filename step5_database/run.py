"""
Step 5 – Standalone database runner
=====================================
Query the SayFit logbook without running the full pipeline.

Usage:
    python -m step5_database.run                        # show today's summary
    python -m step5_database.run --uid alice            # specific user
    python -m step5_database.run --uid alice --days 7   # last 7 days
    python -m step5_database.run --date 2026-03-09      # specific date
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from step5_database.database import get_db  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Step 5 – Database Logbook")
    parser.add_argument("--uid", type=str, default="default_user", help="User ID")
    parser.add_argument("--date", type=str, default=None,
                        help="Date to query (YYYY-MM-DD). Default: today.")
    parser.add_argument("--days", type=int, default=None,
                        help="Show stats for last N days instead of a single day.")
    args = parser.parse_args()

    db = get_db()

    if args.days:
        print("=" * 60)
        print(f"  Step 5 – Stats for '{args.uid}' — last {args.days} days")
        print("=" * 60)
        stats = db.get_stats(args.uid, days=args.days)
        print(f"  Average calories : {stats['average_calories']:.1f} kcal/day")
        print(f"  Average protein  : {stats['average_protein']:.1f} g/day")
        print()
        for day in stats["daily_breakdown"]:
            print(f"  {day['meal_date']}:  {day['calories']:.0f} kcal  "
                  f"({day['meal_count']} meal(s))")
        print("=" * 60)
    else:
        meal_date = args.date or datetime.now().strftime("%Y-%m-%d")
        print("=" * 60)
        print(f"  Step 5 – Logbook for '{args.uid}' on {meal_date}")
        print("=" * 60)
        meals = db.get_meals_for_day(args.uid, meal_date)
        if not meals:
            print(f"  No meals logged for {meal_date}.")
        else:
            for meal in meals:
                print(f"\n  Meal: {meal['meal_id'][:8]}…  |  {meal['total_calories']:.0f} kcal")
                print(f"  Input: \"{meal['input_text']}\"")
                for item in meal["items"]:
                    print(f"    • {item['item_name']} → {item['matched_name']} "
                          f"({item['amount_grams']:.0f}g, {item['calories']:.0f} kcal)")
        print()
        db.print_daily_summary(args.uid, meal_date)


if __name__ == "__main__":
    main()

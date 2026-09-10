#!/usr/bin/env python3
"""Update all cached symbols to include recent data.

This is a simpler alternative to prepare_v2_universe.py - instead of fetching
a new S&P 500 list, it just tops up whatever symbols we already have cached.

Usage:
    # Dry run: show what needs updating
    .venv/bin/python scripts/update_universe_data.py --dry-run

    # Update all stale symbols
    .venv/bin/python scripts/update_universe_data.py --batch-size 30

    # Force update even if recent
    .venv/bin/python scripts/update_universe_data.py --force
"""
import argparse
from pathlib import Path
from datetime import datetime, timedelta
import sys
import time

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from python.data.price_cache import top_up_cached_panel


def check_cache_freshness(cache_dir: Path, days_threshold: int = 7) -> dict:
    """Check which cached symbols need updating.
    
    Returns:
        {
            'fresh': list of (symbol, last_date) that are recent enough,
            'stale': list of (symbol, last_date) that need updating,
            'broken': list of (symbol, error) that can't be read,
        }
    """
    csv_files = list(cache_dir.glob("*.csv"))
    cutoff = pd.Timestamp.now() - timedelta(days=days_threshold)
    
    fresh = []
    stale = []
    broken = []
    
    for csv_file in csv_files:
        symbol = csv_file.stem
        try:
            df = pd.read_csv(csv_file)
            if 'date' not in df.columns:
                broken.append((symbol, "no 'date' column"))
                continue
            
            df['date'] = pd.to_datetime(df['date'])
            last_date = df['date'].max()
            
            if last_date >= cutoff:
                fresh.append((symbol, last_date))
            else:
                stale.append((symbol, last_date))
        except Exception as e:
            broken.append((symbol, str(e)))
    
    return {
        'fresh': fresh,
        'stale': stale,
        'broken': broken,
    }


def update_batch(symbols: list[str], cache_dir: Path) -> dict:
    """Top up a batch of symbols with recent data.
    
    Returns:
        {
            'success': list of successfully updated symbols,
            'failed': list of (symbol, error) that failed,
        }
    """
    success = []
    failed = []
    
    for i, symbol in enumerate(symbols, 1):
        print(f"  [{i}/{len(symbols)}] 更新 {symbol} ...", end=" ", flush=True)
        
        try:
            # top_up_cached_panel handles the incremental update
            result = top_up_cached_panel(
                [symbol],
                end=pd.Timestamp.today().normalize(),
                cache_dir=cache_dir,
            )
            
            if symbol in result.get('topped_up', []) or \
               symbol in result.get('already_current', []):
                print("✓")
                success.append(symbol)
            elif symbol in result.get('readjusted', []):
                print("✓ (調整)")
                success.append(symbol)
            else:
                print(f"✗ (未更新: {result})")
                failed.append((symbol, "not topped up"))
        except Exception as exc:
            print(f"✗ ({exc})")
            failed.append((symbol, str(exc)))
        
        # Small delay to be nice to yfinance
        time.sleep(0.2)
    
    return {'success': success, 'failed': failed}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--batch-size", type=int, default=30,
                    help="Number of symbols to update per batch")
    ap.add_argument("--dry-run", action="store_true",
                    help="Show what would be updated without actually updating")
    ap.add_argument("--force", action="store_true",
                    help="Update all symbols, even if recent")
    ap.add_argument("--cache-dir", type=Path, default=Path("data/history"),
                    help="Cache directory for price data")
    ap.add_argument("--days-threshold", type=int, default=7,
                    help="Consider stale if older than N days (default: 7)")
    args = ap.parse_args()
    
    print("=" * 70)
    print("更新 Universe 價格數據到最新")
    print("=" * 70)
    print()
    
    # Check freshness
    print(f"檢查快取新鮮度（門檻：{args.days_threshold} 天）...")
    status = check_cache_freshness(args.cache_dir, args.days_threshold)
    
    n_total = len(status['fresh']) + len(status['stale']) + len(status['broken'])
    
    print(f"\n快取狀態（總共 {n_total} 檔）：")
    print(f"  最新：{len(status['fresh'])} 檔")
    print(f"  過期：{len(status['stale'])} 檔")
    print(f"  損壞：{len(status['broken'])} 檔")
    
    if status['broken']:
        print(f"\n損壞的檔案：")
        for symbol, error in status['broken'][:10]:
            print(f"  {symbol}: {error}")
        if len(status['broken']) > 10:
            print(f"  ... 還有 {len(status['broken']) - 10} 檔")
    
    # Determine what to update
    if args.force:
        to_update = [sym for sym, _ in status['fresh']] + \
                    [sym for sym, _ in status['stale']]
        print(f"\n強制更新所有 {len(to_update)} 檔")
    else:
        to_update = [sym for sym, _ in status['stale']]
        print(f"\n需要更新 {len(to_update)} 檔過期標的")
    
    if len(to_update) == 0:
        print("\n✅ 所有標的都是最新，無需更新。")
        return
    
    # Show sample
    print(f"\n待更新標的範例：")
    for sym in to_update[:20]:
        print(f"  {sym}", end=" ")
    if len(to_update) > 20:
        print(f"\n  ... 還有 {len(to_update) - 20} 檔")
    else:
        print()
    
    if args.dry_run:
        print("\n[Dry run] 不實際更新。")
        print(f"\n預估時間（batch-size={args.batch_size}）：")
        n_batches = (len(to_update) + args.batch_size - 1) // args.batch_size
        print(f"  批次數：{n_batches}")
        print(f"  每批約 {args.batch_size * 0.5:.0f} 秒")
        print(f"  總計約 {n_batches * args.batch_size * 0.5 / 60:.0f} 分鐘")
        return
    
    # Update in batches
    print(f"\n開始更新（batch-size={args.batch_size}）...")
    print("（可隨時 Ctrl-C 中斷，下次會接續）\n")
    
    all_success = []
    all_failed = []
    
    n_batches = (len(to_update) + args.batch_size - 1) // args.batch_size
    
    for batch_idx in range(n_batches):
        start_idx = batch_idx * args.batch_size
        end_idx = min(start_idx + args.batch_size, len(to_update))
        batch = to_update[start_idx:end_idx]
        
        print(f"批次 {batch_idx + 1}/{n_batches} ({len(batch)} 檔)")
        result = update_batch(batch, args.cache_dir)
        
        all_success.extend(result['success'])
        all_failed.extend(result['failed'])
        
        print(f"  本批：{len(result['success'])} 成功，{len(result['failed'])} 失敗")
        
        # Pause between batches
        if batch_idx < n_batches - 1:
            print("  休息 3 秒 ...")
            time.sleep(3)
        print()
    
    # Summary
    print("=" * 70)
    print("更新完成")
    print("=" * 70)
    print(f"成功：{len(all_success)} 檔")
    print(f"失敗：{len(all_failed)} 檔")
    
    if all_failed:
        print(f"\n失敗的標的（可稍後重試）：")
        for symbol, error in all_failed[:20]:
            print(f"  {symbol}: {error}")
        if len(all_failed) > 20:
            print(f"  ... 還有 {len(all_failed) - 20} 檔")
    
    # Final check
    final_status = check_cache_freshness(args.cache_dir, args.days_threshold)
    print(f"\n最終狀態：")
    print(f"  最新：{len(final_status['fresh'])}/{n_total} 檔")
    print(f"  完成度：{len(final_status['fresh']) / n_total * 100:.1f}%")
    
    if len(final_status['stale']) == 0:
        print("\n✅ 所有標的都已更新到最新。")
    else:
        print(f"\n⏳ 還有 {len(final_status['stale'])} 檔未更新，可重新執行此腳本。")


if __name__ == "__main__":
    main()

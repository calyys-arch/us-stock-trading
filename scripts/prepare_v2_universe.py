#!/usr/bin/env python3
"""Prepare the expanded universe for Strategy V2 (cross-sectional momentum).

V2 needs a larger universe (500+ stocks) for cross-sectional signals to work.
This script fetches S&P 500 constituents and downloads their price history.

Usage:
    # Dry run: see what would be fetched
    .venv/bin/python scripts/prepare_v2_universe.py --dry-run

    # Fetch in batches (safe, resumable)
    .venv/bin/python scripts/prepare_v2_universe.py --batch-size 20

    # Fetch all at once (not recommended, may hit rate limits)
    .venv/bin/python scripts/prepare_v2_universe.py --batch-size 500

INCREMENTAL DESIGN. This script is designed to be run multiple times:
  - Already-cached symbols are skipped
  - Failed symbols are logged and can be retried later
  - Progress is printed so you can Ctrl-C and resume

RATE LIMITING. yfinance has informal rate limits (~2000 requests/hour).
Fetching 500 symbols takes ~500 requests. Running with --batch-size 20 and
waiting between batches is the safe approach.

UNIVERSE SOURCE. We fetch the current S&P 500 list from Wikipedia. This has
survivorship bias (2026 constituents, not point-in-time), but is acceptable for
V2 proof-of-concept. If V2 works, we'll build true PIT data later.
"""
import argparse
import time
from pathlib import Path
from datetime import datetime
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from python.data.price_cache import get_cached_price_panel
from python.simulation.hist_data_us import build_price_panel


def fetch_sp500_list() -> list[str]:
    """Fetch current S&P 500 constituents from Wikipedia.
    
    Returns:
        List of ticker symbols (e.g., ['AAPL', 'MSFT', ...])
    """
    import pandas as pd
    
    print("抓取 S&P 500 成份股列表 ...")
    
    # Wikipedia maintains a live table of S&P 500 constituents
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    
    try:
        tables = pd.read_html(url)
        sp500_table = tables[0]  # First table is the constituents
        symbols = sp500_table['Symbol'].str.replace('.', '-').tolist()
        
        print(f"  找到 {len(symbols)} 檔成份股")
        return symbols
    except Exception as exc:
        raise RuntimeError(f"無法抓取 S&P 500 列表: {exc}") from exc


def check_cached_status(symbols: list[str], cache_dir: Path) -> dict:
    """Check which symbols are already cached and which need fetching.
    
    Returns:
        {
            'cached': list of symbols with data,
            'missing': list of symbols without data,
            'n_cached': int,
            'n_missing': int,
        }
    """
    cached = []
    missing = []
    
    for symbol in symbols:
        csv_path = cache_dir / f"{symbol}.csv"
        if csv_path.exists():
            # Check if file has sufficient data (at least 100 rows = ~6 months)
            try:
                import pandas as pd
                df = pd.read_csv(csv_path)
                if len(df) >= 100:
                    cached.append(symbol)
                else:
                    missing.append(symbol)
            except Exception:
                missing.append(symbol)
        else:
            missing.append(symbol)
    
    return {
        'cached': cached,
        'missing': missing,
        'n_cached': len(cached),
        'n_missing': len(missing),
    }


def fetch_batch(symbols: list[str], start: str, end: str, 
                cache_dir: Path) -> dict:
    """Fetch a batch of symbols and cache them.
    
    Returns:
        {
            'success': list of successfully fetched symbols,
            'failed': list of failed symbols,
        }
    """
    success = []
    failed = []
    
    for i, symbol in enumerate(symbols, 1):
        print(f"  [{i}/{len(symbols)}] 抓取 {symbol} ...", end=" ", flush=True)
        
        try:
            # Use build_price_panel which handles caching
            panel, quality, sources = build_price_panel(
                [symbol], start, end, cache_dir=cache_dir
            )
            
            if symbol in panel.columns and len(panel[symbol].dropna()) >= 100:
                print(f"✓ ({len(panel)} 日)")
                success.append(symbol)
            else:
                print("✗ (資料不足)")
                failed.append(symbol)
        except Exception as exc:
            print(f"✗ ({exc})")
            failed.append(symbol)
        
        # Small delay to be nice to yfinance
        time.sleep(0.1)
    
    return {'success': success, 'failed': failed}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--batch-size", type=int, default=20,
                    help="Number of symbols to fetch per batch (default: 20)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Show what would be fetched without actually fetching")
    ap.add_argument("--cache-dir", type=Path, default=Path("data/history"),
                    help="Cache directory for price data")
    ap.add_argument("--start", default="2016-01-01",
                    help="Start date for price history")
    ap.add_argument("--end", default="2026-12-31",
                    help="End date for price history")
    args = ap.parse_args()
    
    print("=" * 70)
    print("準備 V2 Universe：S&P 500 價格數據")
    print("=" * 70)
    print()
    
    # Fetch S&P 500 list
    sp500 = fetch_sp500_list()
    
    # Check cache status
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    status = check_cached_status(sp500, args.cache_dir)
    
    print(f"\n快取狀態：")
    print(f"  已快取：{status['n_cached']} 檔")
    print(f"  需抓取：{status['n_missing']} 檔")
    
    if status['n_missing'] == 0:
        print("\n✅ 所有標的都已快取，無需抓取。")
        return
    
    # Show what would be fetched
    print(f"\n待抓取標的（{status['n_missing']} 檔）：")
    for i in range(0, len(status['missing']), 10):
        batch = status['missing'][i:i+10]
        print(f"  {', '.join(batch)}")
    
    if args.dry_run:
        print("\n[Dry run] 不實際抓取。")
        print(f"\n預估時間（batch-size={args.batch_size}）：")
        n_batches = (status['n_missing'] + args.batch_size - 1) // args.batch_size
        print(f"  批次數：{n_batches}")
        print(f"  每批約 {args.batch_size * 0.5:.0f} 秒")
        print(f"  總計約 {n_batches * args.batch_size * 0.5 / 60:.0f} 分鐘")
        return
    
    # Fetch in batches
    print(f"\n開始抓取（batch-size={args.batch_size}）...")
    print("（可隨時 Ctrl-C 中斷，下次會接續）\n")
    
    all_success = []
    all_failed = []
    
    missing = status['missing']
    n_batches = (len(missing) + args.batch_size - 1) // args.batch_size
    
    for batch_idx in range(n_batches):
        start_idx = batch_idx * args.batch_size
        end_idx = min(start_idx + args.batch_size, len(missing))
        batch = missing[start_idx:end_idx]
        
        print(f"批次 {batch_idx + 1}/{n_batches} ({len(batch)} 檔)")
        result = fetch_batch(batch, args.start, args.end, args.cache_dir)
        
        all_success.extend(result['success'])
        all_failed.extend(result['failed'])
        
        print(f"  本批：{len(result['success'])} 成功，{len(result['failed'])} 失敗")
        
        # Pause between batches (except for the last one)
        if batch_idx < n_batches - 1:
            print("  休息 5 秒 ...")
            time.sleep(5)
        print()
    
    # Summary
    print("=" * 70)
    print("抓取完成")
    print("=" * 70)
    print(f"成功：{len(all_success)} 檔")
    print(f"失敗：{len(all_failed)} 檔")
    
    if all_failed:
        print(f"\n失敗的標的（可稍後重試）：")
        for symbol in all_failed:
            print(f"  {symbol}")
    
    # Final status
    final_status = check_cached_status(sp500, args.cache_dir)
    print(f"\n最終狀態：")
    print(f"  已快取：{final_status['n_cached']}/{len(sp500)} 檔")
    print(f"  完成度：{final_status['n_cached'] / len(sp500) * 100:.1f}%")
    
    if final_status['n_cached'] >= 450:  # At least 450 out of 500
        print("\n✅ 數據已足夠，可以開始 V2 回測。")
    else:
        print(f"\n⏳ 建議至少 450 檔，目前 {final_status['n_cached']} 檔。")
        print("   可重新執行此腳本補齊剩餘標的。")


if __name__ == "__main__":
    main()

"""Debug script to investigate chunk planning issues with remote datasets."""
from __future__ import annotations

import os
from datetime import datetime, timedelta
from pprint import pprint

import polars as pl
import pytest

import rainbear


def test_debug_chunk_planning_remote() -> None:
    """Debug chunk planning for the remote dataset."""
    remote_ds = os.environ.get("RAINBEAR_REMOTE_MEM")
    if not remote_ds:
        pytest.skip(
            "Set RAINBEAR_REMOTE_MEM=<remote_ds> to run this test against a remote dataset."
        )
    
    # Test filter expression
    predicate = (
        (pl.col("time") == datetime(2025, 12, 30, 0, 0, 0))
        & (pl.col("lead_time") == timedelta(hours=1))
        & (pl.col("y") == 0)
        & (pl.col("x") == 0)
    )
    
    print("\n" + "="*80)
    print("CHUNK PLANNING DEBUG")
    print("="*80)
    
    result = rainbear._core._debug_chunk_planning(remote_ds, predicate)
    
    # Print error first if any
    if "error" in result:
        print(f"\n*** ERROR: {result['error']} ***\n")
    
    # Print metadata info for coordinate arrays
    print("\n--- COORDINATE ARRAY METADATA ---")
    meta = result.get("meta", {})
    for dim in result.get("dims", []):
        if dim in meta:
            arr = meta[dim]
            print(f"\n{dim}:")
            print(f"  shape: {arr.get('shape')}")
            print(f"  dtype: {arr.get('polars_dtype')}")
            print(f"  time_encoding: {arr.get('time_encoding')}")
    
    # Print dimension lengths
    print("\n--- DIMENSION LENGTHS ---")
    print(result.get("dim_lengths", {}))
    
    # Print expression AST (truncated)
    print("\n--- EXPRESSION AST (first 1000 chars) ---")
    expr_ast = result.get("expr_ast", "")
    print(expr_ast[:1000] + ("..." if len(expr_ast) > 1000 else ""))
    
    # Print lazy selection (truncated)
    print("\n--- LAZY SELECTION (first 2000 chars) ---")
    lazy_sel = result.get("lazy_selection", "")
    print(lazy_sel[:2000] + ("..." if len(lazy_sel) > 2000 else ""))
    
    # Print resolution requests
    print("\n--- RESOLUTION REQUESTS ---")
    print(f"Number of requests: {result.get('num_requests', 0)}")
    for req in result.get("resolution_requests", [])[:10]:
        print(f"  {req}")
    if len(result.get("resolution_requests", [])) > 10:
        print("  ... (truncated)")
    
    # Print materialized selection (truncated)
    print("\n--- MATERIALIZED SELECTION (first 2000 chars) ---")
    mat_sel = result.get("materialized_selection", "")
    print(mat_sel[:2000] + ("..." if len(mat_sel) > 2000 else ""))
    
    # Print selection summary
    print("\n--- SELECTION SUMMARY ---")
    print(f"Summary: {result.get('selection_summary', 'N/A')}")
    print(f"Coord reads: {result.get('coord_reads', 'N/A')}")


def test_debug_time_coord_array() -> None:
    """Debug the 'time' coordinate array specifically."""
    remote_ds = os.environ.get("RAINBEAR_REMOTE_MEM")
    if not remote_ds:
        pytest.skip(
            "Set RAINBEAR_REMOTE_MEM=<remote_ds> to run this test against a remote dataset."
        )
    
    print("\n" + "="*80)
    print("TIME COORDINATE ARRAY DEBUG")
    print("="*80)
    
    result = rainbear._core._debug_coord_array(remote_ds, "time", 10)
    pprint(result)
    
    # Also test lead_time
    print("\n" + "="*80)
    print("LEAD_TIME COORDINATE ARRAY DEBUG")
    print("="*80)
    
    result = rainbear._core._debug_coord_array(remote_ds, "lead_time", 10)
    pprint(result)


def test_debug_literal_conversion() -> None:
    """Debug how datetime literals are converted."""
    remote_ds = os.environ.get("RAINBEAR_REMOTE_MEM")
    if not remote_ds:
        pytest.skip(
            "Set RAINBEAR_REMOTE_MEM=<remote_ds> to run this test against a remote dataset."
        )
    
    print("\n" + "="*80)
    print("LITERAL CONVERSION DEBUG - datetime")
    print("="*80)
    
    test_dt = datetime(2025, 12, 30, 0, 0, 0)
    print(f"Test value: {test_dt}")
    print(f"As ns since epoch: {int(test_dt.timestamp() * 1e9)}")
    
    result = rainbear._core._debug_literal_conversion(remote_ds, "time", test_dt)
    pprint(result)
    
    print("\n" + "="*80)
    print("LITERAL CONVERSION DEBUG - timedelta")
    print("="*80)
    
    test_td = timedelta(hours=1)
    print(f"Test value: {test_td}")
    print(f"As ns: {int(test_td.total_seconds() * 1e9)}")
    
    result = rainbear._core._debug_literal_conversion(remote_ds, "lead_time", test_td)
    pprint(result)


if __name__ == "__main__":
    # Run with RAINBEAR_REMOTE_MEM=1 to enable tests
    os.environ["RAINBEAR_REMOTE_MEM"] = "1"
    
    test_debug_time_coord_array()
    test_debug_literal_conversion()
    test_debug_chunk_planning_remote()

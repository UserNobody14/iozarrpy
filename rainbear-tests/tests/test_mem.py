from datetime import datetime, timedelta

import polars as pl

import rainbear


def test_mem():
    scz = rainbear.scan_zarr("/home/benjamin/Code/hs/hzarrz/zarrgentest/output-datasets/hrrr_grid_dataset.zarr")
    scz = scz.filter(
        (pl.col("time") == datetime(2024, 1, 1, 0, 0, 0)) &
        # lead_time now recognized as Duration (units: hours)
        (pl.col("lead_time") == timedelta(hours=1)) &
        (pl.col("y") >= 5) & 
        (pl.col("y") <= 10) &
        (pl.col("x") >= 5) &
        (pl.col("x") <= 15)
    ).select(
        ["time", "latitude", "longitude", "2m_temperature"]
    )
    print("reached here 1")
    df = scz.collect()
    print("reached here 2")
    print(df)
    print("reached here 3")


if __name__ == "__main__":
    test_mem()
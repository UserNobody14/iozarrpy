"""Build small Zarr stores that stress vendor-prefixed numcodecs (V3) and a Zarr v2 baseline.

Forecast-like layout (time, lead_time, latitude, longitude) with **BitRound** only in the
integration path: zarrs currently rejects ``numcodecs.fixedscaleoffset`` when the array
metadata dtype is float32 (decode dtype check). NumPy-style FSO dtype normalization is
covered in Rust (`codec_compat::dtype` tests).
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Self, cast

import numpy as np
import xarray as xr
from numcodecs import Zlib
from zarr.codecs.numcodecs import BitRound
from zarr.codecs.numcodecs._codecs import CODEC_PREFIX, _NumcodecsArrayArrayCodec
from zarr.core.common import JSON
from zarr.registry import register_codec

import rainbear

VENDOR_PREFIX = "testvendor."
V3_CODEC_BITROUND = f"{VENDOR_PREFIX}numcodecs.bitround"


class _NumcodecsV3BitRoundAliasForXarray(_NumcodecsArrayArrayCodec):
    """Map vendor ``name`` in JSON to ``numcodecs.bitround`` for zarr-python."""

    codec_name = CODEC_PREFIX + "bitround"
    _stored_prefix = V3_CODEC_BITROUND.removesuffix("numcodecs.bitround")

    def __init__(self, **codec_config: JSON) -> None:
        super().__init__(**codec_config)

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        data = data.copy()
        data["name"] = cls.codec_name
        return super().from_dict(data)

    def to_dict(self) -> dict[str, JSON]:
        data = super().to_dict()
        data["name"] = self._stored_prefix + cast(str, data["name"])
        return data


def register_xarray_vendor_v3_codecs() -> None:
    register_codec(V3_CODEC_BITROUND, _NumcodecsV3BitRoundAliasForXarray)


def configure_rainbear_vendor_codecs() -> None:
    rainbear.configure_zarr_codecs(
        aliases={
            V3_CODEC_BITROUND: "numcodecs.bitround",
        },
    )


def patch_v3_zarr_json_vendor_bitround(root: Path) -> None:
    old = '"name": "numcodecs.bitround"'
    new = f'"name": "{V3_CODEC_BITROUND}"'
    for zjson in root.rglob("zarr.json"):
        text = zjson.read_text(encoding="utf-8").replace(old, new)
        zjson.write_text(text, encoding="utf-8")


def write_forecast_like_v3_vendor_codecs(path: str | Path) -> None:
    """Zarr v3 group: 1D coords; ``dewpoint`` and ``air_temperature`` use BitRound (vendor names)."""
    path = Path(path)
    if path.exists():
        shutil.rmtree(path)

    nt, nl, nlat, nlon = 2, 2, 3, 3
    rng = np.random.default_rng(42)
    dewpoint = (
        (280.0 + 15.0 * rng.random((nt, nl, nlat, nlon))).astype(np.float32)
    )
    air_temperature = (
        (260.0 + 20.0 * rng.random((nt, nl, nlat, nlon))).astype(np.float32)
    )

    time = np.array(
        [np.datetime64("2024-01-01T00:00:00"), np.datetime64("2024-01-01T06:00:00")],
        dtype="datetime64[ns]",
    )
    lead_time = np.array(
        [np.timedelta64(0, "h"), np.timedelta64(3, "h")],
        dtype="timedelta64[ns]",
    )
    latitude = np.array([10.0, 15.0, 20.0], dtype=np.float64)
    longitude = np.array([100.0, 105.0, 110.0], dtype=np.float64)

    ds = xr.Dataset(
        data_vars={
            "dewpoint": (
                ["time", "lead_time", "latitude", "longitude"],
                dewpoint,
            ),
            "air_temperature": (
                ["time", "lead_time", "latitude", "longitude"],
                air_temperature,
            ),
        },
        coords={
            "time": ("time", time),
            "lead_time": ("lead_time", lead_time),
            "latitude": ("latitude", latitude),
            "longitude": ("longitude", longitude),
        },
    )

    encoding = {
        "dewpoint": {
            "chunks": (1, 1, 2, 2),
            "compressors": [],
            "filters": [BitRound(keepbits=5)],
        },
        "air_temperature": {
            "chunks": (1, 1, 2, 2),
            "compressors": [],
            "filters": [BitRound(keepbits=3)],
        },
    }
    ds.to_zarr(
        path,
        zarr_format=3,
        encoding=encoding,
        consolidated=True,
    )
    patch_v3_zarr_json_vendor_bitround(Path(path))


def write_v2_zlib(path: str | Path) -> None:
    """Zarr v2 store: ``temp`` compressed with zlib (format-2 baseline).

    zarrs does not ship a Zarr V2 plugin for ``bitround`` (only V3); zlib exercises the
    v2 array path alongside the vendor BitRound v3 fixture.
    """
    path = Path(path)
    if path.exists():
        shutil.rmtree(path)

    t = np.arange(6, dtype=np.int64)
    temp_data = np.linspace(270.0, 300.0, 6, dtype=np.float32)
    ds = xr.Dataset(
        data_vars={"temp": (["t"], temp_data)},
        coords={"t": t},
    )
    ds.to_zarr(
        path,
        zarr_format=2,
        consolidated=False,
        encoding={
            "temp": {
                "chunks": (3,),
                "compressor": Zlib(level=1),
            },
        },
    )

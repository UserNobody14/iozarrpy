"""Comprehensive expression variant tests for predicate pushdown.

This test module is designed to rigorously test every Polars expression variant
supported by the chunk planner, using carefully designed datasets and predicates
that constrain multiple dimensions simultaneously and return distinctive,
non-round chunk counts that cannot be achieved by accident or fallthrough.

Key principles:
1. Compute expected chunks mathematically before calling the planner
2. Assert exact chunk count (not just "less than total")
3. Assert exact chunk index sets where feasible
4. Fail if planner returns all chunks (guard against fallthrough)

Dataset: 3D with prime-factored chunk grid (7x5x3=105 chunks)
- Dimension a: 70 elements, chunk size 10 -> 7 chunks (indices 0-6)
- Dimension b: 50 elements, chunk size 10 -> 5 chunks (indices 0-4)
- Dimension c: 30 elements, chunk size 10 -> 3 chunks (indices 0-2)

Coordinate values are designed to map precisely to chunks:
- a coords: 0..69 (chunk i contains values i*10..(i+1)*10-1)
- b coords: 0..49
- c coords: 0..29
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from rainbear import ZarrBackend

if TYPE_CHECKING:
    from conftest import ComprehensiveDatasetInfo

    from rainbear._core import SelectedChunksDebugReturn



# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _chunk_indices(chunks: SelectedChunksDebugReturn) -> set[tuple[int, ...]]:
    """Extract chunk indices as a set of tuples."""
    # Find a grid that includes "data"
    for grid in chunks["grids"]:
        if "data" in grid["variables"]:
            return {tuple(int(x) for x in d["indices"]) for d in grid["chunks"]}
    raise ValueError(f"No grid found for variable 'data' in {chunks}")


def _coord_to_chunk(coord_value: int, chunk_size: int = 10) -> int:
    """Map a coordinate value to its chunk index."""
    return coord_value // chunk_size


def _expected_chunks_for_range(
    dim_idx: int,
    min_val: int,
    max_val: int,
    chunk_grid: tuple[int, int, int],
    chunk_size: int = 10,
) -> set[tuple[int, ...]]:
    """Compute expected chunk indices for a range constraint on one dimension.

    Args:
        dim_idx: Which dimension (0=a, 1=b, 2=c)
        min_val: Minimum coordinate value (inclusive)
        max_val: Maximum coordinate value (inclusive)
        chunk_grid: (a_chunks, b_chunks, c_chunks)
        chunk_size: Size of each chunk
    """
    min_chunk = min_val // chunk_size
    max_chunk = max_val // chunk_size

    result: set[tuple[int, ...]] = set()
    for a in range(chunk_grid[0]):
        for b in range(chunk_grid[1]):
            for c in range(chunk_grid[2]):
                coords = [a, b, c]
                if min_chunk <= coords[dim_idx] <= max_chunk:
                    result.add((a, b, c))
    return result


def _expected_chunks_multidim(
    constraints: list[tuple[int, int, int]],  # (dim_idx, min_val, max_val)
    chunk_grid: tuple[int, int, int],
    chunk_size: int = 10,
) -> set[tuple[int, ...]]:
    """Compute expected chunks for multiple dimension constraints (AND)."""
    if not constraints:
        # No constraints = all chunks
        return {
            (a, b, c)
            for a in range(chunk_grid[0])
            for b in range(chunk_grid[1])
            for c in range(chunk_grid[2])
        }

    result: set[tuple[int, ...]] = set()
    for a in range(chunk_grid[0]):
        for b in range(chunk_grid[1]):
            for c in range(chunk_grid[2]):
                coords = [a, b, c]
                matches = True
                for dim_idx, min_val, max_val in constraints:
                    min_chunk = min_val // chunk_size
                    max_chunk = max_val // chunk_size
                    if not (min_chunk <= coords[dim_idx] <= max_chunk):
                        matches = False
                        break
                if matches:
                    result.add((a, b, c))
    return result


# ---------------------------------------------------------------------------
# Test Classes
# ---------------------------------------------------------------------------


class TestComparison:
    """Tests for comparison operators: Eq, Gt, GtEq, Lt, LtEq."""

    def test_eq_single_value_single_chunk(
        self, comprehensive_3d_dataset: "ComprehensiveDatasetInfo"
    ) -> None:
        """a == 15 should select only chunk a=1, giving 1*5*3 = 15 chunks."""
        url = comprehensive_3d_dataset.path
        pred = pl.col("a") == 15
        chunks = ZarrBackend.from_url(url).selected_chunks_debug( pred)
        idxs = _chunk_indices(chunks)

        # Value 15 is in chunk 1 (values 10-19)
        expected = {(1, b, c) for b in range(5) for c in range(3)}
        assert len(idxs) == 15
        assert idxs == expected
        assert len(idxs) < comprehensive_3d_dataset.total_chunks

    def test_eq_boundary_value_chunk_start(
        self, comprehensive_3d_dataset: "ComprehensiveDatasetInfo"
    ) -> None:
        """a == 20 (chunk boundary) should select chunk a=2."""
        url = comprehensive_3d_dataset.path
        pred = pl.col("a") == 20
        chunks = ZarrBackend.from_url(url).selected_chunks_debug( pred)
        idxs = _chunk_indices(chunks)

        expected = {(2, b, c) for b in range(5) for c in range(3)}
        assert len(idxs) == 15
        assert idxs == expected

    def test_eq_first_chunk(
        self, comprehensive_3d_dataset: "ComprehensiveDatasetInfo"
    ) -> None:
        """a == 0 should select only chunk a=0."""
        url = comprehensive_3d_dataset.path
        pred = pl.col("a") == 0
        chunks = ZarrBackend.from_url(url).selected_chunks_debug( pred)
        idxs = _chunk_indices(chunks)

        expected = {(0, b, c) for b in range(5) for c in range(3)}
        assert len(idxs) == 15
        assert idxs == expected

    def test_eq_last_chunk(
        self, comprehensive_3d_dataset: "ComprehensiveDatasetInfo"
    ) -> None:
        """a == 69 (last value) should select chunk a=6."""
        url = comprehensive_3d_dataset.path
        pred = pl.col("a") == 69
        chunks = ZarrBackend.from_url(url).selected_chunks_debug( pred)
        idxs = _chunk_indices(chunks)

        expected = {(6, b, c) for b in range(5) for c in range(3)}
        assert len(idxs) == 15
        assert idxs == expected

    def test_lt_narrows_to_prefix(
        self, comprehensive_3d_dataset: "ComprehensiveDatasetInfo"
    ) -> None:
        """a < 20 should select chunks a=0,1 giving 2*5*3 = 30 chunks."""
        url = comprehensive_3d_dataset.path
        pred = pl.col("a") < 20
        chunks = ZarrBackend.from_url(url).selected_chunks_debug( pred)
        idxs = _chunk_indices(chunks)

        expected = {(a, b, c) for a in range(2) for b in range(5) for c in range(3)}
        assert len(idxs) == 30
        assert idxs == expected

    def test_lteq_includes_boundary(
        self, comprehensive_3d_dataset: "ComprehensiveDatasetInfo"
    ) -> None:
        """a <= 19 should select chunks a=0,1 (values 0-19)."""
        url = comprehensive_3d_dataset.path
        pred = pl.col("a") <= 19
        chunks = ZarrBackend.from_url(url).selected_chunks_debug( pred)
        idxs = _chunk_indices(chunks)

        expected = {(a, b, c) for a in range(2) for b in range(5) for c in range(3)}
        assert len(idxs) == 30
        assert idxs == expected

    def test_gt_narrows_to_suffix(
        self, comprehensive_3d_dataset: "ComprehensiveDatasetInfo"
    ) -> None:
        """a > 49 should select chunks a=5,6 giving 2*5*3 = 30 chunks."""
        url = comprehensive_3d_dataset.path
        pred = pl.col("a") > 49
        chunks = ZarrBackend.from_url(url).selected_chunks_debug( pred)
        idxs = _chunk_indices(chunks)

        expected = {(a, b, c) for a in range(5, 7) for b in range(5) for c in range(3)}
        assert len(idxs) == 30
        assert idxs == expected

    def test_gteq_includes_boundary(
        self, comprehensive_3d_dataset: "ComprehensiveDatasetInfo"
    ) -> None:
        """a >= 50 should select chunks a=5,6."""
        url = comprehensive_3d_dataset.path
        pred = pl.col("a") >= 50
        chunks = ZarrBackend.from_url(url).selected_chunks_debug( pred)
        idxs = _chunk_indices(chunks)

        expected = {(a, b, c) for a in range(5, 7) for b in range(5) for c in range(3)}
        assert len(idxs) == 30
        assert idxs == expected

    def test_comparison_on_b_dimension(
        self, comprehensive_3d_dataset: "ComprehensiveDatasetInfo"
    ) -> None:
        """b == 25 should select chunk b=2 giving 7*1*3 = 21 chunks."""
        url = comprehensive_3d_dataset.path
        pred = pl.col("b") == 25
        chunks = ZarrBackend.from_url(url).selected_chunks_debug( pred)
        idxs = _chunk_indices(chunks)

        expected = {(a, 2, c) for a in range(7) for c in range(3)}
        assert len(idxs) == 21
        assert idxs == expected

    def test_comparison_on_c_dimension(
        self, comprehensive_3d_dataset: "ComprehensiveDatasetInfo"
    ) -> None:
        """c < 10 should select chunk c=0 giving 7*5*1 = 35 chunks."""
        url = comprehensive_3d_dataset.path
        pred = pl.col("c") < 10
        chunks = ZarrBackend.from_url(url).selected_chunks_debug( pred)
        idxs = _chunk_indices(chunks)

        expected = {(a, b, 0) for a in range(7) for b in range(5)}
        assert len(idxs) == 35
        assert idxs == expected

    def test_reversed_comparison_literal_first(
        self, comprehensive_3d_dataset: "ComprehensiveDatasetInfo"
    ) -> None:
        """15 == a should work the same as a == 15."""
        url = comprehensive_3d_dataset.path
        pred = pl.lit(15) == pl.col("a")
        chunks = ZarrBackend.from_url(url).selected_chunks_debug( pred)
        idxs = _chunk_indices(chunks)

        expected = {(1, b, c) for b in range(5) for c in range(3)}
        assert len(idxs) == 15
        assert idxs == expected

    def test_reversed_lt_literal_first(
        self, comprehensive_3d_dataset: "ComprehensiveDatasetInfo"
    ) -> None:
        """20 > a is equivalent to a < 20."""
        url = comprehensive_3d_dataset.path
        pred = pl.lit(20) > pl.col("a")
        chunks = ZarrBackend.from_url(url).selected_chunks_debug( pred)
        idxs = _chunk_indices(chunks)

        expected = {(a, b, c) for a in range(2) for b in range(5) for c in range(3)}
        assert len(idxs) == 30
        assert idxs == expected


class TestLogical:
    """Tests for logical operators: And, Or, Xor, Not."""

    def test_and_two_dimensions(
        self, comprehensive_3d_dataset: "ComprehensiveDatasetInfo"
    ) -> None:
        """(a < 20) & (b < 20) should give 2*2*3 = 12 chunks."""
        url = comprehensive_3d_dataset.path
        pred = (pl.col("a") < 20) & (pl.col("b") < 20)
        chunks = ZarrBackend.from_url(url).selected_chunks_debug( pred)
        idxs = _chunk_indices(chunks)

        expected = {(a, b, c) for a in range(2) for b in range(2) for c in range(3)}
        assert len(idxs) == 12
        assert idxs == expected
        assert len(idxs) < comprehensive_3d_dataset.total_chunks

    def test_and_three_dimensions(
        self, comprehensive_3d_dataset: "ComprehensiveDatasetInfo"
    ) -> None:
        """(a < 20) & (b < 20) & (c < 20) should give 2*2*2 = 8 chunks."""
        url = comprehensive_3d_dataset.path
        pred = (pl.col("a") < 20) & (pl.col("b") < 20) & (pl.col("c") < 20)
        chunks = ZarrBackend.from_url(url).selected_chunks_debug( pred)
        idxs = _chunk_indices(chunks)

        expected = {(a, b, c) for a in range(2) for b in range(2) for c in range(2)}
        assert len(idxs) == 8
        assert idxs == expected

    def test_and_same_dimension_range(
        self, comprehensive_3d_dataset: "ComprehensiveDatasetInfo"
    ) -> None:
        """(a >= 20) & (a < 40) should give 2*5*3 = 30 chunks (chunks 2,3)."""
        url = comprehensive_3d_dataset.path
        pred = (pl.col("a") >= 20) & (pl.col("a") < 40)
        chunks = ZarrBackend.from_url(url).selected_chunks_debug( pred)
        idxs = _chunk_indices(chunks)

        expected = {(a, b, c) for a in range(2, 4) for b in range(5) for c in range(3)}
        assert len(idxs) == 30
        assert idxs == expected

    def test_and_single_chunk_intersection(
        self, comprehensive_3d_dataset: "ComprehensiveDatasetInfo"
    ) -> None:
        """(a == 15) & (b == 25) & (c == 15) should give exactly 1 chunk."""
        url = comprehensive_3d_dataset.path
        pred = (pl.col("a") == 15) & (pl.col("b") == 25) & (pl.col("c") == 15)
        chunks = ZarrBackend.from_url(url).selected_chunks_debug( pred)
        idxs = _chunk_indices(chunks)

        assert len(idxs) == 1
        assert idxs == {(1, 2, 1)}

    def test_or_disjoint_dimensions(
        self, comprehensive_3d_dataset: "ComprehensiveDatasetInfo"
    ) -> None:
        """(a == 15) | (b == 25) should union chunks."""
        url = comprehensive_3d_dataset.path
        pred = (pl.col("a") == 15) | (pl.col("b") == 25)
        chunks = ZarrBackend.from_url(url).selected_chunks_debug( pred)
        idxs = _chunk_indices(chunks)

        # a=1: 1*5*3 = 15 chunks
        # b=2: 7*1*3 = 21 chunks
        # Overlap: (1,2,c) for c in 0-2 = 3 chunks
        # Total: 15 + 21 - 3 = 33 chunks
        a_chunks = {(1, b, c) for b in range(5) for c in range(3)}
        b_chunks = {(a, 2, c) for a in range(7) for c in range(3)}
        expected = a_chunks | b_chunks
        assert len(idxs) == 33
        assert idxs == expected

    def test_or_same_dimension_disjoint(
        self, comprehensive_3d_dataset: "ComprehensiveDatasetInfo"
    ) -> None:
        """(a == 5) | (a == 55) should give 2*5*3 = 30 chunks."""
        url = comprehensive_3d_dataset.path
        pred = (pl.col("a") == 5) | (pl.col("a") == 55)
        chunks = ZarrBackend.from_url(url).selected_chunks_debug( pred)
        idxs = _chunk_indices(chunks)

        # a=0 (value 5) and a=5 (value 55)
        expected = {(0, b, c) for b in range(5) for c in range(3)} | {
            (5, b, c) for b in range(5) for c in range(3)
        }
        assert len(idxs) == 30
        assert idxs == expected

    def test_xor_excludes_overlap(
        self, comprehensive_3d_dataset: "ComprehensiveDatasetInfo"
    ) -> None:
        """(a < 20) ^ (b < 20) should exclude the intersection."""
        url = comprehensive_3d_dataset.path
        pred = (pl.col("a") < 20) ^ (pl.col("b") < 20)
        chunks = ZarrBackend.from_url(url).selected_chunks_debug( pred)
        idxs = _chunk_indices(chunks)

        # a<20: chunks a=0,1 = 2*5*3 = 30
        # b<20: chunks b=0,1 = 7*2*3 = 42
        # Intersection: a in {0,1} AND b in {0,1} = 2*2*3 = 12
        # XOR: (30-12) + (42-12) = 18 + 30 = 48
        a_set = {(a, b, c) for a in range(2) for b in range(5) for c in range(3)}
        b_set = {(a, b, c) for a in range(7) for b in range(2) for c in range(3)}
        expected = (a_set - b_set) | (b_set - a_set)
        assert len(idxs) == 48
        assert idxs == expected

    def test_not_standalone_is_conservative(
        self, comprehensive_3d_dataset: "ComprehensiveDatasetInfo"
    ) -> None:
        """~(a == 15) cannot be represented precisely, should return all chunks."""
        url = comprehensive_3d_dataset.path
        pred = ~(pl.col("a") == 15)
        chunks = ZarrBackend.from_url(url).selected_chunks_debug( pred)
        idxs = _chunk_indices(chunks)

        # NOT alone is conservative - returns all chunks
        assert len(idxs) == comprehensive_3d_dataset.total_chunks

    def test_and_not_pattern_cuts_hole(
        self, comprehensive_3d_dataset: "ComprehensiveDatasetInfo"
    ) -> None:
        """(a < 30) & ~(a < 10) should select only chunk a=1,2."""
        url = comprehensive_3d_dataset.path
        pred = (pl.col("a") < 30) & ~(pl.col("a") < 10)
        chunks = ZarrBackend.from_url(url).selected_chunks_debug( pred)
        idxs = _chunk_indices(chunks)

        # a<30 gives chunks 0,1,2; a<10 gives chunk 0
        # Difference: chunks 1,2 = 2*5*3 = 30 chunks
        expected = {(a, b, c) for a in range(1, 3) for b in range(5) for c in range(3)}
        assert len(idxs) == 30
        assert idxs == expected

    def test_and_not_multidim_hole(
        self, comprehensive_3d_dataset: "ComprehensiveDatasetInfo"
    ) -> None:
        """(a < 30) & (b < 30) & ~((a < 10) & (b < 10)) cuts corner."""
        url = comprehensive_3d_dataset.path
        outer = (pl.col("a") < 30) & (pl.col("b") < 30)
        hole = (pl.col("a") < 10) & (pl.col("b") < 10)
        pred = outer & ~hole
        chunks = ZarrBackend.from_url(url).selected_chunks_debug( pred)
        idxs = _chunk_indices(chunks)

        # Outer: a in {0,1,2}, b in {0,1,2} = 3*3*3 = 27
        # Hole: a=0, b=0 = 1*1*3 = 3
        # Result: 27 - 3 = 24
        outer_set = {(a, b, c) for a in range(3) for b in range(3) for c in range(3)}
        hole_set = {(0, 0, c) for c in range(3)}
        expected = outer_set - hole_set
        assert len(idxs) == 24
        assert idxs == expected


class TestBooleanFunctions:
    """Tests for is_between, is_in, is_null, is_not_null."""

    def test_is_between_single_chunk(
        self, comprehensive_3d_dataset: "ComprehensiveDatasetInfo"
    ) -> None:
        """a.is_between(10, 19) should select chunk a=1 (15 chunks)."""
        url = comprehensive_3d_dataset.path
        pred = pl.col("a").is_between(10, 19)
        chunks = ZarrBackend.from_url(url).selected_chunks_debug( pred)
        idxs = _chunk_indices(chunks)

        expected = {(1, b, c) for b in range(5) for c in range(3)}
        assert len(idxs) == 15
        assert idxs == expected

    def test_is_between_spanning_chunks(
        self, comprehensive_3d_dataset: "ComprehensiveDatasetInfo"
    ) -> None:
        """a.is_between(15, 35) should select chunks a=1,2,3 (45 chunks)."""
        url = comprehensive_3d_dataset.path
        pred = pl.col("a").is_between(15, 35)
        chunks = ZarrBackend.from_url(url).selected_chunks_debug( pred)
        idxs = _chunk_indices(chunks)

        expected = {(a, b, c) for a in range(1, 4) for b in range(5) for c in range(3)}
        assert len(idxs) == 45
        assert idxs == expected

    def test_is_between_multidim(
        self, comprehensive_3d_dataset: "ComprehensiveDatasetInfo"
    ) -> None:
        """a.is_between(10, 29) & b.is_between(10, 29) = 2*2*3 = 12 chunks."""
        url = comprehensive_3d_dataset.path
        pred = pl.col("a").is_between(10, 29) & pl.col("b").is_between(10, 29)
        chunks = ZarrBackend.from_url(url).selected_chunks_debug( pred)
        idxs = _chunk_indices(chunks)

        expected = {(a, b, c) for a in range(1, 3) for b in range(1, 3) for c in range(3)}
        assert len(idxs) == 12
        assert idxs == expected

    def test_is_in_single_values(
        self, comprehensive_3d_dataset: "ComprehensiveDatasetInfo"
    ) -> None:
        """a.is_in([5, 25, 45]) should select chunks a=0,2,4."""
        url = comprehensive_3d_dataset.path
        pred = pl.col("a").is_in([5, 25, 45])
        chunks = ZarrBackend.from_url(url).selected_chunks_debug( pred)
        idxs = _chunk_indices(chunks)

        # Values 5,25,45 are in chunks 0,2,4
        expected = {(a, b, c) for a in [0, 2, 4] for b in range(5) for c in range(3)}
        assert len(idxs) == 45
        assert idxs == expected

    def test_is_in_same_chunk_values(
        self, comprehensive_3d_dataset: "ComprehensiveDatasetInfo"
    ) -> None:
        """a.is_in([10, 11, 12, 13]) all in same chunk should give 15 chunks."""
        url = comprehensive_3d_dataset.path
        pred = pl.col("a").is_in([10, 11, 12, 13])
        chunks = ZarrBackend.from_url(url).selected_chunks_debug( pred)
        idxs = _chunk_indices(chunks)

        # All values in chunk 1
        expected = {(1, b, c) for b in range(5) for c in range(3)}
        assert len(idxs) == 15
        assert idxs == expected

    def test_is_in_multidim(
        self, comprehensive_3d_dataset: "ComprehensiveDatasetInfo"
    ) -> None:
        """a.is_in([5, 15]) & b.is_in([5, 15]) = 2*2*3 = 12 chunks."""
        url = comprehensive_3d_dataset.path
        pred = pl.col("a").is_in([5, 15]) & pl.col("b").is_in([5, 15])
        chunks = ZarrBackend.from_url(url).selected_chunks_debug( pred)
        idxs = _chunk_indices(chunks)

        expected = {(a, b, c) for a in [0, 1] for b in [0, 1] for c in range(3)}
        assert len(idxs) == 12
        assert idxs == expected

    def test_is_null_is_conservative(
        self, comprehensive_3d_dataset: "ComprehensiveDatasetInfo"
    ) -> None:
        """is_null on a column returns all chunks (conservative)."""
        url = comprehensive_3d_dataset.path
        pred = pl.col("a").is_null()
        chunks = ZarrBackend.from_url(url).selected_chunks_debug( pred)
        idxs = _chunk_indices(chunks)

        # Conservative: returns all
        assert len(idxs) == comprehensive_3d_dataset.total_chunks

    def test_is_not_null_is_conservative(
        self, comprehensive_3d_dataset: "ComprehensiveDatasetInfo"
    ) -> None:
        """is_not_null on a column returns all chunks (conservative)."""
        url = comprehensive_3d_dataset.path
        pred = pl.col("a").is_not_null()
        chunks = ZarrBackend.from_url(url).selected_chunks_debug( pred)
        idxs = _chunk_indices(chunks)

        assert len(idxs) == comprehensive_3d_dataset.total_chunks

    def test_is_null_with_narrowing_and(
        self, comprehensive_3d_dataset: "ComprehensiveDatasetInfo"
    ) -> None:
        """(a < 20) & data.is_null() should still narrow by a."""
        url = comprehensive_3d_dataset.path
        pred = (pl.col("a") < 20) & pl.col("data").is_null()
        chunks = ZarrBackend.from_url(url).selected_chunks_debug( pred)
        idxs = _chunk_indices(chunks)

        # AND with is_null should still narrow
        expected = {(a, b, c) for a in range(2) for b in range(5) for c in range(3)}
        assert len(idxs) == 30
        assert idxs == expected


class TestHorizontalOps:
    """Tests for any_horizontal, all_horizontal."""

    def test_any_horizontal_union(
        self, comprehensive_3d_dataset: "ComprehensiveDatasetInfo"
    ) -> None:
        """any_horizontal([a==5, b==25]) should union the selections."""
        url = comprehensive_3d_dataset.path
        pred = pl.any_horizontal([pl.col("a") == 5, pl.col("b") == 25])
        chunks = ZarrBackend.from_url(url).selected_chunks_debug( pred)
        idxs = _chunk_indices(chunks)

        # a=0: 1*5*3 = 15, b=2: 7*1*3 = 21, overlap: 1*1*3 = 3
        # Total: 15 + 21 - 3 = 33
        a_set = {(0, b, c) for b in range(5) for c in range(3)}
        b_set = {(a, 2, c) for a in range(7) for c in range(3)}
        expected = a_set | b_set
        assert len(idxs) == 33
        assert idxs == expected

    def test_any_horizontal_three_exprs(
        self, comprehensive_3d_dataset: "ComprehensiveDatasetInfo"
    ) -> None:
        """any_horizontal with three expressions."""
        url = comprehensive_3d_dataset.path
        pred = pl.any_horizontal([pl.col("a") == 5, pl.col("b") == 25, pl.col("c") == 15])
        chunks = ZarrBackend.from_url(url).selected_chunks_debug( pred)
        idxs = _chunk_indices(chunks)

        a_set = {(0, b, c) for b in range(5) for c in range(3)}  # 15
        b_set = {(a, 2, c) for a in range(7) for c in range(3)}  # 21
        c_set = {(a, b, 1) for a in range(7) for b in range(5)}  # 35
        expected = a_set | b_set | c_set
        # 15 + 21 + 35 - overlaps
        assert idxs == expected
        assert len(idxs) < comprehensive_3d_dataset.total_chunks

    def test_all_horizontal_intersect(
        self, comprehensive_3d_dataset: "ComprehensiveDatasetInfo"
    ) -> None:
        """all_horizontal([a<20, b<20]) should intersect the selections."""
        url = comprehensive_3d_dataset.path
        pred = pl.all_horizontal([pl.col("a") < 20, pl.col("b") < 20])
        chunks = ZarrBackend.from_url(url).selected_chunks_debug( pred)
        idxs = _chunk_indices(chunks)

        expected = {(a, b, c) for a in range(2) for b in range(2) for c in range(3)}
        assert len(idxs) == 12
        assert idxs == expected

    def test_all_horizontal_three_exprs(
        self, comprehensive_3d_dataset: "ComprehensiveDatasetInfo"
    ) -> None:
        """all_horizontal with three expressions = 2*2*2 = 8 chunks."""
        url = comprehensive_3d_dataset.path
        pred = pl.all_horizontal([pl.col("a") < 20, pl.col("b") < 20, pl.col("c") < 20])
        chunks = ZarrBackend.from_url(url).selected_chunks_debug( pred)
        idxs = _chunk_indices(chunks)

        expected = {(a, b, c) for a in range(2) for b in range(2) for c in range(2)}
        assert len(idxs) == 8
        assert idxs == expected


class TestWrappers:
    """Tests for alias, cast, sort, slice, over, filter wrappers."""

    def test_alias_preserves_pushdown(
        self, comprehensive_3d_dataset: "ComprehensiveDatasetInfo"
    ) -> None:
        """(a == 15).alias("test") should still narrow to 15 chunks."""
        url = comprehensive_3d_dataset.path
        pred = (pl.col("a") == 15).alias("test")
        chunks = ZarrBackend.from_url(url).selected_chunks_debug( pred)
        idxs = _chunk_indices(chunks)

        expected = {(1, b, c) for b in range(5) for c in range(3)}
        assert len(idxs) == 15
        assert idxs == expected

    def test_nested_alias(
        self, comprehensive_3d_dataset: "ComprehensiveDatasetInfo"
    ) -> None:
        """Deeply nested alias should still work."""
        url = comprehensive_3d_dataset.path
        pred = (pl.col("a") == 15).alias("x").alias("y").alias("z")
        chunks = ZarrBackend.from_url(url).selected_chunks_debug( pred)
        idxs = _chunk_indices(chunks)

        expected = {(1, b, c) for b in range(5) for c in range(3)}
        assert len(idxs) == 15
        assert idxs == expected

    def test_cast_preserves_pushdown(
        self, comprehensive_3d_dataset: "ComprehensiveDatasetInfo"
    ) -> None:
        """(a == 15).cast(Boolean) should narrow."""
        url = comprehensive_3d_dataset.path
        pred = (pl.col("a") == 15).cast(pl.Boolean)
        chunks = ZarrBackend.from_url(url).selected_chunks_debug( pred)
        idxs = _chunk_indices(chunks)

        expected = {(1, b, c) for b in range(5) for c in range(3)}
        assert len(idxs) == 15
        assert idxs == expected

    def test_alias_cast_chain(
        self, comprehensive_3d_dataset: "ComprehensiveDatasetInfo"
    ) -> None:
        """(a == 15).alias("x").cast(Boolean).alias("y") should narrow."""
        url = comprehensive_3d_dataset.path
        pred = (pl.col("a") == 15).alias("x").cast(pl.Boolean).alias("y")
        chunks = ZarrBackend.from_url(url).selected_chunks_debug( pred)
        idxs = _chunk_indices(chunks)

        expected = {(1, b, c) for b in range(5) for c in range(3)}
        assert len(idxs) == 15
        assert idxs == expected

    def test_over_window_preserves_pushdown(
        self, comprehensive_3d_dataset: "ComprehensiveDatasetInfo"
    ) -> None:
        """(a == 15).over("b") should still narrow by a."""
        url = comprehensive_3d_dataset.path
        pred = (pl.col("a") == 15).over("b")
        chunks = ZarrBackend.from_url(url).selected_chunks_debug( pred)
        idxs = _chunk_indices(chunks)

        expected = {(1, b, c) for b in range(5) for c in range(3)}
        assert len(idxs) == 15
        assert idxs == expected

    def test_complex_wrapper_chain(
        self, comprehensive_3d_dataset: "ComprehensiveDatasetInfo"
    ) -> None:
        """Complex chain: ((a < 20) & (b < 20)).alias("x").cast(Boolean)."""
        url = comprehensive_3d_dataset.path
        pred = ((pl.col("a") < 20) & (pl.col("b") < 20)).alias("x").cast(pl.Boolean)
        chunks = ZarrBackend.from_url(url).selected_chunks_debug( pred)
        idxs = _chunk_indices(chunks)

        expected = {(a, b, c) for a in range(2) for b in range(2) for c in range(3)}
        assert len(idxs) == 12
        assert idxs == expected


class TestTernary:
    """Tests for when/then/otherwise (ternary) expressions."""

    def test_ternary_true_false_equals_predicate(
        self, comprehensive_3d_dataset: "ComprehensiveDatasetInfo"
    ) -> None:
        """when(a==15).then(True).otherwise(False) == (a==15)."""
        url = comprehensive_3d_dataset.path
        pred = pl.when(pl.col("a") == 15).then(pl.lit(True)).otherwise(pl.lit(False))
        chunks = ZarrBackend.from_url(url).selected_chunks_debug( pred)
        idxs = _chunk_indices(chunks)

        expected = {(1, b, c) for b in range(5) for c in range(3)}
        assert len(idxs) == 15
        assert idxs == expected

    def test_ternary_false_true_is_conservative(
        self, comprehensive_3d_dataset: "ComprehensiveDatasetInfo"
    ) -> None:
        """when(a==15).then(False).otherwise(True) is complement, conservative."""
        url = comprehensive_3d_dataset.path
        pred = pl.when(pl.col("a") == 15).then(pl.lit(False)).otherwise(pl.lit(True))
        chunks = ZarrBackend.from_url(url).selected_chunks_debug( pred)
        idxs = _chunk_indices(chunks)

        # Complement can't be represented, should be conservative
        assert len(idxs) == comprehensive_3d_dataset.total_chunks

    def test_ternary_both_true(
        self, comprehensive_3d_dataset: "ComprehensiveDatasetInfo"
    ) -> None:
        """when(a==15).then(True).otherwise(True) should return all."""
        url = comprehensive_3d_dataset.path
        pred = pl.when(pl.col("a") == 15).then(pl.lit(True)).otherwise(pl.lit(True))
        chunks = ZarrBackend.from_url(url).selected_chunks_debug( pred)
        idxs = _chunk_indices(chunks)

        assert len(idxs) == comprehensive_3d_dataset.total_chunks

    def test_ternary_both_false(
        self, comprehensive_3d_dataset: "ComprehensiveDatasetInfo"
    ) -> None:
        """when(a==15).then(False).otherwise(False) should return empty."""
        url = comprehensive_3d_dataset.path
        pred = pl.when(pl.col("a") == 15).then(pl.lit(False)).otherwise(pl.lit(False))
        chunks = ZarrBackend.from_url(url).selected_chunks_debug( pred)
        idxs = _chunk_indices(chunks)

        assert len(idxs) == 0

    def test_ternary_with_narrowing_predicate(
        self, comprehensive_3d_dataset: "ComprehensiveDatasetInfo"
    ) -> None:
        """when((a<20) & (b<20)).then(True).otherwise(False) narrows."""
        url = comprehensive_3d_dataset.path
        pred = (
            pl.when((pl.col("a") < 20) & (pl.col("b") < 20))
            .then(pl.lit(True))
            .otherwise(pl.lit(False))
        )
        chunks = ZarrBackend.from_url(url).selected_chunks_debug( pred)
        idxs = _chunk_indices(chunks)

        expected = {(a, b, c) for a in range(2) for b in range(2) for c in range(3)}
        assert len(idxs) == 12
        assert idxs == expected


class TestLiterals:
    """Tests for literal true, false, null."""

    def test_literal_true_all_chunks(
        self, comprehensive_3d_dataset: "ComprehensiveDatasetInfo"
    ) -> None:
        """pl.lit(True) should return all 105 chunks."""
        url = comprehensive_3d_dataset.path
        pred = pl.lit(True)
        chunks = ZarrBackend.from_url(url).selected_chunks_debug( pred)
        idxs = _chunk_indices(chunks)

        assert len(idxs) == comprehensive_3d_dataset.total_chunks

    def test_literal_false_empty(
        self, comprehensive_3d_dataset: "ComprehensiveDatasetInfo"
    ) -> None:
        """pl.lit(False) should return 0 chunks."""
        url = comprehensive_3d_dataset.path
        pred = pl.lit(False)
        chunks = ZarrBackend.from_url(url).selected_chunks_debug( pred)
        idxs = _chunk_indices(chunks)

        assert len(idxs) == 0

    def test_literal_null_empty(
        self, comprehensive_3d_dataset: "ComprehensiveDatasetInfo"
    ) -> None:
        """pl.lit(None) (null predicate) should return 0 chunks."""
        url = comprehensive_3d_dataset.path
        pred = pl.lit(None)
        chunks = ZarrBackend.from_url(url).selected_chunks_debug( pred)
        idxs = _chunk_indices(chunks)

        assert len(idxs) == 0

    def test_literal_true_and_narrowing(
        self, comprehensive_3d_dataset: "ComprehensiveDatasetInfo"
    ) -> None:
        """pl.lit(True) & (a < 20) should narrow."""
        url = comprehensive_3d_dataset.path
        pred = pl.lit(True) & (pl.col("a") < 20)
        chunks = ZarrBackend.from_url(url).selected_chunks_debug( pred)
        idxs = _chunk_indices(chunks)

        expected = {(a, b, c) for a in range(2) for b in range(5) for c in range(3)}
        assert len(idxs) == 30
        assert idxs == expected

    def test_literal_false_and_shortcircuit(
        self, comprehensive_3d_dataset: "ComprehensiveDatasetInfo"
    ) -> None:
        """pl.lit(False) & (a < 20) should return empty."""
        url = comprehensive_3d_dataset.path
        pred = pl.lit(False) & (pl.col("a") < 20)
        chunks = ZarrBackend.from_url(url).selected_chunks_debug( pred)
        idxs = _chunk_indices(chunks)

        assert len(idxs) == 0


class TestMultiDim:
    """Tests for multi-dimensional constraint combinations."""

    def test_three_dim_precise_selection(
        self, comprehensive_3d_dataset: "ComprehensiveDatasetInfo"
    ) -> None:
        """
        (a >= 10 & a < 30) & (b >= 20 & b < 40) & (c >= 10 & c < 20)
        Should give exactly 2*2*1 = 4 chunks.
        """
        url = comprehensive_3d_dataset.path
        pred = (
            (pl.col("a") >= 10)
            & (pl.col("a") < 30)
            & (pl.col("b") >= 20)
            & (pl.col("b") < 40)
            & (pl.col("c") >= 10)
            & (pl.col("c") < 20)
        )
        chunks = ZarrBackend.from_url(url).selected_chunks_debug( pred)
        idxs = _chunk_indices(chunks)

        # a: 10-29 -> chunks 1,2
        # b: 20-39 -> chunks 2,3
        # c: 10-19 -> chunk 1
        expected = {(1, 2, 1), (1, 3, 1), (2, 2, 1), (2, 3, 1)}
        assert len(idxs) == 4
        assert idxs == expected

    def test_7_chunk_selection(
        self, comprehensive_3d_dataset: "ComprehensiveDatasetInfo"
    ) -> None:
        """
        Design a predicate that selects exactly 7 chunks.
        (a == 15) & (c < 10) & (b.is_in([5, 25])) = 1*2*1 = 2 chunks? No...
        Let's try: (a < 10) & (b < 10) & (c.is_in([5, 15, 25])) = 1*1*3 = 3
        Or: exact points from 3 dims to get 7.
        """
        # This is tricky with prime factorization 7*5*3.
        # 7 = 7*1*1 -> select all a chunks, 1 b chunk, 1 c chunk
        url = comprehensive_3d_dataset.path
        pred = (pl.col("b") == 25) & (pl.col("c") == 15)
        chunks = ZarrBackend.from_url(url).selected_chunks_debug( pred)
        idxs = _chunk_indices(chunks)

        # b=2, c=1 -> 7*1*1 = 7 chunks
        expected = {(a, 2, 1) for a in range(7)}
        assert len(idxs) == 7
        assert idxs == expected

    def test_11_chunk_selection(
        self, comprehensive_3d_dataset: "ComprehensiveDatasetInfo"
    ) -> None:
        """
        Design a predicate that selects exactly 11 chunks (prime).
        This requires a union of non-overlapping selections.
        
        Strategy: 7 + 3 + 1 = 11
        - (b==25 & c==15) gives 7 chunks: (a, 2, 1) for a in 0..6
        - (a==55 & b==5) gives 3 chunks: (5, 0, c) for c in 0..2
        - (a==65 & b==45 & c==25) gives 1 chunk: (6, 4, 2)
        Check overlaps: (5,0,*) vs (5,2,1) - different b; (6,4,2) vs (6,2,1) - different b,c
        """
        url = comprehensive_3d_dataset.path
        pred = (
            ((pl.col("b") == 25) & (pl.col("c") == 15))
            | ((pl.col("a") == 55) & (pl.col("b") == 5))
            | ((pl.col("a") == 65) & (pl.col("b") == 45) & (pl.col("c") == 25))
        )
        chunks = ZarrBackend.from_url(url).selected_chunks_debug( pred)
        idxs = _chunk_indices(chunks)

        first_set = {(a, 2, 1) for a in range(7)}  # 7 chunks
        second_set = {(5, 0, c) for c in range(3)}  # 3 chunks
        third_set = {(6, 4, 2)}  # 1 chunk
        expected = first_set | second_set | third_set
        assert len(idxs) == 11
        assert idxs == expected

    def test_13_chunk_selection(
        self, comprehensive_3d_dataset: "ComprehensiveDatasetInfo"
    ) -> None:
        """
        Design a predicate that selects exactly 13 chunks (prime).
        """
        url = comprehensive_3d_dataset.path
        # 7 + 5 + 1 = 13: (b==25 & c==15) gives 7, (a==0 & c==25) gives 5, add 1 more
        # (a==0 & c==25) -> (0, b, 2) for b in 0..4 = 5 chunks
        # Overlap with first set: (0, 2, 1)? No, c differs
        # Need to find 1 more non-overlapping chunk
        pred = (
            ((pl.col("b") == 25) & (pl.col("c") == 15))
            | ((pl.col("a") == 5) & (pl.col("c") == 25))
            | ((pl.col("a") == 65) & (pl.col("b") == 45) & (pl.col("c") == 25))
        )
        chunks = ZarrBackend.from_url(url).selected_chunks_debug( pred)
        idxs = _chunk_indices(chunks)

        # First: 7 chunks (a=0..6, b=2, c=1)
        # Second: 5 chunks (a=0, b=0..4, c=2)
        # Third: 1 chunk (a=6, b=4, c=2)
        # Overlaps: (0, 2, 1) from first, (0, *, 2) from second -> no overlap
        first_set = {(a, 2, 1) for a in range(7)}
        second_set = {(0, b, 2) for b in range(5)}
        third_set = {(6, 4, 2)}
        # Check overlap
        assert first_set & second_set == set()
        # (6, 4, 2) is in second_set? No, second set has a=0
        # But wait, (0, 2, 2) is in second_set but not first_set
        expected = first_set | second_set | third_set
        assert len(idxs) == 13
        assert idxs == expected

    def test_asymmetric_range_selection(
        self, comprehensive_3d_dataset: "ComprehensiveDatasetInfo"
    ) -> None:
        """
        a in [0,1,2], b in [0,1], c in [0] = 3*2*1 = 6 chunks.
        """
        url = comprehensive_3d_dataset.path
        pred = (pl.col("a") < 30) & (pl.col("b") < 20) & (pl.col("c") < 10)
        chunks = ZarrBackend.from_url(url).selected_chunks_debug( pred)
        idxs = _chunk_indices(chunks)

        expected = {(a, b, 0) for a in range(3) for b in range(2)}
        assert len(idxs) == 6
        assert idxs == expected


class TestEdgeCases:
    """Tests for boundary values and edge cases."""

    def test_empty_is_in_list(
        self, comprehensive_3d_dataset: "ComprehensiveDatasetInfo"
    ) -> None:
        """a.is_in([]) should return empty."""
        url = comprehensive_3d_dataset.path
        pred = pl.col("a").is_in([])
        chunks = ZarrBackend.from_url(url).selected_chunks_debug( pred)
        idxs = _chunk_indices(chunks)

        assert len(idxs) == 0

    def test_out_of_range_value(
        self, comprehensive_3d_dataset: "ComprehensiveDatasetInfo"
    ) -> None:
        """a == 100 (beyond dimension) should return empty."""
        url = comprehensive_3d_dataset.path
        pred = pl.col("a") == 100
        chunks = ZarrBackend.from_url(url).selected_chunks_debug( pred)
        idxs = _chunk_indices(chunks)

        assert len(idxs) == 0

    def test_negative_value(
        self, comprehensive_3d_dataset: "ComprehensiveDatasetInfo"
    ) -> None:
        """a == -5 (negative) should return empty."""
        url = comprehensive_3d_dataset.path
        pred = pl.col("a") == -5
        chunks = ZarrBackend.from_url(url).selected_chunks_debug( pred)
        idxs = _chunk_indices(chunks)

        assert len(idxs) == 0

    def test_all_chunks_boundary(
        self, comprehensive_3d_dataset: "ComprehensiveDatasetInfo"
    ) -> None:
        """a >= 0 should return all 105 chunks."""
        url = comprehensive_3d_dataset.path
        pred = pl.col("a") >= 0
        chunks = ZarrBackend.from_url(url).selected_chunks_debug( pred)
        idxs = _chunk_indices(chunks)

        assert len(idxs) == comprehensive_3d_dataset.total_chunks

    def test_impossible_range(
        self, comprehensive_3d_dataset: "ComprehensiveDatasetInfo"
    ) -> None:
        """(a > 50) & (a < 40) is impossible, should return empty."""
        url = comprehensive_3d_dataset.path
        pred = (pl.col("a") > 50) & (pl.col("a") < 40)
        chunks = ZarrBackend.from_url(url).selected_chunks_debug( pred)
        idxs = _chunk_indices(chunks)

        assert len(idxs) == 0

    def test_single_point_at_chunk_boundary_start(
        self, comprehensive_3d_dataset: "ComprehensiveDatasetInfo"
    ) -> None:
        """a == 30 (exactly at chunk 3 start) should select chunk 3."""
        url = comprehensive_3d_dataset.path
        pred = pl.col("a") == 30
        chunks = ZarrBackend.from_url(url).selected_chunks_debug( pred)
        idxs = _chunk_indices(chunks)

        expected = {(3, b, c) for b in range(5) for c in range(3)}
        assert len(idxs) == 15
        assert idxs == expected

    def test_single_point_at_chunk_boundary_end(
        self, comprehensive_3d_dataset: "ComprehensiveDatasetInfo"
    ) -> None:
        """a == 29 (exactly at chunk 2 end) should select chunk 2."""
        url = comprehensive_3d_dataset.path
        pred = pl.col("a") == 29
        chunks = ZarrBackend.from_url(url).selected_chunks_debug( pred)
        idxs = _chunk_indices(chunks)

        expected = {(2, b, c) for b in range(5) for c in range(3)}
        assert len(idxs) == 15
        assert idxs == expected

    def test_range_exactly_one_chunk(
        self, comprehensive_3d_dataset: "ComprehensiveDatasetInfo"
    ) -> None:
        """a.is_between(20, 29) exactly covers chunk 2."""
        url = comprehensive_3d_dataset.path
        pred = pl.col("a").is_between(20, 29)
        chunks = ZarrBackend.from_url(url).selected_chunks_debug( pred)
        idxs = _chunk_indices(chunks)

        expected = {(2, b, c) for b in range(5) for c in range(3)}
        assert len(idxs) == 15
        assert idxs == expected

    def test_range_off_by_one_spans_two_chunks(
        self, comprehensive_3d_dataset: "ComprehensiveDatasetInfo"
    ) -> None:
        """a.is_between(19, 30) spans chunks 1,2,3."""
        url = comprehensive_3d_dataset.path
        pred = pl.col("a").is_between(19, 30)
        chunks = ZarrBackend.from_url(url).selected_chunks_debug( pred)
        idxs = _chunk_indices(chunks)

        # 19 is in chunk 1, 30 is in chunk 3
        expected = {(a, b, c) for a in range(1, 4) for b in range(5) for c in range(3)}
        assert len(idxs) == 45
        assert idxs == expected

    def test_float_comparison_on_int_coords(
        self, comprehensive_3d_dataset: "ComprehensiveDatasetInfo"
    ) -> None:
        """a == 15.0 should work the same as a == 15."""
        url = comprehensive_3d_dataset.path
        pred = pl.col("a") == 15.0
        chunks = ZarrBackend.from_url(url).selected_chunks_debug( pred)
        idxs = _chunk_indices(chunks)

        expected = {(1, b, c) for b in range(5) for c in range(3)}
        assert len(idxs) == 15
        assert idxs == expected

    def test_multiple_variables_same_selection(
        self, comprehensive_3d_dataset: "ComprehensiveDatasetInfo"
    ) -> None:
        """Selection should work for different variables."""
        url = comprehensive_3d_dataset.path
        pred = pl.col("a") == 15

        chunks1 = ZarrBackend.from_url(url).selected_chunks_debug( pred)
        chunks2 = ZarrBackend.from_url(url).selected_chunks_debug( pred)

        assert _chunk_indices(chunks1) == _chunk_indices(chunks2)
        assert len(_chunk_indices(chunks1)) == 15

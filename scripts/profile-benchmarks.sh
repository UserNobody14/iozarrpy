#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Profile benchmark tests with perf and generate LLM-readable reports.

Usage:
  scripts/profile-benchmarks.sh [-- <benchmark command...>]

Defaults:
  command: uv run pytest -m benchmark
  output:  profile-results/<timestamp>/

Environment:
  OUT_DIR       Output directory. Default: profile-results/<timestamp>
  PERF_FREQ     Sampling frequency. Default: 997
  DSO_FILTER    DSO substring/list for project-focused reports. Default: _core.abi3.so
  PERCENT_LIMIT Minimum percent shown by perf report. Default: 0.5
  RUSTFLAGS     Rust flags used while running benchmarks. Default: -C debuginfo=1
  RAINBEAR_SKIP_XARRAY_BENCHMARKS
                Set to 1/true/yes/on to skip xarray comparison benchmarks.
  RAINBEAR_REUSE_TEST_DATASETS
                Set to 1/true/yes/on after pre-generating datasets to keep
                xarray/zarr dataset creation out of the perf recording.
  RAINBEAR_PREGENERATE_TEST_DATASETS
                Set to 1/true/yes/on to generate benchmark datasets before
                perf recording and then run with RAINBEAR_REUSE_TEST_DATASETS=1.
  RAINBEAR_GENERATE_TEST_DATASET_GROUPS
                Dataset groups for pre-generation. Default: benchmark.

Examples:
  scripts/profile-benchmarks.sh
  PERCENT_LIMIT=0.1 scripts/profile-benchmarks.sh -- uv run pytest tests/test_benchmark_novel_queries.py -m benchmark
  DSO_FILTER=_core.abi3.so scripts/profile-benchmarks.sh -- uv run pytest -m benchmark -k concurrent_10q
  RAINBEAR_SKIP_XARRAY_BENCHMARKS=1 scripts/profile-benchmarks.sh -- uv run pytest -m benchmark
  uv run pytest --rainbear-generate-test-datasets-only
  RAINBEAR_PREGENERATE_TEST_DATASETS=1 RAINBEAR_SKIP_XARRAY_BENCHMARKS=1 scripts/profile-benchmarks.sh
  RAINBEAR_REUSE_TEST_DATASETS=1 RAINBEAR_SKIP_XARRAY_BENCHMARKS=1 scripts/profile-benchmarks.sh -- uv run pytest -m benchmark
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if ! command -v perf >/dev/null 2>&1; then
  echo "error: perf is not installed or not on PATH" >&2
  exit 1
fi

timestamp="$(date +%Y%m%d-%H%M%S)"
out_dir="${OUT_DIR:-profile-results/${timestamp}}"
perf_freq="${PERF_FREQ:-997}"
dso_filter="${DSO_FILTER:-_core.abi3.so}"
percent_limit="${PERCENT_LIMIT:-0.5}"
export RUSTFLAGS="${RUSTFLAGS:--C debuginfo=1}"

if [[ "${1:-}" == "--" ]]; then
  shift
fi

if [[ "$#" -gt 0 ]]; then
  bench_cmd=("$@")
else
  bench_cmd=(uv run pytest -m benchmark)
fi

mkdir -p "${out_dir}"

perf_data="${out_dir}/perf.data"
self_report="${out_dir}/perf-report-self.txt"
inclusive_report="${out_dir}/perf-report-inclusive.txt"
core_self_report="${out_dir}/perf-report-core-self.txt"
core_inclusive_report="${out_dir}/perf-report-core-inclusive.txt"
summary="${out_dir}/profile-summary.md"

echo "Output directory: ${out_dir}"
echo "Benchmark command: ${bench_cmd[*]}"
echo "RUSTFLAGS=${RUSTFLAGS}"

case "${RAINBEAR_PREGENERATE_TEST_DATASETS:-}" in
  1|true|TRUE|yes|YES|on|ON)
    echo "Pre-generating test datasets outside perf recording..."
    uv run pytest --rainbear-generate-test-datasets-only
    export RAINBEAR_REUSE_TEST_DATASETS=1
    ;;
esac

perf record \
  -F "${perf_freq}" \
  --call-graph dwarf \
  -o "${perf_data}" \
  -- "${bench_cmd[@]}"

report_common=(
  -i "${perf_data}"
  --stdio
  --no-inline
  --sort comm,dso,symbol
  --percent-limit "${percent_limit}"
)

perf report "${report_common[@]}" --no-children > "${self_report}"
perf report "${report_common[@]}" --children > "${inclusive_report}"

if perf report "${report_common[@]}" --no-children --dsos "${dso_filter}" > "${core_self_report}"; then
  :
else
  echo "warning: focused self report failed for DSO_FILTER=${dso_filter}" >&2
  : > "${core_self_report}"
fi

if perf report "${report_common[@]}" --children --dsos "${dso_filter}" > "${core_inclusive_report}"; then
  :
else
  echo "warning: focused inclusive report failed for DSO_FILTER=${dso_filter}" >&2
  : > "${core_inclusive_report}"
fi

python - "${summary}" "${self_report}" "${inclusive_report}" "${core_self_report}" "${core_inclusive_report}" "${perf_data}" "${dso_filter}" "${bench_cmd[@]}" <<'PY'
from __future__ import annotations

import re
import sys
from pathlib import Path

summary = Path(sys.argv[1])
self_report = Path(sys.argv[2])
inclusive_report = Path(sys.argv[3])
core_self_report = Path(sys.argv[4])
core_inclusive_report = Path(sys.argv[5])
perf_data = Path(sys.argv[6])
dso_filter = sys.argv[7]
bench_cmd = sys.argv[8:]

entry_self_with_dso = re.compile(
    r"^\s*(\d+\.\d+)%\s+(\S+)\s+(\S+)\s+\[.\]\s+(.+?)\s{2,}"
)
entry_self_without_dso = re.compile(
    r"^\s*(\d+\.\d+)%\s+(\S+)\s+\[.\]\s+(.+?)\s{2,}"
)
entry_inclusive_with_dso = re.compile(
    r"^\s*(\d+\.\d+)%\s+(\d+\.\d+)%\s+(\S+)\s+(\S+)\s+\[.\]\s+(.+?)\s{2,}"
)
entry_inclusive_without_dso = re.compile(
    r"^\s*(\d+\.\d+)%\s+(\d+\.\d+)%\s+(\S+)\s+\[.\]\s+(.+?)\s{2,}"
)

interesting = re.compile(
    r"(rainbear|iozarr|_core|zarr|Zarr|metadata|schema|compile|chunk|grid|polars|tokio|rayon|FilesystemStore)",
    re.IGNORECASE,
)

category_patterns = [
    (
        "rainbear scan/chunk work",
        re.compile(
            r"(rainbear|_core::(scan|chunk_plan|reader|backend|meta|store|shared))",
            re.IGNORECASE,
        ),
    ),
    (
        "zarr storage/decode",
        re.compile(r"(zarrs|icechunk|FilesystemStore|get_partial|read_at)", re.IGNORECASE),
    ),
    (
        "compression codecs",
        re.compile(r"(zstd|blosc|numcodecs|compress|decompress|HUF_|ZSTD_)", re.IGNORECASE),
    ),
    (
        "polars execution/planning",
        re.compile(r"(polars|DataFrame|Series|schema|AExpr|LazyFrame)", re.IGNORECASE),
    ),
    (
        "filesystem/kernel IO",
        re.compile(r"(\[kernel\.kallsyms\]|_copy_to_iter|filemap_read|pread|read_at|ext4)", re.IGNORECASE),
    ),
    (
        "memory movement/allocation",
        re.compile(r"(memcpy|memmove|memset|malloc|free|realloc|alloc::|__mem)", re.IGNORECASE),
    ),
    (
        "python interpreter/harness",
        re.compile(r"(libpython|_Py|PyObject|pytest|benchmark|asyncio|task_step|gen_send)", re.IGNORECASE),
    ),
    (
        "runtime scheduling/wait",
        re.compile(
            r"(rayon_core::registry|crossbeam|tokio::runtime|wait_until|find_work|steal|start_thread|clone3|pythread_wrapper)",
            re.IGNORECASE,
        ),
    ),
    (
        "numeric libraries",
        re.compile(r"(openblas|blas|scipy|numpy)", re.IGNORECASE),
    ),
]

noise_categories = {
    "python interpreter/harness",
    "runtime scheduling/wait",
}


def clean_symbol(symbol: str) -> str:
    replacements = {
        "_$LT$": "<",
        "$LT$": "<",
        "$GT$": ">",
        "$u20$": " ",
        "$C$": ",",
        "$u7b$": "{",
        "$u7d$": "}",
    }
    for old, new in replacements.items():
        symbol = symbol.replace(old, new)
    return symbol.replace("..", "::").strip()


def classify_text(text: str) -> str:
    for category, pattern in category_patterns:
        if pattern.search(text):
            return category
    return "other"


def row_text(row: tuple) -> str:
    return " ".join(str(part) for part in row)


def summarize_categories(rows: list[tuple], percent_index: int) -> list[tuple[str, float, int]]:
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for row in rows:
        category = classify_text(row_text(row))
        totals[category] = totals.get(category, 0.0) + float(row[percent_index])
        counts[category] = counts.get(category, 0) + 1
    return sorted(
        ((category, pct, counts[category]) for category, pct in totals.items()),
        key=lambda item: item[1],
        reverse=True,
    )


def write_category_table(
    f, rows: list[tuple[str, float, int]], percent_name: str, limit: int = 12
) -> None:
    f.write(f"| category | summed {percent_name} % | rows |\n")
    f.write("|---|---:|---:|\n")
    for category, pct, count in rows[:limit]:
        f.write(f"| {category} | {pct:.2f} | {count} |\n")


def filter_active_rows(rows: list[tuple]) -> list[tuple]:
    return [
        row
        for row in rows
        if classify_text(row_text(row)) not in noise_categories
    ]


def parse_self(path: Path, limit: int = 200) -> list[tuple[float, str, str, str]]:
    rows: list[tuple[float, str, str, str]] = []
    if not path.exists():
        return rows
    for line in path.read_text(errors="replace").splitlines():
        match = entry_self_with_dso.match(line)
        if match:
            pct, command, dso, symbol = match.groups()
        else:
            match = entry_self_without_dso.match(line)
            if not match:
                continue
            pct, command, symbol = match.groups()
            dso = dso_filter
        if not symbol or symbol.startswith("."):
            continue
        rows.append((float(pct), command, dso, clean_symbol(symbol)))
        if len(rows) >= limit:
            break
    return rows


def parse_inclusive(
    path: Path, limit: int = 200
) -> list[tuple[float, float, str, str, str]]:
    rows: list[tuple[float, float, str, str, str]] = []
    if not path.exists():
        return rows
    for line in path.read_text(errors="replace").splitlines():
        match = entry_inclusive_with_dso.match(line)
        if match:
            child, self_pct, command, dso, symbol = match.groups()
        else:
            match = entry_inclusive_without_dso.match(line)
            if not match:
                continue
            child, self_pct, command, symbol = match.groups()
            dso = dso_filter
        if not symbol or symbol.startswith("."):
            continue
        rows.append(
            (float(child), float(self_pct), command, dso, clean_symbol(symbol))
        )
        if len(rows) >= limit:
            break
    return rows


def extract_call_graph_snippets(path: Path, max_blocks: int = 10, max_lines: int = 28) -> list[list[str]]:
    if not path.exists():
        return []
    lines = path.read_text(errors="replace").splitlines()
    blocks: list[list[str]] = []
    i = 0
    while i < len(lines) and len(blocks) < max_blocks:
        line = lines[i]
        if not re.match(r"^\s*\d+\.\d+%\s+", line):
            i += 1
            continue
        block = [line.rstrip()]
        j = i + 1
        while j < len(lines) and len(block) < max_lines:
            if re.match(r"^\s*\d+\.\d+%\s+", lines[j]):
                break
            stripped = lines[j].rstrip()
            if stripped.strip() and not stripped.startswith("#"):
                block.append(stripped[:220])
            j += 1
        if any(interesting.search(item) for item in block) or len(blocks) < 4:
            blocks.append(block)
        i = j
    return blocks


def extract_actionable_call_graph_snippets(
    path: Path, max_blocks: int = 12, max_lines: int = 36
) -> list[list[str]]:
    if not path.exists():
        return []
    lines = path.read_text(errors="replace").splitlines()
    blocks: list[list[str]] = []
    i = 0
    while i < len(lines) and len(blocks) < max_blocks:
        line = lines[i]
        if not re.match(r"^\s*\d+\.\d+%\s+", line):
            i += 1
            continue
        block = [line.rstrip()]
        j = i + 1
        while j < len(lines) and len(block) < max_lines:
            if re.match(r"^\s*\d+\.\d+%\s+", lines[j]):
                break
            stripped = lines[j].rstrip()
            if stripped.strip() and not stripped.startswith("#"):
                block.append(stripped[:220])
            j += 1

        joined = "\n".join(block)
        categories = {classify_text(item) for item in block}
        scheduler_only = categories <= {"runtime scheduling/wait", "other"}
        has_actionable_frame = any(
            category in categories
            for category in (
                "rainbear scan/chunk work",
                "zarr storage/decode",
                "compression codecs",
                "filesystem/kernel IO",
                "polars execution/planning",
                "memory movement/allocation",
            )
        )
        if has_actionable_frame and not scheduler_only and not re.search(
            r"blas_thread_server", joined, re.IGNORECASE
        ):
            blocks.append(block)
        i = j
    return blocks


def write_self_table(f, rows: list[tuple[float, str, str, str]], limit: int = 30) -> None:
    f.write("| self % | command | object | symbol |\n")
    f.write("|---:|---|---|---|\n")
    for pct, command, dso, symbol in rows[:limit]:
        f.write(f"| {pct:.2f} | `{command}` | `{dso}` | `{symbol}` |\n")


def write_inclusive_table(
    f, rows: list[tuple[float, float, str, str, str]], limit: int = 30
) -> None:
    f.write("| children % | self % | command | object | symbol |\n")
    f.write("|---:|---:|---|---|---|\n")
    for child, self_pct, command, dso, symbol in rows[:limit]:
        f.write(
            f"| {child:.2f} | {self_pct:.2f} | `{command}` | `{dso}` | `{symbol}` |\n"
        )


self_rows = parse_self(self_report)
inclusive_rows = parse_inclusive(inclusive_report)
core_self_rows = parse_self(core_self_report)
core_inclusive_rows = parse_inclusive(core_inclusive_report)
interesting_self_rows = [
    row
    for row in self_rows
    if interesting.search(row[2]) or interesting.search(row[3])
]
interesting_inclusive_rows = [
    row
    for row in inclusive_rows
    if interesting.search(row[3]) or interesting.search(row[4])
]
active_self_rows = filter_active_rows(self_rows)
active_inclusive_rows = filter_active_rows(inclusive_rows)

summary.parent.mkdir(parents=True, exist_ok=True)
with summary.open("w") as f:
    f.write("# LLM Profile Summary\n\n")
    f.write(f"Benchmark command: `{' '.join(bench_cmd)}`\n\n")
    f.write(f"Perf data: `{perf_data}`\n\n")
    f.write("Reports were generated with `perf report --stdio --no-inline` to avoid addr2line inline expansion failures.\n\n")
    f.write("## Interpretation Hints\n\n")
    f.write("- `runtime scheduling/wait` rows are usually Rayon/Tokio worker overhead or idle stealing, not leaf work to optimize directly.\n")
    f.write("- `python interpreter/harness` rows are pytest, asyncio, Polars-Python, and benchmark harness cost around the extension.\n")
    f.write("- `compression codecs`, `filesystem/kernel IO`, and `zarr storage/decode` often point to data format, chunking, cache, or benchmark setup choices rather than Rust scan logic.\n")
    f.write("- Percentages in the category tables are summed from the parsed top report rows, so use them as ranking hints, not exact whole-program accounting.\n\n")

    f.write("## Category Summary\n\n")
    f.write("### Self Cost Categories\n\n")
    write_category_table(f, summarize_categories(self_rows, 0), "self")
    f.write("\n### Inclusive / Children Cost Categories\n\n")
    write_category_table(f, summarize_categories(inclusive_rows, 0), "children")
    f.write("\n")

    f.write("## De-noised Active Rows\n\n")
    f.write("Excludes `python interpreter/harness` and `runtime scheduling/wait` categories.\n\n")
    f.write("### Active Self Cost\n\n")
    write_self_table(f, active_self_rows, limit=30)
    f.write("\n### Active Inclusive / Children Cost\n\n")
    write_inclusive_table(f, active_inclusive_rows, limit=30)
    f.write("\n")

    f.write("## Focused Project DSO Reports\n\n")
    f.write(f"DSO filter: `{dso_filter}`\n\n")
    if not core_self_rows and not core_inclusive_rows:
        f.write("No focused rows matched the DSO filter. Check the DSO name in the full report and rerun with `DSO_FILTER=<name>`.\n\n")
    else:
        f.write("### Project Self Cost\n\n")
        write_self_table(f, core_self_rows, limit=40)
        f.write("\n### Project Inclusive / Children Cost\n\n")
        write_inclusive_table(f, core_inclusive_rows, limit=40)
        f.write("\n")

    f.write("## Filtered Interesting Frames\n\n")
    f.write("Filtered from the full reports for project/runtime terms: `_core`, rainbear, zarr, metadata, schema, compile, chunk, grid, polars, tokio, rayon, filesystem.\n\n")
    f.write("### Filtered Self Cost\n\n")
    write_self_table(f, interesting_self_rows, limit=40)
    f.write("\n### Filtered Inclusive / Children Cost\n\n")
    write_inclusive_table(f, interesting_inclusive_rows, limit=40)
    f.write("\n")

    f.write("## Full Top Self Cost\n\n")
    write_self_table(f, self_rows, limit=30)
    f.write("\n## Full Top Inclusive / Children Cost\n\n")
    write_inclusive_table(f, inclusive_rows, limit=30)
    f.write("\n## Focused Call Graph Snippets\n\n")
    snippets = extract_actionable_call_graph_snippets(inclusive_report, max_blocks=8)
    if snippets:
        f.write("These snippets are selected from the full inclusive report for actionable frames, not just `_core` DSO frames.\n\n")
    if not snippets:
        snippets = extract_call_graph_snippets(core_inclusive_report, max_blocks=8)
    if not snippets:
        snippets = extract_call_graph_snippets(inclusive_report, max_blocks=8)
    for index, block in enumerate(snippets, 1):
        f.write(f"### Stack snippet {index}\n\n```text\n")
        for line in block:
            f.write(line + "\n")
        f.write("```\n\n")

print(summary)
PY

echo
echo "Generated:"
printf '  %s\n' \
  "${perf_data}" \
  "${self_report}" \
  "${inclusive_report}" \
  "${core_self_report}" \
  "${core_inclusive_report}" \
  "${summary}"

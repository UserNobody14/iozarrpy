set dotenv-load

default:
  @just --list


[group: 'tests']
[doc('Run the test suite')]
test:
    uv run pytest

[group: 'build']
[doc('Build the project')]
build:
    just clean
    # Debug build by default for stability/repro (avoids gcc/rustc ICEs seen at -O3).
    CARGO_BUILD_JOBS=1 CFLAGS=-O0 CXXFLAGS=-O0 uv run --group build maturin develop --uv

[group: 'build']
[doc('Build the project (release). This may stress compilers; use only if needed.')]
build-for-publish:
    just clean
    CARGO_BUILD_JOBS=1 CFLAGS=-O1 CXXFLAGS=-O1 uv run --group build maturin build --zig --release

[group: 'build']
[doc('Build the project (release). This may stress compilers; use only if needed.')]
build-release:
    just clean
    CARGO_BUILD_JOBS=1 CFLAGS=-O1 CXXFLAGS=-O1 uv run --group build maturin develop --uv --release

[group: 'build']
[doc('Build manylinux wheels')]
build-manylinux:
    just clean
    docker run --rm -v $(pwd):/io ghcr.io/pyo3/maturin build --release

[group: 'build']
[doc('Clean the project')]
clean:
    rm -rf target dist build
    cargo clean
    rm -rf **/*.so
    uv clean

[group: 'publish']
[doc('Publish the project')]
publish:
    export UV_PUBLISH_TOKEN=$PYPI_TOKEN && uv publish

[group: 'tests']
[doc('Run the smoke test')]
smoke-test WHEEL_OR_SOURCE:
    uv run --isolated --no-project --with dist/*.{{ if WHEEL_OR_SOURCE == "wheel" { "whl" } else { "tar.gz" } }} tests/smoke_test.py
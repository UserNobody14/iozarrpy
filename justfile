set dotenv-load

default:
  @just --list


[group: 'tests']
[doc('Run the test suite')]
[working-directory: 'rainbear-tests']
test:
    uv run pytest

[group: 'build']
[doc('Build the project')]
[working-directory: 'rainbear']
build:
    just clean
    # Debug build by default for stability/repro (avoids gcc/rustc ICEs seen at -O3).
    CARGO_BUILD_JOBS=1 CFLAGS=-O0 CXXFLAGS=-O0 uv run --group build maturin develop --uv

[group: 'build']
[doc('Build the project (release). This may stress compilers; use only if needed.')]
[working-directory: 'rainbear']
build-release:
    just clean
    CARGO_BUILD_JOBS=1 CFLAGS=-O1 CXXFLAGS=-O1 uv run --group build maturin develop --uv --release

[group: 'build']
[doc('Clean the project')]
clean:
    rm -rf target dist build
    cd rainbear && rm -rf target dist build
    cd rainbear && cargo clean
    cd rainbear && rm -rf **/*.so
    uv clean

[group: 'publish']
[doc('Publish the project')]
[working-directory: 'rainbear']
publish:
    uv publish

[group: 'tests']
[doc('Run the smoke test')]
smoke-test WHEEL_OR_SOURCE:
    uv run --isolated --no-project --with dist/*.{{ if WHEEL_OR_SOURCE == "wheel" { "whl" } else { "tar.gz" } }} rainbear-tests/tests/smoke_test.py
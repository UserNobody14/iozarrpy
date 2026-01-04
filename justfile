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
    cargo build --release && uv build

[group: 'build']
[doc('Clean the project')]
clean:
    rm -rf target dist build
    cd rainbear && rm -rf target dist build
    cd rainbear && cargo clean
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
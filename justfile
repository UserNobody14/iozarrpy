[group: 'tests']
[doc('Run the test suite')]
[working-directory: 'rainbear-tests']
test:
    uv run pytest

[group: 'build']
[doc('Build the project')]
[working-directory: 'rainbear']
build:
    cargo build --release && uv build


[group: 'build']
[doc('Build the project')]
build-clean:
    rm -rf target dist build
    cd rainbear && rm -rf target dist build && cargo build --release && uv build
# Simple smoke test to ensure the package is installed correctly



def test_smoke():
    print("Smoke test started")
    import rainbear._core
    assert rainbear._core.hello_from_bin() == "Hello from rainbear!"
    print("Smoke test passed")

if __name__ == "__main__":
    test_smoke()
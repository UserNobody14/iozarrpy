# Simple smoke test to ensure the package is installed correctly



def test_smoke():
    print("Smoke test started")
    import rainbear._core
    assert rainbear._core.print_extension_info() == "Rainbear extension module loaded successfully"
    print("Smoke test passed")

if __name__ == "__main__":
    test_smoke()
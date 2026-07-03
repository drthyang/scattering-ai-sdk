import scattering_ai


def test_version():
    assert isinstance(scattering_ai.__version__, str)
    assert scattering_ai.__version__


def test_public_api_exports():
    for name in scattering_ai.__all__:
        assert getattr(scattering_ai, name, None) is not None

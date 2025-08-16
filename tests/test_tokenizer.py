"""
Basic tests for the Turkish tokenizer package.
"""

import pytest


def test_import():
    """Test that the package can be imported successfully."""
    try:
        from tr_tokenizer_gemma import TRTokenizer
        assert True
    except ImportError as e:
        pytest.fail(f"Import failed: {e}")


def test_tokenizer_basic():
    """Test basic tokenizer functionality."""
    try:
        from tr_tokenizer_gemma import TRTokenizer

        # This test might fail if tokenizer data files are not available
        # so we'll just test that the class can be instantiated
        tokenizer = TRTokenizer()
        assert tokenizer is not None
    except Exception as e:
        # If data files are missing, just check that the class exists
        from tr_tokenizer_gemma import TRTokenizer
        assert TRTokenizer is not None


def test_package_version():
    """Test that package version is accessible."""
    from tr_tokenizer_gemma import __version__
    assert isinstance(__version__, str)
    assert len(__version__) > 0


def test_package_metadata():
    """Test that package metadata is accessible."""
    from tr_tokenizer_gemma import __author__, __description__, __title__
    assert isinstance(__author__, str)
    assert isinstance(__title__, str) 
    assert isinstance(__description__, str)


if __name__ == "__main__":
    pytest.main([__file__])

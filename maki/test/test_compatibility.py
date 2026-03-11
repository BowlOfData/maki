#!/usr/bin/env python3
"""
Test script to verify that the Maki class is now compatible with the new structure
and can be used by both old and new code.
"""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_maki_initialization():
    """Test that Maki can be initialized with parameters"""
    try:
        from maki.maki import Maki

        # Test initialization with all parameters
        maki = Maki("localhost", "11434", "llama3", 0.7)
        print("✓ Maki initialization successful")

        # Test that attributes are set correctly
        assert maki.url == "localhost"
        assert maki.port == "11434"
        assert maki.model == "llama3"
        assert maki.temperature == 0.7
        print("✓ Maki attribute assignment successful")

        return True
    except Exception as e:
        print(f"✗ Maki initialization failed: {e}")
        return False

def test_maki_methods():
    """Test that Maki has all the required methods"""
    try:
        from maki.maki import Maki

        maki = Maki("localhost", "11434", "llama3", 0.7)

        # Test that methods exist
        assert hasattr(maki, 'request')
        assert hasattr(maki, 'version')
        assert hasattr(maki, '_compose_data')
        assert hasattr(maki, 'request_with_images')
        assert hasattr(maki, '_get_model')
        assert hasattr(maki, '_get_temperature')
        print("✓ All Maki methods present")

        return True
    except Exception as e:
        print(f"✗ Maki methods test failed: {e}")
        return False

def test_backward_compatibility():
    """Test that existing code patterns work"""
    try:
        from maki.maki import Maki

        # Test pattern used in existing tests
        test_maki = Maki("localhost", "11434", "llama3", 0.7)

        # Test different model configurations
        maki_llama = Maki("localhost", "11434", "llama3", 0.7)
        maki_mistral = Maki("localhost", "11434", "mistral", 0.3)

        assert maki_llama.model == "llama3"
        assert maki_mistral.model == "mistral"
        assert maki_llama.temperature == 0.7
        assert maki_mistral.temperature == 0.3

        print("✓ Backward compatibility maintained")
        return True
    except Exception as e:
        print(f"✗ Backward compatibility test failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing Maki class compatibility...")

    tests = [
        test_maki_initialization,
        test_maki_methods,
        test_backward_compatibility
    ]

    results = []
    for test in tests:
        results.append(test())

    if all(results):
        print("\n✓ All compatibility tests passed!")
        sys.exit(0)
    else:
        print("\n✗ Some compatibility tests failed!")
        sys.exit(1)
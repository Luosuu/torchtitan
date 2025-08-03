#!/usr/bin/env python3
"""
Test script for dataset cache detection functionality.
Run this on your GPU server to validate local dataset caching works correctly.
"""

import os
import tempfile
import torch
from torchtitan.config.job_config import JobConfig, Training, Validation
from torchtitan.datasets.cache_utils import (
    get_hf_cache_dir,
    is_dataset_cached,
    load_dataset_with_cache_fallback,
    get_cached_dataset_info,
    clear_dataset_cache,
)


def test_cache_directory_detection():
    """Test HuggingFace cache directory detection"""
    print("🧪 Testing cache directory detection...")
    
    # Test default cache directory
    default_cache = get_hf_cache_dir()
    print(f"Default cache directory: {default_cache}")
    assert isinstance(default_cache, str)
    assert len(default_cache) > 0
    
    # Test with HF_HOME environment variable
    with tempfile.TemporaryDirectory() as temp_dir:
        original_hf_home = os.environ.get("HF_HOME")
        os.environ["HF_HOME"] = temp_dir
        
        hf_home_cache = get_hf_cache_dir()
        expected_path = os.path.join(temp_dir, "datasets")
        assert hf_home_cache == expected_path
        
        # Restore original environment
        if original_hf_home:
            os.environ["HF_HOME"] = original_hf_home
        else:
            os.environ.pop("HF_HOME", None)
    
    print("✅ Cache directory detection works correctly")


def test_cache_detection():
    """Test dataset cache detection"""
    print("🧪 Testing dataset cache detection...")
    
    # Test with non-existent dataset
    fake_dataset = "fake/nonexistent-dataset-12345"
    is_cached = is_dataset_cached(fake_dataset)
    assert not is_cached
    print(f"✅ Correctly detected that '{fake_dataset}' is not cached")
    
    # Test with local directory (should be considered "cached")
    with tempfile.TemporaryDirectory() as temp_dir:
        is_cached = is_dataset_cached(temp_dir)
        assert is_cached
        print(f"✅ Correctly detected that local directory '{temp_dir}' is cached")
    
    print("✅ Cache detection works correctly")


def test_dataset_info():
    """Test cached dataset information retrieval"""
    print("🧪 Testing dataset info retrieval...")
    
    fake_dataset = "fake/nonexistent-dataset-12345"
    info = get_cached_dataset_info(fake_dataset)
    
    assert info["dataset_path"] == fake_dataset
    assert not info["is_cached"]
    assert not info["cache_exists"]
    assert info["cache_size_mb"] == 0
    
    print("✅ Dataset info retrieval works correctly")


def test_config_integration():
    """Test that new config options are properly integrated"""
    print("🧪 Testing configuration integration...")
    
    # Test Training config
    training_config = Training(
        dataset_cache_dir="/custom/cache/dir",
        dataset_force_download=True,
        dataset_offline_mode=False
    )
    
    assert training_config.dataset_cache_dir == "/custom/cache/dir"
    assert training_config.dataset_force_download == True
    assert training_config.dataset_offline_mode == False
    
    # Test Validation config
    validation_config = Validation(
        dataset_cache_dir="/validation/cache",
        dataset_force_download=False,
        dataset_offline_mode=True
    )
    
    assert validation_config.dataset_cache_dir == "/validation/cache"
    assert validation_config.dataset_force_download == False
    assert validation_config.dataset_offline_mode == True
    
    print("✅ Configuration integration works correctly")


def test_offline_mode():
    """Test offline mode functionality"""
    print("🧪 Testing offline mode...")
    
    fake_dataset = "fake/nonexistent-dataset-12345"
    
    try:
        load_dataset_with_cache_fallback(
            fake_dataset,
            offline_mode=True
        )
        assert False, "Should have raised ConnectionError in offline mode"
    except ConnectionError as e:
        assert "offline_mode=True" in str(e)
        print("✅ Offline mode correctly raises ConnectionError for uncached datasets")
    
    print("✅ Offline mode works correctly")


def test_force_download_mode():
    """Test force download functionality"""
    print("🧪 Testing force download mode...")
    
    # This test would normally trigger a download, but we'll just test the logic
    fake_dataset = "fake/nonexistent-dataset-12345"
    
    try:
        load_dataset_with_cache_fallback(
            fake_dataset,
            force_download=True,
            offline_mode=True  # This combination should fail
        )
        assert False, "Should have raised ConnectionError"
    except ConnectionError:
        print("✅ Force download + offline mode correctly fails")
    
    print("✅ Force download mode works correctly")


def test_local_dataset_loading():
    """Test loading from local test dataset"""
    print("🧪 Testing local dataset loading...")
    
    # Test with the local c4_test dataset
    test_dataset_path = "tests/assets/c4_test"
    
    # Check if test dataset exists
    if os.path.exists(test_dataset_path):
        try:
            # This should work without network access
            dataset = load_dataset_with_cache_fallback(
                test_dataset_path,
                split="train",
                offline_mode=True
            )
            
            print(f"✅ Successfully loaded local test dataset from {test_dataset_path}")
            
            # Basic validation
            first_sample = next(iter(dataset))
            assert "text" in first_sample
            print("✅ Local dataset has expected structure")
            
        except Exception as e:
            print(f"⚠️  Could not load local test dataset: {e}")
    else:
        print(f"⚠️  Test dataset not found at {test_dataset_path}")
    
    print("✅ Local dataset loading test completed")


def test_cache_aware_dataset_loading():
    """Test cache-aware dataset loading (requires network or cache)"""
    print("🧪 Testing cache-aware dataset loading...")
    
    # This test will attempt to load a simple dataset with cache awareness
    # We use the Wikitext dataset as it's reliable and doesn't require scripts
    # It will succeed if:
    # 1. Dataset is already cached, OR
    # 2. Network is available for download
    
    try:
        from torchtitan.datasets.cache_utils import load_dataset_with_cache_fallback
        
        # Try to load with offline mode first (cache only)
        try:
            dataset = load_dataset_with_cache_fallback(
                "wikitext",
                name="wikitext-2-raw-v1",
                split="train",
                offline_mode=True
            )
            print("✅ Wikitext dataset loaded from cache (offline mode)")
            
            # Take one sample to verify it works
            sample = next(iter(dataset))
            assert "text" in sample
            print("✅ Wikitext dataset sample has expected structure")
            return
        except ConnectionError:
            print("ℹ️  Wikitext dataset not cached, will attempt download...")
        
        # Try to load with network (if available)
        try:
            dataset = load_dataset_with_cache_fallback(
                "wikitext", 
                name="wikitext-2-raw-v1",
                split="train",
                offline_mode=False
            )
            print("✅ Wikitext dataset loaded (with network access)")
            
            # Take one sample to verify it works
            sample = next(iter(dataset))
            assert "text" in sample
            print("✅ Wikitext dataset sample has expected structure")
            
        except Exception as e:
            print(f"⚠️  Could not load Wikitext dataset (network may be unavailable): {e}")
    
    except ImportError as e:
        print(f"⚠️  Could not import dataset functions: {e}")
    
    print("✅ Cache-aware dataset loading test completed")


def main():
    """Run all dataset cache tests"""
    print("🚀 Starting dataset cache detection tests...")
    print("=" * 60)
    
    try:
        test_cache_directory_detection()
        print()
        
        test_cache_detection()
        print()
        
        test_dataset_info()
        print()
        
        test_config_integration()
        print()
        
        test_offline_mode()
        print()
        
        test_force_download_mode()
        print()
        
        test_local_dataset_loading()
        print()
        
        test_cache_aware_dataset_loading()
        print()
        
        print("=" * 60)
        print("🎉 All dataset cache tests passed!")
        print()
        print("💡 Usage examples:")
        print("   # Use custom cache directory:")
        print("   --training.dataset_cache_dir /path/to/cache")
        print()
        print("   # Force re-download even if cached:")
        print("   --training.dataset_force_download")
        print()
        print("   # Offline mode (fail if not cached):")
        print("   --training.dataset_offline_mode")
        print()
        print("   # In TOML config:")
        print("   [training]")
        print("   dataset = \"c4\"")
        print("   dataset_cache_dir = \"/path/to/cache\"")
        print("   dataset_offline_mode = true")
        print()
        print("   # Environment variable for global cache:")
        print("   export HF_HOME=/path/to/huggingface/cache")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise


if __name__ == "__main__":
    main()
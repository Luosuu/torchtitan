# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from pathlib import Path
from typing import Optional

from datasets import Dataset, load_dataset
from huggingface_hub import HfApi
from torchtitan.tools.logging import logger


def get_hf_cache_dir() -> str:
    """Get HuggingFace cache directory from environment variable or default location."""
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return os.path.join(hf_home, "datasets")
    
    # Set default HF_HOME to ~/huggingface_cache if not set
    home = os.path.expanduser("~")
    default_hf_home = os.path.join(home, "huggingface_cache")
    
    # Set HF_HOME environment variable for this session
    if "HF_HOME" not in os.environ:
        os.environ["HF_HOME"] = default_hf_home
        logger.debug(f"Set HF_HOME to default location: {default_hf_home}")
    
    return os.path.join(default_hf_home, "datasets")


def _normalize_dataset_name(dataset_name: str) -> str:
    """Normalize dataset name for cache directory structure."""
    # HuggingFace replaces '/' with '--' in cache directory names
    return dataset_name.replace("/", "--")


def is_dataset_cached(
    dataset_path: str, 
    name: Optional[str] = None, 
    split: Optional[str] = None,
    cache_dir: Optional[str] = None,
    **kwargs  # Accept additional keyword arguments and ignore them
) -> bool:
    """
    Check if a HuggingFace dataset is already cached locally.
    
    Args:
        dataset_path: Dataset identifier (e.g., 'allenai/c4')
        name: Dataset configuration name (e.g., 'en')
        split: Dataset split (e.g., 'train', 'validation')
        cache_dir: Custom cache directory path
        
    Returns:
        True if dataset appears to be cached, False otherwise
    """
    cache_root = cache_dir or get_hf_cache_dir()
    
    # Handle local paths - if it's a local directory, consider it "cached"
    if os.path.isdir(dataset_path):
        logger.debug(f"Dataset path {dataset_path} is a local directory")
        return True
    
    # Normalize dataset name for cache structure
    normalized_name = _normalize_dataset_name(dataset_path)
    
    # Build potential cache paths
    cache_paths = []
    
    # Basic cache path
    base_cache_path = os.path.join(cache_root, normalized_name)
    cache_paths.append(base_cache_path)
    
    # With config name
    if name:
        config_cache_path = os.path.join(base_cache_path, name)
        cache_paths.append(config_cache_path)
    
    # Check if any cache directory exists and contains data files
    for cache_path in cache_paths:
        if os.path.isdir(cache_path):
            # Look for common dataset file patterns
            cache_dir_path = Path(cache_path)
            
            # Check for various file patterns that indicate cached data
            patterns = [
                "*.arrow",        # Arrow format files
                "*.parquet",      # Parquet files
                "*.json",         # JSON files
                "dataset_info.json",  # Dataset metadata
                "**/dataset_info.json",  # Nested dataset metadata
            ]
            
            for pattern in patterns:
                if list(cache_dir_path.glob(pattern)):
                    logger.debug(f"Found cached dataset files in {cache_path}")
                    return True
            
            # Also check subdirectories for split-specific data
            for subdir in cache_dir_path.iterdir():
                if subdir.is_dir():
                    for pattern in patterns:
                        if list(subdir.glob(pattern)):
                            logger.debug(f"Found cached dataset files in {subdir}")
                            return True
    
    logger.debug(f"Dataset {dataset_path} not found in cache")
    return False


def load_dataset_with_cache_fallback(
    dataset_path: str,
    force_download: bool = False,
    offline_mode: bool = False,
    cache_dir: Optional[str] = None,
    **kwargs
) -> Dataset:
    """
    Load dataset with intelligent cache detection and fallback.
    
    Args:
        dataset_path: Dataset identifier or local path
        force_download: Force re-download even if cached
        offline_mode: Fail if dataset not cached (no network access)
        cache_dir: Custom cache directory
        **kwargs: Additional arguments passed to load_dataset
        
    Returns:
        Loaded dataset
        
    Raises:
        ConnectionError: If offline_mode=True and dataset not cached
    """
    # Check if dataset is cached (unless forcing download)
    # Extract relevant parameters for cache checking
    cache_check_kwargs = {k: v for k, v in kwargs.items() if k in ['name', 'split']}
    if not force_download and is_dataset_cached(dataset_path, cache_dir=cache_dir, **cache_check_kwargs):
        logger.info(f"Loading dataset '{dataset_path}' from local cache")
        try:
            # Set cache_dir if provided
            load_kwargs = kwargs.copy()
            if cache_dir:
                load_kwargs['cache_dir'] = cache_dir
                
            return load_dataset(dataset_path, **load_kwargs)
        except Exception as e:
            if offline_mode:
                raise ConnectionError(
                    f"Failed to load cached dataset '{dataset_path}' in offline mode: {e}"
                )
            logger.warning(f"Failed to load from cache, falling back to download: {e}")
    
    # Handle offline mode
    if offline_mode:
        raise ConnectionError(
            f"Dataset '{dataset_path}' not found in cache and offline_mode=True"
        )
    
    # Download from HuggingFace Hub
    logger.info(f"Downloading dataset '{dataset_path}' from HuggingFace Hub")
    load_kwargs = kwargs.copy()
    if cache_dir:
        load_kwargs['cache_dir'] = cache_dir
        
    return load_dataset(dataset_path, **load_kwargs)


def get_cached_dataset_info(dataset_path: str, cache_dir: Optional[str] = None) -> dict:
    """
    Get information about a cached dataset.
    
    Args:
        dataset_path: Dataset identifier
        cache_dir: Custom cache directory
        
    Returns:
        Dictionary with cache information
    """
    cache_root = cache_dir or get_hf_cache_dir()
    normalized_name = _normalize_dataset_name(dataset_path)
    base_cache_path = os.path.join(cache_root, normalized_name)
    
    info = {
        "dataset_path": dataset_path,
        "is_cached": is_dataset_cached(dataset_path, cache_dir=cache_dir),
        "cache_directory": base_cache_path,
        "cache_exists": os.path.isdir(base_cache_path),
        "cache_size_mb": 0,
    }
    
    if info["cache_exists"]:
        # Calculate cache size
        total_size = 0
        try:
            for dirpath, dirnames, filenames in os.walk(base_cache_path):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    if os.path.isfile(filepath):
                        total_size += os.path.getsize(filepath)
            info["cache_size_mb"] = total_size / (1024 * 1024)  # Convert to MB
        except (OSError, PermissionError):
            info["cache_size_mb"] = -1  # Indicate error calculating size
    
    return info


def clear_dataset_cache(dataset_path: str, cache_dir: Optional[str] = None) -> bool:
    """
    Clear cached data for a specific dataset.
    
    Args:
        dataset_path: Dataset identifier
        cache_dir: Custom cache directory
        
    Returns:
        True if cache was cleared successfully
    """
    import shutil
    
    cache_root = cache_dir or get_hf_cache_dir()
    normalized_name = _normalize_dataset_name(dataset_path)
    cache_path = os.path.join(cache_root, normalized_name)
    
    if os.path.isdir(cache_path):
        try:
            shutil.rmtree(cache_path)
            logger.info(f"Cleared cache for dataset '{dataset_path}' at {cache_path}")
            return True
        except (OSError, PermissionError) as e:
            logger.error(f"Failed to clear cache for '{dataset_path}': {e}")
            return False
    else:
        logger.info(f"No cache found for dataset '{dataset_path}'")
        return True
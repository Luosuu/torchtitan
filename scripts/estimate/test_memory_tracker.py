#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Test script to validate the memory tracker integration.
"""

import tempfile
import os
from collections import namedtuple

from torchtitan.components.memory_tracker import (
    MemoryDataCollector, 
    MemorySnapshot, 
    initialize_memory_collector,
    get_memory_collector
)

# Create a fake DeviceMemStats for testing to avoid circular import issues
DeviceMemStats = namedtuple(
    "DeviceMemStats",
    [
        "max_active_gib",
        "max_active_pct", 
        "max_reserved_gib",
        "max_reserved_pct",
        "num_alloc_retries",
        "num_ooms",
        "peak_allocated_gib",
        "peak_allocated_pct",
    ],
)


def test_memory_collector():
    """Test the basic functionality of the memory data collector."""
    print("Testing MemoryDataCollector...")
    
    # Create a collector
    collector = MemoryDataCollector(max_history=10)
    
    # Create some fake memory stats
    fake_stats = DeviceMemStats(
        max_active_gib=10.5,
        max_active_pct=25.0,
        max_reserved_gib=12.0,
        max_reserved_pct=30.0,
        num_alloc_retries=0,
        num_ooms=0,
        peak_allocated_gib=11.2,
        peak_allocated_pct=28.0
    )
    
    # Record some data
    for step in range(1, 6):
        metadata = {"loss": 2.5 - (step * 0.1), "tps": 1000 + step * 10}
        collector.record_memory_data(step, fake_stats, metadata)
    
    # Test getting current memory
    current = collector.get_current_memory()
    assert current is not None
    assert current.step == 5
    print(f"✓ Current memory: step {current.step}, peak_allocated: {current.device_mem_stats.peak_allocated_gib} GiB")
    
    # Test getting history
    history = collector.get_memory_history()
    assert len(history) == 5
    print(f"✓ History contains {len(history)} snapshots")
    
    # Test peak memory calculations
    peak_allocated = collector.get_peak_memory_allocated()
    peak_reserved = collector.get_peak_memory_reserved()
    assert peak_allocated == 11.2
    assert peak_reserved == 12.0
    print(f"✓ Peak allocated: {peak_allocated} GiB, Peak reserved: {peak_reserved} GiB")
    
    # Test summary
    summary = collector.get_memory_summary()
    print(f"✓ Summary: {summary['total_steps']} steps, peak {summary['peak_allocated_gib']} GiB")
    
    # Test JSON export
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
        temp_path = temp_file.name
    
    try:
        collector.export_to_json(temp_path)
        print(f"✓ JSON export successful: {temp_path}")
        
        # Verify the file exists and has content
        with open(temp_path, 'r') as f:
            content = f.read()
            assert len(content) > 100  # Should have substantial content
            print(f"✓ JSON file contains {len(content)} characters")
    finally:
        os.unlink(temp_path)
    
    print("✓ All MemoryDataCollector tests passed!")
    return True


def test_global_collector():
    """Test the global collector functionality."""
    print("\\nTesting global memory collector...")
    
    # Initialize global collector
    collector = initialize_memory_collector(max_history=5)
    assert collector is not None
    print("✓ Global collector initialized")
    
    # Get global collector
    retrieved = get_memory_collector()
    assert retrieved is collector
    print("✓ Global collector retrieved successfully")
    
    # Test that it works the same way
    fake_stats = DeviceMemStats(
        max_active_gib=5.0,
        max_active_pct=12.5,
        max_reserved_gib=6.0,
        max_reserved_pct=15.0,
        num_alloc_retries=0,
        num_ooms=0,
        peak_allocated_gib=5.5,
        peak_allocated_pct=13.75
    )
    
    retrieved.record_memory_data(1, fake_stats, {"test": "global"})
    current = retrieved.get_current_memory()
    assert current.step == 1
    assert current.metadata["test"] == "global"
    print("✓ Global collector records data correctly")
    
    print("✓ All global collector tests passed!")
    return True


def test_callback_system():
    """Test the callback system."""
    print("\\nTesting callback system...")
    
    collector = MemoryDataCollector()
    
    # Track calls to the callback
    callback_calls = []
    
    def test_callback(snapshot: MemorySnapshot):
        callback_calls.append(snapshot.step)
    
    # Add callback
    collector.add_callback(test_callback)
    
    # Record some data
    fake_stats = DeviceMemStats(
        max_active_gib=1.0, max_active_pct=5.0,
        max_reserved_gib=1.2, max_reserved_pct=6.0,
        num_alloc_retries=0, num_ooms=0,
        peak_allocated_gib=1.1, peak_allocated_pct=5.5
    )
    
    collector.record_memory_data(1, fake_stats)
    collector.record_memory_data(2, fake_stats)
    
    # Check that callback was called
    assert len(callback_calls) == 2
    assert callback_calls == [1, 2]
    print("✓ Callback system works correctly")
    
    # Remove callback
    collector.remove_callback(test_callback)
    collector.record_memory_data(3, fake_stats)
    
    # Should not have been called again
    assert len(callback_calls) == 2
    print("✓ Callback removal works correctly")
    
    print("✓ All callback tests passed!")
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing TorchTitan Memory Tracker System")
    print("=" * 60)
    
    try:
        success = True
        success &= test_memory_collector()
        success &= test_global_collector()
        success &= test_callback_system()
        
        if success:
            print("\\n" + "=" * 60)
            print("🎉 ALL TESTS PASSED! Memory tracker system is working correctly.")
            print("=" * 60)
            return 0
        else:
            print("\\n" + "=" * 60)
            print("❌ Some tests failed!")
            print("=" * 60)
            return 1
            
    except Exception as e:
        print(f"\\n❌ Test error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
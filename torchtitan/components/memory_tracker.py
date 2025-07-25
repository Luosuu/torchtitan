# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Memory data collection interface for programmatic access to memory metrics.
Provides structured access to memory data without requiring log parsing.
"""

import json
import time
from collections import deque
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from torchtitan.components.metrics import DeviceMemStats


@dataclass
class MemorySnapshot:
    """A snapshot of memory data at a specific training step."""
    step: int
    timestamp: float
    device_mem_stats: "DeviceMemStats"
    metadata: Dict[str, Any]


class MemoryDataCallback(Protocol):
    """Protocol for memory data callbacks."""
    
    def on_memory_data(self, snapshot: MemorySnapshot) -> None:
        """Called when new memory data is available."""
        ...


class MemoryDataCollector:
    """Collects and provides structured access to memory data during training."""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.history: deque[MemorySnapshot] = deque(maxlen=max_history)
        self.callbacks: List[MemoryDataCallback] = []
        self.current_snapshot: Optional[MemorySnapshot] = None
        
    def add_callback(self, callback: MemoryDataCallback) -> None:
        """Register a callback to receive memory data updates."""
        self.callbacks.append(callback)
        
    def remove_callback(self, callback: MemoryDataCallback) -> None:
        """Unregister a memory data callback."""
        if callback in self.callbacks:
            self.callbacks.remove(callback)
            
    def record_memory_data(
        self, 
        step: int, 
        device_mem_stats: "DeviceMemStats", 
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record new memory data for the given step."""
        snapshot = MemorySnapshot(
            step=step,
            timestamp=time.time(),
            device_mem_stats=device_mem_stats,
            metadata=metadata or {}
        )
        
        self.current_snapshot = snapshot
        self.history.append(snapshot)
        
        # Notify all callbacks
        for callback in self.callbacks:
            try:
                callback.on_memory_data(snapshot)
            except Exception as e:
                # Don't let callback errors break training
                print(f"Warning: Memory data callback error: {e}")
                
    def get_current_memory(self) -> Optional[MemorySnapshot]:
        """Get the most recent memory snapshot."""
        return self.current_snapshot
        
    def get_memory_history(self, steps: Optional[int] = None) -> List[MemorySnapshot]:
        """Get memory history, optionally limited to last N steps."""
        if steps is None:
            return list(self.history)
        else:
            return list(self.history)[-steps:]
            
    def get_peak_memory_allocated(self) -> Optional[float]:
        """Get the peak allocated memory across all recorded steps."""
        if not self.history:
            return None
        return max(snapshot.device_mem_stats.peak_allocated_gib for snapshot in self.history)
        
    def get_peak_memory_reserved(self) -> Optional[float]:
        """Get the peak reserved memory across all recorded steps."""
        if not self.history:
            return None
        return max(snapshot.device_mem_stats.max_reserved_gib for snapshot in self.history)
        
    def export_to_json(self, filepath: str) -> None:
        """Export all memory data to a JSON file."""
        data = {
            "metadata": {
                "export_timestamp": time.time(),
                "total_snapshots": len(self.history),
                "max_history": self.max_history
            },
            "snapshots": [
                {
                    "step": snapshot.step,
                    "timestamp": snapshot.timestamp,
                    "memory_stats": asdict(snapshot.device_mem_stats),
                    "metadata": snapshot.metadata
                }
                for snapshot in self.history
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
            
    def get_memory_summary(self) -> Dict[str, Any]:
        """Get a summary of memory usage statistics."""
        if not self.history:
            return {"error": "No memory data available"}
            
        allocated_values = [s.device_mem_stats.peak_allocated_gib for s in self.history]
        reserved_values = [s.device_mem_stats.max_reserved_gib for s in self.history]
        
        return {
            "total_steps": len(self.history),
            "peak_allocated_gib": max(allocated_values),
            "avg_allocated_gib": sum(allocated_values) / len(allocated_values),
            "peak_reserved_gib": max(reserved_values),
            "avg_reserved_gib": sum(reserved_values) / len(reserved_values),
            "first_step": self.history[0].step if self.history else None,
            "last_step": self.history[-1].step if self.history else None,
        }


class FileExportCallback:
    """Callback that exports memory data to a file when training completes."""
    
    def __init__(self, output_path: str, export_on_steps: Optional[List[int]] = None):
        self.output_path = output_path
        self.export_on_steps = export_on_steps or []
        self.collector: Optional[MemoryDataCollector] = None
        
    def set_collector(self, collector: MemoryDataCollector) -> None:
        """Set the memory collector reference for export."""
        self.collector = collector
        
    def on_memory_data(self, snapshot: MemorySnapshot) -> None:
        """Export data when specific steps are reached."""
        if snapshot.step in self.export_on_steps and self.collector:
            export_path = f"{self.output_path}_step_{snapshot.step}.json"
            self.collector.export_to_json(export_path)
            print(f"Memory data exported to: {export_path}")


# Global memory data collector instance
_global_memory_collector: Optional[MemoryDataCollector] = None


def get_memory_collector() -> Optional[MemoryDataCollector]:
    """Get the global memory data collector instance."""
    return _global_memory_collector


def initialize_memory_collector(max_history: int = 1000) -> MemoryDataCollector:
    """Initialize the global memory data collector."""
    global _global_memory_collector
    _global_memory_collector = MemoryDataCollector(max_history=max_history)
    return _global_memory_collector


def cleanup_memory_collector() -> None:
    """Clean up the global memory data collector."""
    global _global_memory_collector
    _global_memory_collector = None
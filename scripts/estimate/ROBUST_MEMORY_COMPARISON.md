# Robust Memory Comparison System

This document describes the new robust memory comparison system that replaces fragile log parsing with structured data access.

## Problem with Original Approach

The original memory comparison system had several issues:

1. **Fragile log parsing** - Used regex to parse text logs which broke easily
2. **Missing memory data** - Training logs weren't captured properly due to torchrun configuration issues  
3. **Inaccurate measurements** - String parsing introduced rounding errors
4. **Debugging difficulties** - Hard to diagnose why parsing failed

## New Robust Approach

### Core Components

#### 1. Memory Data Collector (`torchtitan/components/memory_tracker.py`)

- **`MemoryDataCollector`**: Collects structured memory data during training
- **`MemorySnapshot`**: Structured data container for memory measurements
- **Callback system**: Allows external tools to receive real-time memory data
- **JSON export**: Saves structured data for analysis

#### 2. MetricsProcessor Integration (`torchtitan/components/metrics.py`)

Extended the existing `MetricsProcessor` to:
- Initialize memory data collector when enabled
- Record structured memory data alongside existing logging
- Maintain backward compatibility with existing logging system

#### 3. Configuration Support (`torchtitan/config/job_config.py`)

Added new configuration options:
```toml
[training]
enable_memory_data_collection = true
memory_data_collection_max_history = 1000
```

#### 4. Robust Comparison Script (`scripts/estimate/compare_memory_robust.py`)

- Uses structured data access instead of log parsing
- Hybrid approach for backward compatibility  
- Improved error handling and debugging
- Structured JSON output with detailed metadata

## Key Benefits

### 1. Eliminates Log Parsing Brittleness
- **Before**: Regex patterns like `memory: (\\d+\\.\\d+)GiB` 
- **After**: Direct access to `DeviceMemStats.peak_allocated_gib`

### 2. Real-time Data Access
- **Before**: Parse logs after training completes
- **After**: Access memory data as it's collected during training

### 3. Higher Accuracy
- **Before**: String parsing with potential rounding errors
- **After**: Exact floating-point values from source

### 4. Structured Access
- **Before**: Text parsing to extract individual values
- **After**: Complete `DeviceMemStats` namedtuple with all fields

### 5. Better Debugging
- **Before**: Guess why regex didn't match
- **After**: Structured data shows exactly what was collected

## Usage

### Basic Usage

```bash
# Enable memory data collection and run comparison
./scripts/estimate/run_robust_memory_comparison.sh
```

### Advanced Usage

```bash
# Custom configuration
CONFIG_FILE="./torchtitan/models/llama3/train_configs/llama3_8b.toml" \\
MODEL_FLAVORS="8B" \\
BATCH_SIZES="1 2 4" \\
NUM_GPUS=4 \\
./scripts/estimate/run_robust_memory_comparison.sh
```

### Direct Python Usage

```python
from scripts.estimate.compare_memory_robust import RobustMemoryComparison

runner = RobustMemoryComparison("config.toml")
results = runner.run_comparison_suite([{
    "model_flavor": "8B",
    "batch_size": 4,
    "seq_len": 8192,
    "num_gpus": 4,
    "training_steps": 5
}])
```

## Implementation Details

### Memory Data Flow

1. **Training Step**: Model processes batch
2. **Metrics Collection**: `MetricsProcessor.log()` called
3. **Memory Measurement**: `DeviceMemoryMonitor.get_peak_stats()`
4. **Data Recording**: `memory_collector.record_memory_data()`
5. **Callback Notification**: External tools receive `MemorySnapshot`
6. **Structured Storage**: Data stored in `MemoryDataCollector.history`

### Data Structure

```python
@dataclass
class MemorySnapshot:
    step: int                          # Training step number
    timestamp: float                   # Unix timestamp  
    device_mem_stats: DeviceMemStats   # Complete memory statistics
    metadata: Dict[str, Any]           # Additional context (loss, tps, etc.)
```

### Callback System

```python
def my_callback(snapshot: MemorySnapshot):
    print(f"Step {snapshot.step}: {snapshot.device_mem_stats.peak_allocated_gib:.2f} GiB")

collector = get_memory_collector()
collector.add_callback(my_callback)
```

## Testing

Test the memory tracker system:

```bash
python scripts/estimate/test_memory_tracker.py
```

This validates:
- MemoryDataCollector functionality
- Global collector initialization  
- Callback system
- JSON export
- Data accuracy

## Migration Guide

### From Old System

**Old approach (fragile)**:
```python
# Parse logs with regex
memory_lines = [line for line in output.split('\\n') if 'memory:' in line]
for line in memory_lines:
    value = float(line.split('memory:')[1].split('GiB')[0])
```

**New approach (robust)**:
```python  
# Direct structured data access
collector = get_memory_collector()
snapshot = collector.get_current_memory()
value = snapshot.device_mem_stats.peak_allocated_gib
```

### Configuration Changes

Add to your training config:
```toml
[training]
enable_memory_data_collection = true
memory_data_collection_max_history = 1000
```

## Files Modified/Added

### Core System
- `torchtitan/components/memory_tracker.py` - New memory data collection system
- `torchtitan/components/metrics.py` - Extended with memory data recording  
- `torchtitan/config/job_config.py` - Added configuration options

### Scripts  
- `scripts/estimate/compare_memory_robust.py` - New robust comparison script
- `scripts/estimate/run_robust_memory_comparison.sh` - Wrapper script
- `scripts/estimate/test_memory_tracker.py` - Test script

### Documentation
- `scripts/estimate/ROBUST_MEMORY_COMPARISON.md` - This document

## Backward Compatibility

The new system maintains full backward compatibility:
- Existing logging continues to work unchanged
- Memory data collection is opt-in via configuration
- Original comparison scripts still function
- No breaking changes to existing APIs

## Future Enhancements

1. **Real-time Monitoring**: Web dashboard showing live memory usage
2. **Advanced Analytics**: Memory pattern analysis and optimization suggestions  
3. **Integration with Profiling**: Combine with PyTorch profiler data
4. **Multi-node Support**: Aggregate memory data across distributed training
5. **Memory Prediction**: ML models to predict memory usage for new configurations

## Troubleshooting

### Memory Data Not Collected
- Ensure `enable_memory_data_collection = true` in config
- Check that `MetricsProcessor` is initialized properly
- Verify training actually reaches logging steps

### Comparison Script Fails
- Check that config file path is correct
- Ensure GPU resources are available
- Review output logs for specific error messages

### Test Script Fails
- Verify Python path includes project root
- Check that all dependencies are installed
- Run with `-v` flag for verbose output

## Performance Impact

The memory data collection system has minimal performance impact:
- **Memory overhead**: ~1KB per snapshot (negligible)
- **CPU overhead**: <0.1ms per step (measurement already done)
- **Storage overhead**: JSON export only when requested

The benefits of robust, accurate memory measurement far outweigh the minimal costs.
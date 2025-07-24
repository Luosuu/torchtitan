# FSDP Memory Estimation

This directory contains tools for estimating and analyzing memory usage in FSDP (Fully Sharded Data Parallel) training with TorchTitan.

## Overview

The memory estimator helps predict peak memory usage for distributed LLM training, which is critical for determining the largest batch size that can be used without running into OOM (Out Of Memory) errors.

## Tools

### 1. Memory Estimator (`estimation.py`)

**Purpose**: Estimates peak memory usage for FSDP/HSDP training using PyTorch's `FSDPMemTracker`.

**How it works**:
- Uses `FSDPMemTracker` from PyTorch to track memory during simulated training
- Runs in `FakeTensorMode` by default (no actual memory allocation) or real mode
- Simulates 2 training iterations with dummy data
- Tracks memory through FSDP operations (forward pass, backward pass, optimizer step)
- Provides module-wise memory breakdown

**Usage**:
```bash
# Basic usage with debug model
./run_memory_estimation.sh

# With custom configuration
CONFIG_FILE="./torchtitan/models/llama3/train_configs/llama3_8b.toml" \
NGPU=8 ./run_memory_estimation.sh

# With additional overrides
./run_memory_estimation.sh --training.local_batch_size 4 --model.flavor 8B
```

**What it supports**:
- ✅ FSDP (Fully Sharded Data Parallel)
- ✅ HSDP (Hybrid Sharded Data Parallel)
- ✅ Module-wise memory breakdown
- ✅ Peak memory tracking per iteration
- ✅ Accuracy comparison between tracker and actual memory stats

**Limitations**:
- ❌ No support for Tensor Parallelism (TP)
- ❌ No support for Pipeline Parallelism (PP) 
- ❌ No support for Context Parallelism (CP)
- ❌ No support for Expert Parallelism (EP)
- ❌ Compile mode not supported (automatically disabled)

### 2. Runner Script (`run_memory_estimation.sh`)

**Purpose**: Wrapper script to run the memory estimator with proper environment setup.

**Environment Variables**:
- `NGPU`: Number of GPUs (default: 8)
- `NNODES`: Number of nodes (default: 1)
- `CONFIG_FILE`: Path to training config file (default: debug_model.toml)

## Output Interpretation

### Memory Statistics Output

```
Peak Memory at iter: 0
┌──────────────────────┬─────────────┬─────────────┬─────────────┐
│ Event                │   Allocated │    Reserved │       Total │
├──────────────────────┼─────────────┼─────────────┼─────────────┤
│ Model                │     XXX MiB │     XXX MiB │     XXX MiB │
│ Gradient             │     XXX MiB │     XXX MiB │     XXX MiB │
│ Optimizer            │     XXX MiB │     XXX MiB │     XXX MiB │
│ Activation           │     XXX MiB │     XXX MiB │     XXX MiB │
│ Temp                 │     XXX MiB │     XXX MiB │     XXX MiB │
└──────────────────────┴─────────────┴─────────────┴─────────────┘
```

**Memory Categories**:
- **Model**: Memory used by model parameters
- **Gradient**: Memory used by gradients
- **Optimizer**: Memory used by optimizer states (momentum, variance, etc.)
- **Activation**: Memory used by forward pass activations
- **Temp**: Temporary memory used during computations

### Final Summary Output

```
peak active: X.XX GiB | peak reserved: X.XX GiB | num_retries: X
Tracker Max: X.XX GiB
Tracker Accuracy: X.XX
```

**Metrics Explained**:
- **peak active**: Peak memory actively used by tensors
- **peak reserved**: Peak memory reserved by CUDA memory allocator
- **num_retries**: Number of memory allocation retries (indicates memory pressure)
- **Tracker Max**: Maximum memory tracked by FSDPMemTracker
- **Tracker Accuracy**: Ratio of tracked memory to actual peak active memory

### Module-wise Breakdown

The estimator also provides a detailed breakdown by model modules:

```
┌─────────────────────────────────┬─────────────┬─────────────┬─────────────┐
│ Module                          │   Allocated │    Reserved │       Total │
├─────────────────────────────────┼─────────────┼─────────────┼─────────────┤
│ layers.0.attention              │     XXX MiB │     XXX MiB │     XXX MiB │
│ layers.0.feed_forward           │     XXX MiB │     XXX MiB │     XXX MiB │
│ layers.1.attention              │     XXX MiB │     XXX MiB │     XXX MiB │
│ ...                             │         ... │         ... │         ... │
└─────────────────────────────────┴─────────────┴─────────────┴─────────────┘
```

This helps identify which model components use the most memory.

## Configuration Options

### Memory Estimation Specific

In your `.toml` config file:

```toml
[memory_estimation]
enabled = true                    # Enable memory estimation mode
disable_fake_mode = false        # Use real tensors instead of fake tensors
```

**Note**: When `disable_fake_mode = true`, the estimator uses real memory allocation and provides more accurate results but requires actual GPU memory.

### Supported Training Configurations

The estimator works with standard TorchTitan configurations:

```toml
[training]
local_batch_size = 8             # Batch size per GPU
seq_len = 2048                   # Sequence length

[parallelism]
data_parallel_shard_degree = -1  # FSDP sharding (-1 = use all available)
data_parallel_replicate_degree = 1  # HSDP replication
```

## Best Practices

1. **Start with fake mode**: Use default fake tensor mode for quick estimates
2. **Validate with real mode**: Use `--memory_estimation.disable_fake_mode` for accuracy validation
3. **Test multiple batch sizes**: Run with different `local_batch_size` values to understand scaling
4. **Consider sequence length**: Memory usage scales quadratically with sequence length for attention
5. **Account for safety margin**: Add 10-20% buffer to estimates for production usage

## Common Issues

### Accuracy Concerns
- FakeTensorMode provides estimates but may not capture all memory patterns
- Real mode (`disable_fake_mode=true`) gives more accurate results but requires GPU memory
- Tracker accuracy varies by model architecture and parallelism strategy

### Unsupported Configurations
- The estimator will warn and exit for unsupported parallelism modes
- Compile mode is automatically disabled for compatibility

### Memory Pressure Indicators
- High `num_retries` indicates memory pressure
- Large gap between reserved and active memory suggests fragmentation
- Low tracker accuracy may indicate missing memory tracking

## Example Workflows

### Basic Memory Estimation
```bash
# Estimate memory for debug model
./run_memory_estimation.sh

# Estimate for 8B model with larger batch size
CONFIG_FILE="./torchtitan/models/llama3/train_configs/llama3_8b.toml" \
./run_memory_estimation.sh --training.local_batch_size 2
```

### Batch Size Optimization
```bash
# Test different batch sizes to find maximum
for bs in 1 2 4 8; do
  echo "Testing batch size: $bs"
  ./run_memory_estimation.sh --training.local_batch_size $bs
done
```

### Real vs Fake Mode Comparison
```bash
# Get estimate with fake tensors
./run_memory_estimation.sh > fake_mode_results.txt

# Get accurate measurement with real tensors
./run_memory_estimation.sh --memory_estimation.disable_fake_mode > real_mode_results.txt
```
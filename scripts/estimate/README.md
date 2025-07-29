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

## Technical Deep Dive: How FSDP2 Memory Estimation Works

This section explains the internal architecture and step-by-step process of how FSDP2 memory estimation works under the hood.

### Architecture Overview

The FSDP2 memory estimation system is built on a multi-layer architecture:

```
┌─────────────────────────────────────────────────────────┐
│ TorchTitan Memory Estimator (estimation.py)            │
│ - Orchestrates estimation process                       │
│ - Handles configuration and setup                       │
└─────────────────────────────────────────────────────────┘
                               │
┌─────────────────────────────────────────────────────────┐
│ FSDPMemTracker (fsdp2_mem_tracker.py)                  │
│ - FSDP-specific memory tracking                         │
│ - Instruments FSDP operations                           │
│ - Categorizes memory by FSDP semantics                  │
└─────────────────────────────────────────────────────────┘
                               │
┌─────────────────────────────────────────────────────────┐
│ Base MemTracker (mem_tracker.py)                       │
│ - Core memory tracking infrastructure                   │
│ - Weak reference management                             │
│ - PyTorch dispatch mode integration                     │
└─────────────────────────────────────────────────────────┘
                               │
┌─────────────────────────────────────────────────────────┐
│ PyTorch Memory System                                   │
│ - CUDA memory allocator                                 │
│ - Tensor storage tracking                               │
│ - Device memory management                              │
└─────────────────────────────────────────────────────────┘
```

#### Key Components:

- **`estimation.py`**: Main orchestrator that sets up FSDP training simulation and runs the memory tracking
- **`fsdp2_mem_tracker.py`**: FSDP-specific tracker that instruments FSDP forward/backward hooks and categorizes memory
- **`mem_tracker.py`**: Base infrastructure providing core memory tracking via PyTorch TorchDispatchMode

### Memory Categorization System

The tracker uses sophisticated categorization to attribute memory usage to different FSDP components:

#### FSDP-Specific Memory Types:

- **SHARDED_PARAM**: Parameters in their default sharded state across ranks
- **UNSHARDED_PARAM**: Parameters temporarily gathered (unsharded) for computation
- **SHARDED_GRAD**: Gradients in sharded state after reduce-scatter
- **UNSHARDED_GRAD**: Gradients initially computed in unsharded form
- **ALL_GATHER**: Communication buffers used during parameter unsharding
- **REDUCE_SCATTER**: Communication buffers used during gradient reduction
- **ACTIVATION**: Forward pass activations tracked via TorchDispatchMode
- **TEMP**: Temporary tensors created during backward computation
- **OPTIMIZER**: Optimizer state tensors (momentum, variance, etc.)
- **BUFFER**: Model buffers (BatchNorm statistics, embeddings, etc.)

#### Memory Attribution Logic:

**Forward Pass Memory Flow:**
```
SHARDED_PARAM → [all-gather] → UNSHARDED_PARAM + ALL_GATHER buffers
                ↓ [computation] 
              ACTIVATIONS created
                ↓ [post-forward]
UNSHARDED_PARAM → [reshard] → SHARDED_PARAM (memory freed)
```

**Backward Pass Memory Flow:**
```
SHARDED_PARAM → [all-gather] → UNSHARDED_PARAM + ALL_GATHER buffers
                ↓ [grad computation]
              UNSHARDED_GRAD + TEMP tensors created
                ↓ [post-backward]
UNSHARDED_GRAD → [reduce-scatter] → SHARDED_GRAD + REDUCE_SCATTER buffers
```

### FSDP State Transitions and Memory Snapshots

The tracker captures detailed memory snapshots at critical points by instrumenting FSDP's internal methods:

#### FSDP Module States:

- **BEF_PRE_FW**: Before parameter unsharding
- **AFT_PRE_FW**: After parameter unsharding
- **BEF_POST_FW**: Before parameter resharding
- **AFT_POST_FW**: After parameter resharding
- **BEF_PRE_BW**: Before gradient computation setup
- **AFT_PRE_BW**: After gradient computation setup
- **BEF_POST_BW**: Before gradient reduction
- **AFT_POST_BW**: After gradient reduction
- **PEAK_FW/PEAK_BW**: Peak memory during forward/backward passes

#### Memory Snapshot Timeline:

```
Training Iteration:
┌─────────────────────────────────────────────────────────────────┐
│                    FORWARD PASS                                 │
├─────────────────────────────────────────────────────────────────┤
│ BEF_PRE_FW → [unsharding] → AFT_PRE_FW → [compute] →           │
│ BEF_POST_FW → [resharding] → AFT_POST_FW                       │
├─────────────────────────────────────────────────────────────────┤
│                    BACKWARD PASS                                │  
├─────────────────────────────────────────────────────────────────┤
│ BEF_PRE_BW → [grad setup] → AFT_PRE_BW → [grad compute] →      │
│ BEF_POST_BW → [grad reduction] → AFT_POST_BW                   │
└─────────────────────────────────────────────────────────────────┘
```

### Hook Instrumentation Strategy

The tracker works by wrapping three key FSDP methods:

1. **Pre-Forward Hook**: Captures parameter unsharding (SHARDED_PARAM → UNSHARDED_PARAM)
2. **Post-Forward Hook**: Captures parameter resharding (UNSHARDED_PARAM → SHARDED_PARAM)
3. **Pre/Post-Backward Hooks**: Capture gradient computation and reduction phases

Each hook takes memory snapshots before and after the original FSDP operation, allowing precise attribution of memory changes to specific FSDP behaviors.

### Step-by-Step Tracking Process

#### Initialization Phase:
1. Set up fake process group for distributed simulation
2. Create model on meta device (no actual memory allocation in fake mode)
3. Apply FSDP parallelization to wrap modules
4. Initialize FSDPMemTracker with model and optimizer
5. Register input tensors for tracking

#### Training Loop (per iteration):
1. **Forward Pass**: For each FSDP module, capture memory at 4 state transitions
2. **Backward Pass**: For each FSDP module, capture memory at 4 state transitions  
3. **Optimizer Step**: Track optimizer state memory allocation
4. **Peak Calculation**: Update running peak memory for each module

#### Final Analysis:
1. Compare tracker estimates with CUDA memory statistics
2. Calculate accuracy ratio (tracker_peak / actual_peak)
3. Generate module-wise memory breakdown
4. Report memory pressure indicators (allocation retries)

### Fake Mode vs Real Mode

#### Fake Mode (Default):
- Uses PyTorch's `FakeTensorMode` - tensors have metadata but no data
- No real GPU memory allocation
- Fast execution, can estimate models larger than available memory
- Memory calculated from tensor metadata: `size × element_size × cuda_block_size`

#### Real Mode (`--memory_estimation.disable_fake_mode`):
- Uses real tensors with actual memory allocation
- Slower but provides accuracy validation against CUDA stats
- Required for measuring tracker accuracy
- Limited by available GPU memory

#### CUDA Memory Allocation Granularity:
The tracker accounts for CUDA's block allocation behavior:
```python
# CUDA allocates in 512-byte minimum blocks  
mem = size * element_size
if device.type == "cuda":
    return math.ceil(mem / 512) * 512  # Round up to block size
```

### Accuracy Analysis

The system provides comprehensive accuracy validation:

#### Metrics:
- **peak_active**: Real GPU memory actively used by tensors
- **peak_reserved**: Memory reserved by CUDA allocator (includes fragmentation)
- **tracker_peak**: Maximum memory tracked by FSDPMemTracker
- **num_retries**: Memory allocation retry count (indicates pressure)
- **accuracy**: `tracker_peak / peak_active` ratio

#### Interpretation:
- **Accuracy near 1.0**: Excellent tracking precision
- **High retries**: Approaching memory limits
- **Large reserved/active gap**: Memory fragmentation
- **Low accuracy**: Potential missing memory categories

### Practical Applications

This detailed tracking enables:

1. **Batch Size Optimization**: Test different batch sizes to find memory limits
2. **Module-wise Analysis**: Identify which model components use most memory
3. **Memory Pattern Understanding**: See when memory peaks occur during training
4. **Communication Overhead**: Quantify all-gather and reduce-scatter buffer costs
5. **Production Planning**: Add appropriate safety margins based on accuracy analysis

The technical implementation provides the foundation for accurate memory estimation that helps optimize distributed LLM training configurations.

## How Peak Memory Measurement Works

A common question is exactly how the memory estimator measures peak memory usage during forward and backward passes. The answer involves a sophisticated **continuous tracking and comparison system** rather than taking snapshots at specific points.

### Continuous Memory Tracking via TorchDispatchMode

The base `MemTracker` class extends `TorchDispatchMode`, which means it intercepts **every tensor operation** that happens during execution:

```python
class MemTracker(TorchDispatchMode):
    def __init__(self):
        self._curr_mem_snap: dict[torch.device, dict[str, int]] = {}  # Current memory state
        self._peak_mem: dict[torch.device, int] = {}                  # Global peak per device
        self._peak_mem_snap: dict[torch.device, dict[str, int]] = {} # Peak snapshot details
```

### Real-Time Memory Updates

Every time a tensor is created, deleted, or resized, the tracker updates the current memory snapshot via `_update_snap()`. This categorizes memory by type (PARAM, GRAD, ACT, etc.) and maintains a running total.

### Peak Detection Mechanism

The key method `_update_peak_stats()` is called continuously and compares current memory against stored peaks:

```python
def _update_peak_stats(self, peak_state: _State) -> None:
    curr_snap = self._curr_mem_snap  # Current memory usage
    
    # Update per-module peaks
    for mod_stats in self.memory_tracking.values():
        if mod_stats.mod_fqn in self._mod_tracker.parents:  # Only active modules
            for dev, dev_snap in curr_snap.items():
                # Compare current total against module's recorded peak
                if mod_stats.local_peak.get(dev, 0) < dev_snap[_TOTAL_KEY]:
                    mod_stats.local_peak[dev] = dev_snap[_TOTAL_KEY]  # New peak!
                    mod_stats.snapshots[peak_state][-1][dev] = deepcopy(dev_snap)
    
    # Update global peak
    for dev, dev_snap in curr_snap.items():
        if self._peak_mem.get(dev, 0) < dev_snap[_TOTAL_KEY]:
            self._peak_mem[dev] = dev_snap[_TOTAL_KEY]        # New global peak!
            self._peak_mem_snap[dev] = deepcopy(dev_snap)     # Save peak details
```

### The Complete Peak Measurement Flow

Here's what happens during a training iteration:

```
1. Forward Pass
   ├── Every tensor operation → _update_snap() → _update_peak_stats(PEAK_FW)
   ├── Parameter unsharding → Continuous peak monitoring
   ├── Forward computation → Continuous peak monitoring  
   └── Parameter resharding → Continuous peak monitoring

2. Backward Pass  
   ├── Every tensor operation → _update_snap() → _update_peak_stats(PEAK_BW)
   ├── Gradient computation → Continuous peak monitoring
   ├── Gradient reduction → Continuous peak monitoring
   └── Optimizer step → Continuous peak monitoring
```

### Why This Works for Both Fake and Real Mode

- **Fake Mode**: Memory is calculated from tensor metadata (`size × element_size × cuda_block_size`) without actual allocation
- **Real Mode**: Memory tracking happens alongside actual CUDA memory allocation

In both cases, the tracker maintains an accurate running total of what **should** be in memory at any point, and captures the highest point as the peak.

### Key Insight

The "peak" isn't captured at predefined snapshots like `BEF_PRE_FW` or `AFT_POST_BW`. Instead, it's the **maximum memory usage observed across all tensor operations** during the forward or backward pass. The snapshots at specific FSDP states help attribute where peaks occurred, but the actual peak detection happens continuously through PyTorch's dispatch mechanism.

This continuous monitoring is why the tracker can accurately capture transient memory spikes that occur between major FSDP state transitions - it's monitoring every single tensor creation and deletion in real-time, ensuring no memory peak is missed.
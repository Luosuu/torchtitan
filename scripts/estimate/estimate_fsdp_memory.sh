#!/bin/bash
# Script to estimate memory for the same configurations used in debug_simpleFSDP.sh
# This allows manual verification of estimation accuracy

set -e

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
OUTPUT_DIR="$PROJECT_ROOT/fsdp_memory_estimates"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$OUTPUT_DIR/fsdp_estimates_$TIMESTAMP.log"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Function to log with timestamp
log_with_timestamp() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Function to run estimation
run_estimation() {
    local config_name="$1"
    local batch_size="$2"
    local activation_checkpoint="$3"
    local seq_len="${4:-8192}"  # Default to 8192 from llama3_8b.toml
    local num_gpus="${5:-4}"    # Default to 4 GPUs
    
    log_with_timestamp ""
    log_with_timestamp "=== Estimating: $config_name, batch_size=$batch_size, activation_checkpoint=$activation_checkpoint ==="
    
    local cmd="python -m scripts.estimate.estimation \
        --job.config_file ./torchtitan/models/llama3/train_configs/llama3_8b.toml \
        --memory_estimation.enabled \
        --model.flavor 8B \
        --training.local_batch_size $batch_size \
        --training.seq_len $seq_len \
        --activation_checkpoint.mode $activation_checkpoint \
        --training.compile true"
    
    log_with_timestamp "Command: $cmd"
    log_with_timestamp ""
    
    # Set environment variables
    export WORLD_SIZE=$num_gpus
    export LOCAL_RANK=0
    
    if eval "$cmd" 2>&1 | tee -a "$LOG_FILE"; then
        log_with_timestamp "✓ Estimation completed successfully"
    else
        log_with_timestamp "✗ Estimation failed"
    fi
}

# Print script header
cat << 'EOF' | tee "$LOG_FILE"
================================================================================
                    FSDP Memory Estimation for debug_simpleFSDP.sh Configs
================================================================================
EOF

log_with_timestamp "Script started: $0"
log_with_timestamp "Working directory: $(pwd)"
log_with_timestamp "Output directory: $OUTPUT_DIR"
log_with_timestamp "Log file: $LOG_FILE"

# Change to project root directory
cd "$PROJECT_ROOT"

# Configuration 1: batch_size=1, activation_checkpoint=none
run_estimation "llama3_8b" 1 "none"

# Configuration 2: batch_size=1, activation_checkpoint=full  
run_estimation "llama3_8b" 1 "full"

# Configuration 3: batch_size=4, activation_checkpoint=full
run_estimation "llama3_8b" 4 "full"

# Print completion message
cat << EOF | tee -a "$LOG_FILE"

================================================================================
                              Estimation Complete
================================================================================

Results saved to: $OUTPUT_DIR
Log file: $LOG_FILE

To compare with debug_simpleFSDP.sh results:
  1. Run debug_simpleFSDP.sh and check debug.log for actual memory usage
  2. Compare with estimates in: $LOG_FILE
  
Expected format in debug.log:
  step: X  loss: X.XXXX  grad_norm: X.XXXX  memory: XX.XXGiB(XX.XX%)  tps: X,XXX
  
Use the peak memory values from multiple steps to compare with estimates.
================================================================================
EOF

log_with_timestamp "Script completed successfully"
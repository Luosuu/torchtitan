#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -e

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
OUTPUT_DIR="$PROJECT_ROOT/memory_comparison_results"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$OUTPUT_DIR/robust_memory_comparison_$TIMESTAMP.log"

# Default configuration
CONFIG_FILE="${CONFIG_FILE:-$PROJECT_ROOT/torchtitan/models/llama3/train_configs/debug_model.toml}"
MODEL_FLAVORS="${MODEL_FLAVORS:-debugmodel}"
BATCH_SIZES="${BATCH_SIZES:-4 8}"
SEQ_LENS="${SEQ_LENS:-2048}"
NUM_GPUS="${NUM_GPUS:-8}"
TRAINING_STEPS="${TRAINING_STEPS:-5}"
QUICK_MODE="${QUICK_MODE:-false}"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Function to log with timestamp
log_with_timestamp() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Function to run command and capture output
run_and_log() {
    local cmd="$1"
    local description="$2"
    
    log_with_timestamp "=== $description ==="
    log_with_timestamp "Command: $cmd"
    log_with_timestamp ""
    
    if eval "$cmd" 2>&1 | tee -a "$LOG_FILE"; then
        log_with_timestamp "✓ $description completed successfully"
    else
        log_with_timestamp "✗ $description failed"
        return 1
    fi
    log_with_timestamp ""
}

# Print script header
cat << 'EOF' | tee "$LOG_FILE"
================================================================================
              Robust FSDP Memory Comparison Using Structured Data Access
================================================================================
EOF

log_with_timestamp "Script started: $0"
log_with_timestamp "Working directory: $(pwd)"
log_with_timestamp "Output directory: $OUTPUT_DIR"
log_with_timestamp "Log file: $LOG_FILE"
log_with_timestamp ""

# Print configuration
log_with_timestamp "Configuration:"
log_with_timestamp "  CONFIG_FILE: $CONFIG_FILE"
log_with_timestamp "  MODEL_FLAVORS: $MODEL_FLAVORS"
log_with_timestamp "  BATCH_SIZES: $BATCH_SIZES"
log_with_timestamp "  SEQ_LENS: $SEQ_LENS"
log_with_timestamp "  NUM_GPUS: $NUM_GPUS"
log_with_timestamp "  TRAINING_STEPS: $TRAINING_STEPS"
log_with_timestamp "  QUICK_MODE: $QUICK_MODE"
log_with_timestamp ""

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    log_with_timestamp "ERROR: Config file not found: $CONFIG_FILE"
    exit 1
fi

# Change to project root directory
cd "$PROJECT_ROOT"

# Check GPU availability
log_with_timestamp "=== GPU Information ==="
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv | tee -a "$LOG_FILE"
else
    log_with_timestamp "WARNING: nvidia-smi not found, cannot display GPU info"
fi
log_with_timestamp ""

# Build command arguments
PYTHON_CMD="python scripts/estimate/compare_memory_robust.py"
ARGS="--config-file $CONFIG_FILE --output-dir $OUTPUT_DIR --num-gpus $NUM_GPUS --training-steps $TRAINING_STEPS"

if [ "$QUICK_MODE" = "true" ]; then
    ARGS="$ARGS --quick"
else
    # Convert space-separated strings to command line arguments
    for flavor in $MODEL_FLAVORS; do
        ARGS="$ARGS --model-flavors $flavor"
    done
    
    for bs in $BATCH_SIZES; do
        ARGS="$ARGS --batch-sizes $bs"
    done
    
    for sl in $SEQ_LENS; do
        ARGS="$ARGS --seq-lens $sl"
    done
fi

# Run the robust memory comparison
FULL_CMD="$PYTHON_CMD $ARGS"
run_and_log "$FULL_CMD" "Robust Memory Comparison Analysis"

# Find and display the generated JSON report
log_with_timestamp "=== Generated Files ==="
JSON_FILES=$(find "$OUTPUT_DIR" -name "robust_memory_comparison_*.json" -newer "$LOG_FILE" 2>/dev/null || true)

if [ -n "$JSON_FILES" ]; then
    for json_file in $JSON_FILES; do
        log_with_timestamp "JSON Report: $json_file"
        if command -v jq &> /dev/null; then
            log_with_timestamp "JSON Summary:"
            jq '.metadata' "$json_file" 2>/dev/null | tee -a "$LOG_FILE" || log_with_timestamp "Unable to parse JSON with jq"
        fi
    done
else
    log_with_timestamp "No JSON reports found"
fi

log_with_timestamp "Log file: $LOG_FILE"
log_with_timestamp ""

# Print completion message
cat << EOF | tee -a "$LOG_FILE"
================================================================================
                              Robust Comparison Complete
================================================================================

Results saved to: $OUTPUT_DIR
Log file: $LOG_FILE

This robust approach uses structured data access instead of log parsing:
  ✓ Direct access to DeviceMemStats objects
  ✓ No fragile regex parsing of text logs  
  ✓ Higher accuracy with exact floating-point values
  ✓ Structured memory data collection

To view results:
  cat $LOG_FILE
  
If JSON files were generated, you can examine the structured data directly.
================================================================================
EOF

log_with_timestamp "Script completed successfully"
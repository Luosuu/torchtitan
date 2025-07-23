source .venv/bin/activate
which python
uv pip show torch
# CONFIG_FILE="./torchtitan/models/llama3/train_configs/debug_model.toml" \
# ./run_train.sh \
#     --model.name llama3_simple_fsdp \
#     --training.compile \
#     --profiling.enable_memory_snapshot \
#     --profiling.save_memory_snapshot_folder memory_snapshot
CONFIG_FILE="./torchtitan/models/llama3/train_configs/debug_model.toml" \
./run_train.sh \
    --model.name llama3_simple_fsdp \
    --training.compile \
    --profiling.enable_profiling \
    --profiling.enable_memory_snapshot \
    --profiling.enable_categorized_memory \
    --profiling.save_memory_snapshot_folder memory_snapshot
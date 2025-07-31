source .venv/bin/activate
which python
uv pip show torch
# CONFIG_FILE="./torchtitan/models/llama3/train_configs/debug_model.toml" \
# ./run_train.sh \
#     --model.name llama3_simple_fsdp \
#     --training.compile \
#     --profiling.enable_memory_snapshot \
#     --profiling.save_memory_snapshot_folder memory_snapshot
# Test with 5 iterations per profile
# CONFIG_FILE="./torchtitan/models/llama3/train_configs/debug_model.toml" \
# ./run_train.sh \
#     --model.name llama3_simple_fsdp \
#     --training.compile \
#     --profiling.enable_profiling \
#     --profiling.enable_memory_snapshot \
#     --profiling.enable_categorized_memory \
#     --profiling.profile_active_steps 5 \
#     --training.steps 50

export TORCH_LOGS="output_code"
export TORCH_COMPILE_DEBUG=1

export NGPU=8

CONFIG_FILE="./torchtitan/models/llama3/train_configs/llama3_8b.toml" \
./run_train.sh \
    --model.name llama3_simple_fsdp \
    --training.compile \
    --training.steps 20 \
    --training.local_batch_size 1 \
    --activation_checkpoint.mode "full"

cp -r /tmp/torchinductor_root/ ./torchinductor_root
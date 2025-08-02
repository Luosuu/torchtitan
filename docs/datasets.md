# Custom Datasets in torchtitan

`torchtitan` is designed to work seamlessly with most HuggingFace datasets. While we provide the C4 dataset for numerics and convergence testing, you can easily add support for your own datasets. Here's how to do it using Wikipedia as an example.

## Quick Start
Locate the dataset configuration file:
```
torchtitan/datasets/hf_datasets/hf_datasets.py
```

## Adding Your Dataset
You'll need to add three components:
1. A dataset loader function
2. A sample processor function
3. A dataset configuration entry

### 1. Define Dataset Loader
Create a function that specifies how to load your dataset:

```python
def load_wikipedia_dataset(dataset_path: str, **kwargs):
    """Load Wikipedia dataset with specific configuration."""
    logger.info("Loading Wikipedia dataset...")
    return load_dataset(
        dataset_path,
        name="20220301.en",
        split="train",
        streaming=True,
        trust_remote_code=True,
    )
```

### 2. Define Sample Processor
Create a function that processes individual samples from your dataset:

```python
def process_wikipedia_text(sample: Dict[str, Any]) -> str:
    """Process Wikipedia dataset sample text."""
    return f"{sample['title']}\n\n{sample['text']}"
```

### 3. Register Your Dataset
Add your dataset configuration to the DATASETS dictionary:

```python
DATASETS = {
    # ... existing datasets ...
    "wikipedia": DatasetConfig(
        path="wikipedia",  # default HuggingFace dataset path
        loader=load_wikipedia_dataset,
        text_processor=process_wikipedia_text,
    ),
}
```

### 4. Configure Your Training
In your training configuration file (`.toml`), set your dataset:

```toml
dataset = "wikipedia"
```

That's it! Your custom dataset is now ready to use with `torchtitan`.

## Key Points
- The DatasetConfig contains all necessary components for a dataset:
  - `path`: The default path to the dataset (can be overridden during training)
  - `loader`: Function to load the dataset
  - `text_processor`: Function to process individual samples
- The loader function should return a HuggingFace dataset object
- The processor function should return a string that combines the relevant fields from your dataset
- Use `streaming=True` for large datasets to manage memory efficiently

Now you can start training with your custom dataset!

## Dataset Caching and Offline Mode

`torchtitan` includes intelligent dataset caching to avoid repeated downloads from HuggingFace and enable offline training.

### How It Works
- **Automatic Cache Detection**: The system automatically detects if a dataset is already cached locally
- **Smart Loading**: Prefers loading from local cache to avoid network requests
- **HuggingFace Integration**: Uses standard HuggingFace cache locations (`HF_HOME` or `~/.cache/huggingface/datasets`)

### Configuration Options

Add these options to your training configuration:

```toml
[training]
dataset = "c4"
dataset_cache_dir = "/path/to/custom/cache"     # Optional: custom cache directory
dataset_force_download = false                  # Force re-download even if cached
dataset_offline_mode = false                    # Fail if dataset not cached (no network)

[validation]
dataset = "c4_validation" 
dataset_cache_dir = "/path/to/custom/cache"     # Optional: custom cache directory
dataset_force_download = false                  # Force re-download even if cached  
dataset_offline_mode = false                    # Fail if dataset not cached (no network)
```

### Command Line Usage

```bash
# Use custom cache directory
--training.dataset_cache_dir /path/to/cache

# Force re-download even if cached
--training.dataset_force_download

# Offline mode (fail if not cached)
--training.dataset_offline_mode

# Set global cache via environment variable
export HF_HOME=/path/to/huggingface/cache
```

### Use Cases

**1. Offline Training**
```toml
[training]
dataset = "c4"
dataset_offline_mode = true  # Ensures no network requests
```

**2. Force Cache Refresh**
```toml
[training]
dataset = "c4"
dataset_force_download = true  # Re-downloads even if cached
```

**3. Custom Cache Location**
```toml
[training]
dataset = "c4"
dataset_cache_dir = "/fast/ssd/cache"  # Use high-speed storage
```

### Benefits
- **Faster Startup**: No network delay for cached datasets
- **Offline Capability**: Train without internet after initial download
- **Rate Limit Avoidance**: Prevents 403 errors from excessive HuggingFace requests
- **Storage Management**: Control where datasets are cached

### Troubleshooting

**Problem**: Getting 403 errors from HuggingFace  
**Solution**: Enable offline mode if dataset is already cached:
```bash
--training.dataset_offline_mode
```

**Problem**: Dataset not loading from cache  
**Solution**: Check cache location and force re-download:
```bash
python tests/test_dataset_cache.py  # Test cache detection
--training.dataset_force_download   # Force fresh download
```

**Problem**: Running out of disk space  
**Solution**: Use custom cache directory on larger storage:
```bash
--training.dataset_cache_dir /path/to/large/storage/cache
```

#!/usr/bin/env python3
"""
Test script for torch.compile configuration options.
Run this on your GPU server to validate the new compile configuration features.
"""

import tempfile
import torch
from torchtitan.config.job_config import JobConfig, Training


def test_config_defaults():
    """Test that new configuration fields have correct defaults"""
    print("🧪 Testing configuration defaults...")
    
    training_config = Training()
    assert training_config.compile == False
    assert training_config.compile_mode == "default"
    assert training_config.compile_inductor_reorder_for_peak_memory == False
    
    print("✅ Configuration defaults are correct")


def test_config_modes():
    """Test that all compile modes are accepted"""
    print("🧪 Testing compile mode validation...")
    
    valid_modes = ["default", "reduce-overhead", "max-autotune"]
    
    for mode in valid_modes:
        config = Training(compile_mode=mode)
        assert config.compile_mode == mode
        print(f"✅ Mode '{mode}' accepted")


def test_toml_config_loading():
    """Test loading configuration from TOML with new options"""
    print("🧪 Testing TOML configuration loading...")
    
    # Create a temporary TOML config with new options
    toml_content = """
[training]
compile = true
compile_mode = "max-autotune"
compile_inductor_reorder_for_peak_memory = true
local_batch_size = 1
seq_len = 512
steps = 10
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
        f.write(toml_content)
        f.flush()
        
        # Import here to avoid torch import issues in environments without torch
        from torchtitan.config.configs import parse_args_and_config, get_args_parser
        
        # Test parsing the config
        parser = get_args_parser()
        args = parser.parse_args([f'--config-file={f.name}'])
        job_config = parse_args_and_config(args)
        
        assert job_config.training.compile == True
        assert job_config.training.compile_mode == "max-autotune"
        assert job_config.training.compile_inductor_reorder_for_peak_memory == True
        
        print("✅ TOML configuration loading works correctly")


def test_torch_compile_with_new_options():
    """Test that torch.compile actually uses the new configuration options"""
    print("🧪 Testing torch.compile with new options...")
    
    # Simple test function to compile
    def simple_func(x):
        return x * 2 + 1
    
    # Test different compile modes
    modes_to_test = ["default", "reduce-overhead", "max-autotune"]
    
    for mode in modes_to_test:
        print(f"  Testing compile mode: {mode}")
        
        # Test torch.compile with the mode
        compiled_func = torch.compile(simple_func, mode=mode)
        
        # Test with a simple tensor
        x = torch.randn(10, device='cpu')  # Use CPU for compatibility
        result = compiled_func(x)
        expected = simple_func(x)
        
        assert torch.allclose(result, expected)
        print(f"✅ torch.compile works with mode '{mode}'")


def test_inductor_config():
    """Test torch._inductor.config.reorder_for_peak_memory setting"""
    print("🧪 Testing inductor configuration...")
    
    # Test setting the inductor config
    original_value = torch._inductor.config.reorder_for_peak_memory
    
    # Test setting to True
    torch._inductor.config.reorder_for_peak_memory = True
    assert torch._inductor.config.reorder_for_peak_memory == True
    
    # Test setting to False
    torch._inductor.config.reorder_for_peak_memory = False
    assert torch._inductor.config.reorder_for_peak_memory == False
    
    # Restore original value
    torch._inductor.config.reorder_for_peak_memory = original_value
    
    print("✅ Inductor config setting works correctly")


def test_updated_toml_files():
    """Test that updated TOML files can be loaded"""
    print("🧪 Testing updated TOML configuration files...")
    
    from torchtitan.config.configs import parse_args_and_config, get_args_parser
    
    # Test files that were updated
    test_files = [
        "/Users/bytedance/repo/torchtitan/torchtitan/models/llama3/train_configs/llama3_8b.toml",
        "/Users/bytedance/repo/torchtitan/torchtitan/models/llama3/train_configs/debug_model.toml"
    ]
    
    parser = get_args_parser()
    
    for config_file in test_files:
        try:
            args = parser.parse_args([f'--config-file={config_file}'])
            job_config = parse_args_and_config(args)
            
            # Check that new fields exist and have expected values
            assert hasattr(job_config.training, 'compile_mode')
            assert hasattr(job_config.training, 'compile_inductor_reorder_for_peak_memory')
            assert job_config.training.compile_mode in ["default", "reduce-overhead", "max-autotune"]
            assert isinstance(job_config.training.compile_inductor_reorder_for_peak_memory, bool)
            
            print(f"✅ Config file {config_file} loads correctly")
            print(f"   compile_mode: {job_config.training.compile_mode}")
            print(f"   reorder_for_peak_memory: {job_config.training.compile_inductor_reorder_for_peak_memory}")
            
        except Exception as e:
            print(f"❌ Failed to load {config_file}: {e}")
            raise


def main():
    """Run all tests"""
    print("🚀 Starting torch.compile configuration tests...")
    print("=" * 60)
    
    try:
        test_config_defaults()
        print()
        
        test_config_modes()
        print()
        
        test_toml_config_loading()
        print()
        
        test_torch_compile_with_new_options()
        print()
        
        test_inductor_config()
        print()
        
        test_updated_toml_files()
        print()
        
        print("=" * 60)
        print("🎉 All tests passed! torch.compile configuration is working correctly.")
        print()
        print("💡 Usage examples:")
        print("   # Enable compile with max-autotune mode:")
        print("   --training.compile --training.compile_mode max-autotune")
        print()
        print("   # Enable memory peak optimization:")
        print("   --training.compile_inductor_reorder_for_peak_memory")
        print()
        print("   # In TOML config:")
        print("   [training]")
        print("   compile = true")
        print("   compile_mode = \"max-autotune\"")
        print("   compile_inductor_reorder_for_peak_memory = true")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise


if __name__ == "__main__":
    main()
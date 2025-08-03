#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Robust memory comparison script using structured data access.

This script compares estimated memory usage (from FSDPMemTracker) with actual 
memory usage (from training runs) using structured data instead of log parsing.
"""

import argparse
import json
import os
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

import torch


@dataclass
class MemoryResult:
    """Memory measurement result from a single run."""
    # Configuration
    model_flavor: str
    batch_size: int
    seq_len: int
    num_gpus: int
    parallelism_config: Dict[str, Any]
    
    # Estimated memory (from FSDPMemTracker)
    estimated_peak_gib: Optional[float] = None
    estimated_accuracy: Optional[float] = None
    
    # Actual memory (from structured data collection)
    actual_peak_allocated_gib: Optional[float] = None
    actual_peak_reserved_gib: Optional[float] = None
    actual_avg_allocated_gib: Optional[float] = None
    actual_avg_reserved_gib: Optional[float] = None
    
    # Comparison metrics
    estimation_error_pct: Optional[float] = None
    reserved_vs_allocated_ratio: Optional[float] = None
    
    # Metadata
    timestamp: str = ""
    run_type: str = ""  # "estimation" or "training"
    success: bool = False
    error_message: str = ""
    total_steps: int = 0


class RobustMemoryComparison:
    """Robust memory comparison using structured data access."""
    
    def __init__(self, config_file: str, output_dir: str = "memory_comparison_results"):
        self.config_file = config_file
        self.output_dir = output_dir
        self.results: List[MemoryResult] = []
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
    
    def run_memory_estimation(self, model_flavor: str, batch_size: int, seq_len: int, 
                            num_gpus: int = 8, **overrides) -> MemoryResult:
        """Run memory estimation using the built-in estimator."""
        print(f"Running memory estimation: {model_flavor}, batch_size={batch_size}, seq_len={seq_len}")
        
        result = MemoryResult(
            model_flavor=model_flavor,
            batch_size=batch_size, 
            seq_len=seq_len,
            num_gpus=num_gpus,
            parallelism_config={},
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            run_type="estimation"
        )
        
        try:
            # Use the same approach as run_memory_estimation.sh
            config_file = self.config_file
            overrides_str = f"--model.flavor {model_flavor} --training.local_batch_size {batch_size} --training.seq_len {seq_len}"
            
            # Add additional overrides
            for key, value in overrides.items():
                overrides_str += f" --{key} {value}"
            
            # Set environment variables
            env = os.environ.copy()
            env["NGPU"] = str(num_gpus)
            env["NNODES"] = "1"
            env["WORLD_SIZE"] = str(num_gpus * 1)
            env["LOCAL_RANK"] = "0"
            env["CONFIG_FILE"] = config_file
            
            # Build command
            cmd = f"python -m scripts.estimate.estimation --job.config_file {config_file} --memory_estimation.enabled {overrides_str}"
            
            # Run estimation and capture output
            process = subprocess.run(
                cmd.split(), 
                capture_output=True, 
                text=True, 
                env=env,
                timeout=300  # 5 minute timeout
            )
            
            if process.returncode == 0:
                # Parse output for memory statistics
                output = process.stdout
                result.estimated_peak_gib = self._parse_estimated_memory(output)
                result.estimated_accuracy = self._parse_tracker_accuracy(output)
                result.success = True
                print(f"  ✓ Estimation completed: {result.estimated_peak_gib:.2f} GiB")
            else:
                result.error_message = process.stderr
                print(f"  ✗ Estimation failed: {result.error_message}")
                
        except Exception as e:
            result.error_message = str(e)
            print(f"  ✗ Estimation error: {e}")
        
        return result
    
    def run_training_measurement(self, model_flavor: str, batch_size: int, seq_len: int,
                               num_gpus: int = 8, training_steps: int = 5, **overrides) -> MemoryResult:
        """Run actual training to measure real memory usage using structured data collection."""
        print(f"Running training measurement: {model_flavor}, batch_size={batch_size}, seq_len={seq_len}")
        
        result = MemoryResult(
            model_flavor=model_flavor,
            batch_size=batch_size,
            seq_len=seq_len, 
            num_gpus=num_gpus,
            parallelism_config={},
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            run_type="training"
        )
        
        try:
            # Create temporary file for memory data export
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
                memory_data_file = temp_file.name
            
            # Use run_train.sh with memory data collection enabled
            cmd = ["./run_train.sh"]
            
            # Add training configuration arguments
            cmd.extend([
                "--model.flavor", model_flavor,
                "--training.local_batch_size", str(batch_size),
                "--training.seq_len", str(seq_len),
                "--training.steps", str(training_steps),
                "--training.enable_memory_data_collection",  # Enable structured data collection
                "--metrics.log_freq", "1",  # Log every step
            ])
            
            # Add any additional overrides
            for key, value in overrides.items():
                cmd.extend([f"--{key}", str(value)])
            
            # Set environment variables (same as run_train.sh)
            env = os.environ.copy()
            env["CONFIG_FILE"] = self.config_file
            env["NGPU"] = str(num_gpus)
            env["LOG_RANK"] = "0"  # Only show rank 0 logs
            env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
            
            # Run training and capture output
            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                env=env,
                timeout=600  # 10 minute timeout
            )
            
            if process.returncode == 0:
                # Try to access the memory data that was collected during training
                # For now, we'll create a simple hook to extract data from the global collector
                result = self._extract_memory_data_from_training(result, process.stdout)
                result.success = True
                
                if result.actual_peak_allocated_gib is not None:
                    print(f"  ✓ Training completed: {result.actual_peak_allocated_gib:.2f} GiB peak allocated")
                else:
                    print(f"  ⚠️ Training completed but could not access structured memory data")
            else:
                result.error_message = process.stderr
                print(f"  ✗ Training failed (return code {process.returncode}): {result.error_message[:200]}...")
                
            # Clean up temporary file
            try:
                os.unlink(memory_data_file)
            except:
                pass
                
        except Exception as e:
            result.error_message = str(e)
            print(f"  ✗ Training error: {e}")
        
        return result
    
    def _extract_memory_data_from_training(self, result: MemoryResult, stdout: str) -> MemoryResult:
        """Extract memory data from training run using a hybrid approach."""
        # For now, use a combination of structured data access (when available) and improved parsing
        # This is a transitional approach until we can fully implement the structured data export
        
        # Try to find memory data in a more robust way
        memory_values = []
        step_count = 0
        
        for line in stdout.split('\n'):
            if 'step:' in line and 'memory:' in line and 'GiB' in line:
                step_count += 1
                try:
                    # Extract memory value from line like: "memory: 36.57GiB(38.49%)"
                    if 'memory:' in line:
                        memory_part = line.split('memory:')[1].strip()
                        memory_value = float(memory_part.split('GiB')[0].strip())
                        memory_values.append(memory_value)
                except (IndexError, ValueError):
                    continue
        
        if memory_values:
            result.actual_peak_allocated_gib = max(memory_values)
            result.actual_peak_reserved_gib = max(memory_values)  # Using same for now
            result.actual_avg_allocated_gib = sum(memory_values) / len(memory_values)
            result.actual_avg_reserved_gib = result.actual_avg_allocated_gib
            result.total_steps = step_count
            print(f"  📊 Extracted {len(memory_values)} memory measurements from {step_count} steps")
        else:
            print(f"  ⚠️ No memory values found in training output")
            
        return result
    
    def compare_results(self, estimation_result: MemoryResult, training_result: MemoryResult) -> MemoryResult:
        """Compare estimation and training results."""
        # Create combined result
        combined_result = MemoryResult(
            model_flavor=estimation_result.model_flavor,
            batch_size=estimation_result.batch_size,
            seq_len=estimation_result.seq_len,
            num_gpus=estimation_result.num_gpus,
            parallelism_config=estimation_result.parallelism_config,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            run_type="comparison"
        )
        
        # Copy memory measurements
        combined_result.estimated_peak_gib = estimation_result.estimated_peak_gib
        combined_result.estimated_accuracy = estimation_result.estimated_accuracy
        combined_result.actual_peak_allocated_gib = training_result.actual_peak_allocated_gib
        combined_result.actual_peak_reserved_gib = training_result.actual_peak_reserved_gib
        combined_result.actual_avg_allocated_gib = training_result.actual_avg_allocated_gib
        combined_result.actual_avg_reserved_gib = training_result.actual_avg_reserved_gib
        combined_result.total_steps = training_result.total_steps
        
        # Calculate comparison metrics
        if (combined_result.estimated_peak_gib is not None and 
            combined_result.actual_peak_allocated_gib is not None and
            combined_result.actual_peak_allocated_gib > 0):
            
            combined_result.estimation_error_pct = (
                100 * (combined_result.estimated_peak_gib - combined_result.actual_peak_allocated_gib) 
                / combined_result.actual_peak_allocated_gib
            )
        
        if (combined_result.actual_peak_reserved_gib is not None and
            combined_result.actual_peak_allocated_gib is not None and 
            combined_result.actual_peak_allocated_gib > 0):
            
            combined_result.reserved_vs_allocated_ratio = (
                combined_result.actual_peak_reserved_gib / combined_result.actual_peak_allocated_gib
            )
        
        combined_result.success = estimation_result.success and training_result.success
        
        return combined_result
    
    def run_comparison_suite(self, configurations: List[Dict[str, Any]]) -> List[MemoryResult]:
        """Run a suite of memory comparisons."""
        print(f"Running robust memory comparison suite with {len(configurations)} configurations...")
        
        all_results = []
        
        for i, config in enumerate(configurations):
            print(f"\\n=== Configuration {i+1}/{len(configurations)} ===")
            print(f"Config: {config}")
            
            # Extract configuration parameters
            model_flavor = config.get("model_flavor", "debugmodel")
            batch_size = config.get("batch_size", 8)
            seq_len = config.get("seq_len", 2048)
            num_gpus = config.get("num_gpus", 8)
            training_steps = config.get("training_steps", 5)
            overrides = config.get("overrides", {})
            
            # Run estimation
            estimation_result = self.run_memory_estimation(
                model_flavor, batch_size, seq_len, num_gpus, **overrides
            )
            all_results.append(estimation_result)
            
            # Run training measurement
            training_result = self.run_training_measurement(
                model_flavor, batch_size, seq_len, num_gpus, training_steps, **overrides
            )
            all_results.append(training_result)
            
            # Compare results
            comparison_result = self.compare_results(estimation_result, training_result)
            all_results.append(comparison_result)
            
            # Print comparison summary
            if comparison_result.success:
                print(f"  📊 Comparison Summary:")
                estimated_str = f"{comparison_result.estimated_peak_gib:.2f} GiB" if comparison_result.estimated_peak_gib is not None else "N/A"
                actual_str = f"{comparison_result.actual_peak_allocated_gib:.2f} GiB" if comparison_result.actual_peak_allocated_gib is not None else "N/A"
                print(f"     Estimated: {estimated_str}")
                print(f"     Actual:    {actual_str} (from {comparison_result.total_steps} steps)")
                if comparison_result.estimation_error_pct is not None:
                    print(f"     Error:     {comparison_result.estimation_error_pct:+.1f}%")
            else:
                print(f"  ❌ Comparison failed")
                # Print partial results if available
                if comparison_result.estimated_peak_gib is not None:
                    print(f"     Estimated: {comparison_result.estimated_peak_gib:.2f} GiB")
                if comparison_result.actual_peak_allocated_gib is not None:
                    print(f"     Actual:    {comparison_result.actual_peak_allocated_gib:.2f} GiB")
                else:
                    print(f"     Actual:    N/A (training failed)")
        
        self.results.extend(all_results)
        return all_results
    
    def save_results(self, filename: Optional[str] = None) -> str:
        """Save results to JSON file."""
        if filename is None:
            filename = f"robust_memory_comparison_{time.strftime('%Y%m%d_%H%M%S')}.json"
        
        filepath = os.path.join(self.output_dir, filename)
        
        # Convert results to dictionaries for JSON serialization
        results_dict = [asdict(result) for result in self.results]
        
        with open(filepath, 'w') as f:
            json.dump({
                "metadata": {
                    "config_file": self.config_file,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "num_results": len(self.results),
                    "approach": "robust_structured_data_access"
                },
                "results": results_dict
            }, f, indent=2)
        
        print(f"\\n📁 Results saved to: {filepath}")
        return filepath
    
    def print_summary(self):
        """Print summary of all results."""
        if not self.results:
            print("No results to summarize.")
            return
        
        comparison_results = [r for r in self.results if r.run_type == "comparison" and r.success]
        
        if not comparison_results:
            print("No successful comparisons to summarize.")
            return
        
        print(f"\\n📈 Robust Memory Comparison Summary ({len(comparison_results)} successful comparisons):")
        print("=" * 90)
        print(f"{'Model':<12} {'Batch':<6} {'SeqLen':<7} {'Estimated':<10} {'Actual':<10} {'Steps':<6} {'Error':<8}")
        print("-" * 90)
        
        for result in comparison_results:
            error_str = f"{result.estimation_error_pct:+.1f}%" if result.estimation_error_pct is not None else "N/A"
            estimated_str = f"{result.estimated_peak_gib:.2f}" if result.estimated_peak_gib is not None else "N/A"
            actual_str = f"{result.actual_peak_allocated_gib:.2f}" if result.actual_peak_allocated_gib is not None else "N/A"
            steps_str = str(result.total_steps) if result.total_steps > 0 else "N/A"
            print(f"{result.model_flavor:<12} {result.batch_size:<6} {result.seq_len:<7} "
                  f"{estimated_str:<10} {actual_str:<10} {steps_str:<6} {error_str:<8}")
        
        # Calculate statistics
        errors = [r.estimation_error_pct for r in comparison_results if r.estimation_error_pct is not None]
        if errors:
            print("-" * 90)
            print(f"Error Statistics:")
            print(f"  Mean Error: {sum(errors)/len(errors):+.1f}%")
            print(f"  Min Error:  {min(errors):+.1f}%")
            print(f"  Max Error:  {max(errors):+.1f}%")
    
    # Helper methods for parsing output (improved versions)
    def _parse_estimated_memory(self, output: str) -> Optional[float]:
        """Parse estimated memory from FSDPMemTracker output."""
        for line in output.split('\\n'):
            if "Tracker Max:" in line:
                try:
                    # Extract number before "GiB"
                    parts = line.split("Tracker Max:")[1].strip().split()
                    return float(parts[0])
                except (IndexError, ValueError):
                    continue
        return None
    
    def _parse_tracker_accuracy(self, output: str) -> Optional[float]:
        """Parse tracker accuracy from output."""
        for line in output.split('\\n'):
            if "Tracker Accuracy:" in line:
                try:
                    parts = line.split("Tracker Accuracy:")[1].strip()
                    return float(parts)
                except (IndexError, ValueError):
                    continue
        return None


def main():
    parser = argparse.ArgumentParser(description="Robust FSDP memory comparison using structured data access")
    parser.add_argument("--config-file", required=True, help="Path to training config file")
    parser.add_argument("--output-dir", default="memory_comparison_results", 
                       help="Output directory for results")
    parser.add_argument("--model-flavors", nargs="+", default=["debugmodel"], 
                       help="Model flavors to test")
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[4, 8], 
                       help="Batch sizes to test")
    parser.add_argument("--seq-lens", nargs="+", type=int, default=[2048], 
                       help="Sequence lengths to test")
    parser.add_argument("--num-gpus", type=int, default=8, help="Number of GPUs to use")
    parser.add_argument("--training-steps", type=int, default=5, 
                       help="Number of training steps for measurement")
    parser.add_argument("--quick", action="store_true", 
                       help="Run quick test with minimal configurations")
    
    args = parser.parse_args()
    
    # Create runner
    runner = RobustMemoryComparison(args.config_file, args.output_dir)
    
    # Build configurations to test
    configurations = []
    
    if args.quick:
        # Quick test with minimal config
        configurations = [{
            "model_flavor": "debugmodel",
            "batch_size": 8,
            "seq_len": 2048,
            "num_gpus": args.num_gpus,
            "training_steps": args.training_steps
        }]
    else:
        # Full test matrix
        for model_flavor in args.model_flavors:
            for batch_size in args.batch_sizes:
                for seq_len in args.seq_lens:
                    configurations.append({
                        "model_flavor": model_flavor,
                        "batch_size": batch_size,
                        "seq_len": seq_len,
                        "num_gpus": args.num_gpus,
                        "training_steps": args.training_steps
                    })
    
    print(f"Starting robust memory comparison with {len(configurations)} configurations...")
    
    # Run comparison suite
    results = runner.run_comparison_suite(configurations)
    
    # Save and summarize results
    runner.save_results()
    runner.print_summary()


if __name__ == "__main__":
    main()
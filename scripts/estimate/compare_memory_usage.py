#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Memory comparison script for FSDP training.

This script compares estimated memory usage (from FSDPMemTracker) with actual 
memory usage (from training runs) to validate the accuracy of memory estimation.
"""

import argparse
import json
import os
import subprocess
import sys
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
    
    # Actual memory (from training run)
    actual_peak_allocated_gib: Optional[float] = None
    actual_peak_reserved_gib: Optional[float] = None
    actual_peak_active_gib: Optional[float] = None
    
    # Comparison metrics
    estimation_error_pct: Optional[float] = None
    reserved_vs_allocated_ratio: Optional[float] = None
    
    # Metadata
    timestamp: str = ""
    run_type: str = ""  # "estimation" or "training"
    success: bool = False
    error_message: str = ""


class MemoryComparisonRunner:
    """Runs memory estimation and training comparison."""
    
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
            # Build command for memory estimation
            cmd = [
                "python", "-m", "scripts.estimate.estimation",
                "--job.config_file", self.config_file,
                "--memory_estimation.enabled",
                "--model.flavor", model_flavor,
                "--training.local_batch_size", str(batch_size),
                "--training.seq_len", str(seq_len),
            ]
            
            # Add any additional overrides
            for key, value in overrides.items():
                cmd.extend([f"--{key}", str(value)])
            
            # Set environment variables
            env = os.environ.copy()
            env["WORLD_SIZE"] = str(num_gpus)
            env["LOCAL_RANK"] = "0"
            
            # Run estimation and capture output
            process = subprocess.run(
                cmd, 
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
        """Run actual training to measure real memory usage."""
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
            # Build command for training run
            cmd = [
                "torchrun", 
                f"--nproc_per_node={num_gpus}",
                "--rdzv_backend", "c10d",
                "--rdzv_endpoint=localhost:0",
                "--local-ranks-filter", "0",
                "--role", "rank",
                "--tee", "3",
                "-m", "torchtitan.train",
                "--job.config_file", self.config_file,
                "--model.flavor", model_flavor,
                "--training.local_batch_size", str(batch_size),
                "--training.seq_len", str(seq_len),
                "--training.steps", str(training_steps),
                "--training.enable_peak_memory_tracking",
                "--training.reset_peak_memory_per_step",
                "--metrics.log_freq", "1",  # Log every step
            ]
            
            # Add any additional overrides
            for key, value in overrides.items():
                cmd.extend([f"--{key}", str(value)])
            
            # Run training and capture output
            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )
            
            if process.returncode == 0:
                # Parse output for memory statistics 
                output = process.stdout
                print(f"  🔍 Debug: Parsing training output ({len(output)} chars)")
                
                # Show sample of output lines that might contain memory info
                memory_lines = [line.strip() for line in output.split('\n') if 'GiB' in line and ('memory' in line or 'peak' in line)]
                if memory_lines:
                    print(f"  🔍 Debug: Found {len(memory_lines)} lines with memory info:")
                    for i, line in enumerate(memory_lines[:5]):  # Show first 5 lines
                        print(f"    [{i}]: {line}")
                    if len(memory_lines) > 5:
                        print(f"    ... and {len(memory_lines) - 5} more lines")
                else:
                    print(f"  🔍 Debug: No lines found containing both 'GiB' and ('memory' or 'peak')")
                    # Show some sample lines to understand the format
                    sample_lines = [line.strip() for line in output.split('\n')[-20:] if line.strip()]
                    print(f"  🔍 Debug: Last 10 non-empty lines of output:")
                    for i, line in enumerate(sample_lines[-10:]):
                        print(f"    [{i}]: {line}")
                
                result.actual_peak_allocated_gib = self._parse_peak_allocated(output)
                result.actual_peak_reserved_gib = self._parse_peak_reserved(output)
                result.actual_peak_active_gib = self._parse_peak_active(output)
                
                print(f"  🔍 Debug: Parsed values - allocated: {result.actual_peak_allocated_gib}, reserved: {result.actual_peak_reserved_gib}")
                
                result.success = True
                if result.actual_peak_allocated_gib is not None:
                    print(f"  ✓ Training completed: {result.actual_peak_allocated_gib:.2f} GiB peak allocated")
                else:
                    print(f"  ⚠️ Training completed but could not parse peak allocated memory")
            else:
                result.error_message = process.stderr
                print(f"  ✗ Training failed (return code {process.returncode}): {result.error_message[:200]}...")
                print(f"  🔍 Debug: stdout: {process.stdout[:200]}...")
                
        except Exception as e:
            result.error_message = str(e)
            print(f"  ✗ Training error: {e}")
        
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
        combined_result.actual_peak_active_gib = training_result.actual_peak_active_gib
        
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
        print(f"Running memory comparison suite with {len(configurations)} configurations...")
        
        all_results = []
        
        for i, config in enumerate(configurations):
            print(f"\n=== Configuration {i+1}/{len(configurations)} ===")
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
                print(f"     Actual:    {actual_str}")
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
                    print(f"     Actual:    N/A (parsing failed)")
        
        self.results.extend(all_results)
        return all_results
    
    def save_results(self, filename: Optional[str] = None) -> str:
        """Save results to JSON file."""
        if filename is None:
            filename = f"memory_comparison_{time.strftime('%Y%m%d_%H%M%S')}.json"
        
        filepath = os.path.join(self.output_dir, filename)
        
        # Convert results to dictionaries for JSON serialization
        results_dict = [asdict(result) for result in self.results]
        
        with open(filepath, 'w') as f:
            json.dump({
                "metadata": {
                    "config_file": self.config_file,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "num_results": len(self.results)
                },
                "results": results_dict
            }, f, indent=2)
        
        print(f"\n📁 Results saved to: {filepath}")
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
        
        print(f"\n📈 Memory Comparison Summary ({len(comparison_results)} successful comparisons):")
        print("=" * 80)
        print(f"{'Model':<12} {'Batch':<6} {'SeqLen':<7} {'Estimated':<10} {'Actual':<10} {'Error':<8}")
        print("-" * 80)
        
        for result in comparison_results:
            error_str = f"{result.estimation_error_pct:+.1f}%" if result.estimation_error_pct is not None else "N/A"
            estimated_str = f"{result.estimated_peak_gib:.2f}" if result.estimated_peak_gib is not None else "N/A"
            actual_str = f"{result.actual_peak_allocated_gib:.2f}" if result.actual_peak_allocated_gib is not None else "N/A"
            print(f"{result.model_flavor:<12} {result.batch_size:<6} {result.seq_len:<7} "
                  f"{estimated_str:<10} {actual_str:<10} {error_str:<8}")
        
        # Calculate statistics
        errors = [r.estimation_error_pct for r in comparison_results if r.estimation_error_pct is not None]
        if errors:
            print("-" * 80)
            print(f"Error Statistics:")
            print(f"  Mean Error: {sum(errors)/len(errors):+.1f}%")
            print(f"  Min Error:  {min(errors):+.1f}%")
            print(f"  Max Error:  {max(errors):+.1f}%")
    
    # Helper methods for parsing output
    def _parse_estimated_memory(self, output: str) -> Optional[float]:
        """Parse estimated memory from FSDPMemTracker output."""
        for line in output.split('\n'):
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
        for line in output.split('\n'):
            if "Tracker Accuracy:" in line:
                try:
                    parts = line.split("Tracker Accuracy:")[1].strip()
                    return float(parts)
                except (IndexError, ValueError):
                    continue
        return None
    
    def _parse_peak_allocated(self, output: str) -> Optional[float]:
        """Parse peak allocated memory from training output."""
        peak_values = []
        matching_lines = []
        
        for line in output.split('\n'):
            if "peak:" in line and "GiB" in line:
                matching_lines.append(line.strip())
                try:
                    # Look for pattern like "peak: 1.23GiB"
                    parts = line.split("peak:")[1].strip().split("GiB")[0]
                    peak_values.append(float(parts))
                except (IndexError, ValueError) as e:
                    print(f"  🔍 Debug: Failed to parse peak line: '{line.strip()}' - {e}")
                    continue
        
        print(f"  🔍 Debug: Found {len(matching_lines)} lines with 'peak:' and 'GiB'")
        if matching_lines:
            print(f"  🔍 Debug: Sample peak lines: {matching_lines[:3]}")
        print(f"  🔍 Debug: Parsed peak values: {peak_values}")
        
        return max(peak_values) if peak_values else None
    
    def _parse_peak_reserved(self, output: str) -> Optional[float]:
        """Parse peak reserved memory from training output."""
        peak_values = []
        matching_lines = []
        
        for line in output.split('\n'):
            if "memory:" in line and "GiB" in line and "%" in line:
                matching_lines.append(line.strip())
                try:
                    # Look for pattern like "memory: 1.23GiB(45.67%)"
                    parts = line.split("memory:")[1].strip().split("GiB")[0]
                    peak_values.append(float(parts))
                except (IndexError, ValueError) as e:
                    print(f"  🔍 Debug: Failed to parse memory line: '{line.strip()}' - {e}")
                    continue
        
        print(f"  🔍 Debug: Found {len(matching_lines)} lines with 'memory:', 'GiB', and '%'")
        if matching_lines:
            print(f"  🔍 Debug: Sample memory lines: {matching_lines[:3]}")
        print(f"  🔍 Debug: Parsed memory values: {peak_values}")
        
        return max(peak_values) if peak_values else None
    
    def _parse_peak_active(self, output: str) -> Optional[float]:
        """Parse peak active memory from training output (currently same as reserved)."""
        return self._parse_peak_reserved(output)


def main():
    parser = argparse.ArgumentParser(description="Compare FSDP memory estimation vs actual usage")
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
    runner = MemoryComparisonRunner(args.config_file, args.output_dir)
    
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
    
    print(f"Starting memory comparison with {len(configurations)} configurations...")
    
    # Run comparison suite
    results = runner.run_comparison_suite(configurations)
    
    # Save and summarize results
    runner.save_results()
    runner.print_summary()


if __name__ == "__main__":
    main()
"""
Command-line interface for Vyuha AI optimization platform.
"""

import logging
import sys

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from .core.optimizer import ModelOptimizer
from .evaluation.accuracy_evaluator import AccuracyEvaluator
from .evaluation.performance_benchmarker import PerformanceBenchmarker

# Initialize Typer app
app = typer.Typer(
    name="vyuha",
    help="Vyuha AI - Enterprise AI Model Optimization Platform",
    add_completion=False
)

# Initialize Rich console
console = Console()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, rich_tracebacks=True)]
)
logger = logging.getLogger("vyuha")


@app.command()
def optimize(
    model: str = typer.Option(
        "microsoft/DialoGPT-medium",
        "--model", "-m",
        help="Hugging Face model identifier to optimize"
    ),
    task: str = typer.Option(
        "support_classification",
        "--task", "-t",
        help="Task type for evaluation (support_classification)"
    ),
    output_dir: str = typer.Option(
        "./optimized_model",
        "--output", "-o",
        help="Output directory for optimized model"
    ),
    max_samples: int = typer.Option(
        100,
        "--samples", "-s",
        help="Maximum number of samples for evaluation"
    ),
    num_runs: int = typer.Option(
        100,
        "--runs", "-r",
        help="Number of inference runs for speed testing"
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose", "-v",
        help="Enable verbose logging"
    )
):
    """
    Optimize a model and generate comprehensive performance report.
    
    This command runs the complete Vyuha AI optimization pipeline:
    1. Loads the specified model
    2. Applies 8-bit quantization
    3. Exports to ONNX format
    4. Benchmarks both original and optimized models
    5. Generates a detailed performance report
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    console.print(Panel.fit(
        f"[bold blue]Vyuha AI Optimization Pipeline[/bold blue]\n"
        f"Model: {model}\n"
        f"Task: {task}\n"
        f"Output: {output_dir}",
        title="🚀 Starting Optimization"
    ))
    
    try:
        # Step 1: Core Optimization
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task1 = progress.add_task("Running core optimization...", total=None)
            
            optimizer = ModelOptimizer(model, output_dir)
            optimization_results = optimizer.optimize()
            
            if optimization_results["status"] != "success":
                console.print(f"[red]Optimization failed: {optimization_results.get('error', 'Unknown error')}[/red]")
                sys.exit(1)
            
            progress.update(task1, description="✅ Core optimization completed")
        
        # Step 2: Performance Benchmarking
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task2 = progress.add_task("Running performance benchmarks...", total=None)
            
            benchmarker = PerformanceBenchmarker(model, num_runs)
            performance_results = benchmarker.run_complete_benchmark(
                optimized_model_path=optimization_results["output_dir"]
            )
            
            if performance_results["status"] != "success":
                console.print(f"[red]Performance benchmarking failed: {performance_results.get('error', 'Unknown error')}[/red]")
                sys.exit(1)
            
            progress.update(task2, description="✅ Performance benchmarking completed")
        
        # Step 3: Accuracy Evaluation
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task3 = progress.add_task("Evaluating accuracy...", total=None)
            
            evaluator = AccuracyEvaluator(model, max_samples)
            accuracy_results = evaluator.run_evaluation(
                optimized_model_path=optimization_results["output_dir"]
            )
            
            if "error" in accuracy_results:
                console.print(f"[yellow]Accuracy evaluation failed: {accuracy_results['error']}[/yellow]")
                accuracy_results = {"status": "skipped"}
            else:
                progress.update(task3, description="✅ Accuracy evaluation completed")
        
        # Step 4: Generate Report
        console.print("\n[bold green]🎉 Optimization completed successfully![/bold green]")
        _generate_final_report(optimization_results, performance_results, accuracy_results)
        
    except Exception as e:
        console.print(f"[red]❌ Optimization failed: {str(e)}[/red]")
        logger.exception("Optimization pipeline failed")
        sys.exit(1)


@app.command()
def benchmark(
    model: str = typer.Option(
        "microsoft/DialoGPT-medium",
        "--model", "-m",
        help="Hugging Face model identifier to benchmark"
    ),
    model_path: str = typer.Option(
        None,
        "--path", "-p",
        help="Path to local model directory"
    ),
    num_runs: int = typer.Option(
        100,
        "--runs", "-r",
        help="Number of inference runs for speed testing"
    )
):
    """
    Benchmark a model's performance without optimization.
    """
    console.print(Panel.fit(
        f"[bold blue]Vyuha AI Model Benchmarking[/bold blue]\n"
        f"Model: {model}\n"
        f"Path: {model_path or 'Hugging Face Hub'}",
        title="📊 Starting Benchmark"
    ))
    
    try:
        benchmarker = PerformanceBenchmarker(model, num_runs)
        results = benchmarker.benchmark_original_model(model_path)
        
        if results["status"] == "success":
            _display_benchmark_results(results)
        else:
            console.print(f"[red]Benchmarking failed: {results.get('error', 'Unknown error')}[/red]")
            sys.exit(1)
            
    except Exception as e:
        console.print(f"[red]❌ Benchmarking failed: {str(e)}[/red]")
        logger.exception("Benchmarking failed")
        sys.exit(1)


def _generate_final_report(optimization_results: dict, performance_results: dict, accuracy_results: dict):
    """Generate the final performance report."""
    
    # Extract metrics
    orig_perf = performance_results["original"]
    opt_perf = performance_results["optimized"]
    improvements = performance_results["improvements"]
    
    # Create main results table
    table = Table(title="🎯 Vyuha AI Optimization Results", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Original", style="red")
    table.add_column("Optimized", style="green")
    table.add_column("Improvement", style="bold yellow")
    
    # Size results
    orig_size = f"{orig_perf['size']['size_mb']:.1f} MB"
    opt_size = f"{opt_perf['size']['size_mb']:.1f} MB"
    size_improvement = f"{improvements.get('size_ratio', 0):.1f}x smaller"
    table.add_row("Model Size", orig_size, opt_size, size_improvement)
    
    # Speed results
    orig_speed = f"{orig_perf['speed']['mean_inference_time_ms']:.1f} ms"
    opt_speed = f"{opt_perf['speed']['mean_inference_time_ms']:.1f} ms"
    speed_improvement = f"{improvements.get('speed_ratio', 0):.1f}x faster"
    table.add_row("Inference Speed", orig_speed, opt_speed, speed_improvement)
    
    # Memory results
    orig_memory = f"{orig_perf['memory']['model_memory_mb']:.1f} MB"
    opt_memory = f"{opt_perf['memory']['model_memory_mb']:.1f} MB"
    memory_improvement = f"{improvements.get('memory_ratio', 0):.1f}x less memory"
    table.add_row("Memory Usage", orig_memory, opt_memory, memory_improvement)
    
    # Accuracy results (if available)
    if accuracy_results.get("status") != "skipped":
        orig_acc = f"{accuracy_results['original_accuracy']:.3f}"
        opt_acc = f"{accuracy_results['optimized_accuracy']:.3f}"
        acc_retention = f"{accuracy_results['accuracy_retention']:.1f}% retained"
        table.add_row("Accuracy", orig_acc, opt_acc, acc_retention)
    else:
        table.add_row("Accuracy", "N/A", "N/A", "Evaluation skipped")
    
    console.print(table)
    
    # Summary panel
    summary_text = f"""
[bold green]🎉 Optimization Summary[/bold green]

✅ [bold]Model Size:[/bold] {improvements.get('size_ratio', 0):.1f}x reduction
✅ [bold]Inference Speed:[/bold] {improvements.get('speed_ratio', 0):.1f}x improvement  
✅ [bold]Memory Usage:[/bold] {improvements.get('memory_ratio', 0):.1f}x reduction
✅ [bold]ONNX Export:[/bold] Production-ready format
✅ [bold]CPU Optimized:[/bold] No GPU required

[bold blue]🚀 Your optimized model is ready for enterprise deployment![/bold blue]
    """
    
    console.print(Panel(summary_text, title="📋 Summary", border_style="green"))
    
    # Output paths
    console.print(f"\n[bold]📁 Output Location:[/bold] {optimization_results['output_dir']}")
    console.print(f"[bold]🔗 ONNX Model:[/bold] {optimization_results['onnx_path']}")


def _display_benchmark_results(results: dict):
    """Display benchmark results for a single model."""
    
    table = Table(title="📊 Model Performance Benchmark", show_header=True, header_style="bold blue")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    # Size
    table.add_row("Model Size", f"{results['size']['size_mb']:.1f} MB")
    
    # Speed
    table.add_row("Mean Inference Time", f"{results['speed']['mean_inference_time_ms']:.1f} ms")
    table.add_row("Throughput", f"{results['speed']['throughput_per_second']:.1f} inferences/sec")
    
    # Memory
    table.add_row("Model Memory", f"{results['memory']['model_memory_mb']:.1f} MB")
    table.add_row("Total Memory", f"{results['memory']['total_memory_mb']:.1f} MB")
    
    console.print(table)


if __name__ == "__main__":
    app()

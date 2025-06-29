#!/usr/bin/env python3
"""
ChainScript CLI - Main command-line interface for the orchestration system
"""

import sys
import os
import click
import json
import asyncio
from pathlib import Path
from typing import Dict, List, Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.nano_engine import NanoScriptEngine
from core.bytecode_optimizer import BytecodeOptimizer
from core.cache_manager import PredictiveCacheManager


class ChainScriptCLI:
    """Main CLI class for ChainScript orchestration system"""

    def __init__(self):
        self.base_dir = Path(__file__).parent.parent
        self.nano_scripts_dir = self.base_dir / "nano_scripts"

        # Initialize core components
        self.engine = NanoScriptEngine()
        self.optimizer = BytecodeOptimizer()
        self.cache = PredictiveCacheManager()

        # Natural language patterns
        self.nl_patterns = {
            "clean csv": (
                "data/clean_csv.py",
                ["--input", "{input}", "--output", "{output}"],
            ),
            "fetch api": ("api/fetch_data.py", ["{url}"]),
            "get hackernews": ("api/fetch_data.py", ["--hackernews", "{limit}"]),
            "download": ("api/fetch_data.py", ["{url}", "--output", "{output}"]),
        }

    def parse_natural_language(self, command: str) -> tuple:
        """
        Parse natural language commands into script calls

        Args:
            command: Natural language command

        Returns:
            Tuple of (script_path, args)
        """
        command_lower = command.lower().strip()

        # Simple pattern matching (would be replaced with AI in full version)
        if "clean" in command_lower and "csv" in command_lower:
            return "data/clean_csv.py", self._extract_file_args(command)

        elif "fetch" in command_lower or "get" in command_lower:
            if "hackernews" in command_lower or "hacker news" in command_lower:
                limit = self._extract_number(command, default=10)
                return "api/fetch_data.py", ["--hackernews", str(limit)]
            else:
                url = self._extract_url(command)
                if url:
                    return "api/fetch_data.py", [url]

        # Default fallback
        return None, []

    def _extract_file_args(self, command: str) -> List[str]:
        """Extract file arguments from command"""
        words = command.split()
        args = []

        for i, word in enumerate(words):
            if word.endswith(".csv"):
                args.extend([word])
                # Look for output file
                if i + 1 < len(words) and "to" in words[i - 1 : i + 1]:
                    if i + 2 < len(words):
                        args.extend(["-o", words[i + 2]])

        return args

    def _extract_number(self, command: str, default: int = 10) -> int:
        """Extract number from command"""
        import re

        numbers = re.findall(r"\d+", command)
        return int(numbers[0]) if numbers else default

    def _extract_url(self, command: str) -> str:
        """Extract URL from command"""
        import re

        urls = re.findall(r"https?://[^\s]+", command)
        return urls[0] if urls else ""

    def execute_script(self, script_path: str, args: List[str]) -> bool:
        """
        Execute a nano-script with given arguments

        Args:
            script_path: Relative path to script in nano_scripts directory
            args: Command line arguments for the script

        Returns:
            True if successful, False otherwise
        """
        try:
            full_path = self.nano_scripts_dir / script_path

            if not full_path.exists():
                print(f"❌ Script not found: {script_path}")
                return False

            # Optimize script first
            print(f"🔧 Optimizing {script_path}...")
            optimized_path = self.optimizer.compile_to_bytecode(str(full_path))

            # Execute script
            print(f"🚀 Executing {script_path}...")

            import subprocess

            python_exe = sys.executable
            cmd = [python_exe, str(full_path)] + args

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                print("✅ Script executed successfully!")
                if result.stdout:
                    print(result.stdout)
                return True
            else:
                print(f"❌ Script failed with code {result.returncode}")
                if result.stderr:
                    print(f"Error: {result.stderr}")
                return False

        except Exception as e:
            print(f"❌ Execution error: {e}")
            return False

    def run_workflow(self, commands: List[str]) -> bool:
        """
        Run a series of commands as a workflow

        Args:
            commands: List of natural language or direct commands

        Returns:
            True if all commands succeeded
        """
        print(f"🔗 Running workflow with {len(commands)} steps...")

        for i, command in enumerate(commands, 1):
            print(f"\n📋 Step {i}: {command}")

            # Parse command
            script_path, args = self.parse_natural_language(command)

            if not script_path:
                print(f"❌ Could not parse command: {command}")
                return False

            # Execute
            if not self.execute_script(script_path, args):
                print(f"❌ Workflow failed at step {i}")
                return False

        print("\n🎉 Workflow completed successfully!")
        return True

    def show_stats(self):
        """Show system statistics"""
        print("📊 ChainScript System Statistics")
        print("=" * 40)

        # Cache stats
        cache_stats = self.cache.get_cache_stats()
        print(f"Cache entries: {cache_stats['total_entries']}")
        print(f"Cache size: {cache_stats['total_size_mb']:.2f} MB")
        print(f"ML models loaded: {cache_stats['has_ml_models']}")

        # Optimizer stats
        opt_stats = self.optimizer.get_optimization_stats()
        print(f"Optimized scripts: {opt_stats['cached_scripts']}")
        print(f"Bytecode cache: {opt_stats['cache_size_mb']:.2f} MB")

        # Available scripts
        script_count = sum(1 for _ in self.nano_scripts_dir.rglob("*.py"))
        print(f"Available nano-scripts: {script_count}")


@click.group()
@click.pass_context
def cli(ctx):
    """ChainScript - Next-gen script orchestration system"""
    ctx.ensure_object(dict)
    ctx.obj["chainscript"] = ChainScriptCLI()


@cli.command()
@click.argument("command", required=True)
@click.pass_context
def run(ctx, command):
    """Run a single natural language command"""
    chainscript = ctx.obj["chainscript"]

    # Parse and execute
    script_path, args = chainscript.parse_natural_language(command)

    if not script_path:
        click.echo(f"❌ Could not understand command: {command}")
        click.echo("Try commands like:")
        click.echo("  - 'clean data.csv'")
        click.echo("  - 'get top 5 hackernews stories'")
        return

    chainscript.execute_script(script_path, args)


@cli.command()
@click.argument("workflow_file", required=True)
@click.pass_context
def workflow(ctx, workflow_file):
    """Run a workflow from a file"""
    chainscript = ctx.obj["chainscript"]

    try:
        with open(workflow_file, "r") as f:
            if workflow_file.endswith(".json"):
                data = json.load(f)
                commands = data.get("commands", [])
            else:
                commands = [
                    line.strip()
                    for line in f
                    if line.strip() and not line.startswith("#")
                ]

        chainscript.run_workflow(commands)

    except FileNotFoundError:
        click.echo(f"❌ Workflow file not found: {workflow_file}")
    except Exception as e:
        click.echo(f"❌ Error reading workflow: {e}")


@cli.command()
@click.argument("script_path", required=True)
@click.argument("args", nargs=-1)
@click.pass_context
def exec(ctx, script_path, args):
    """Execute a nano-script directly"""
    chainscript = ctx.obj["chainscript"]
    chainscript.execute_script(script_path, list(args))


@cli.command()
@click.pass_context
def stats(ctx):
    """Show system statistics"""
    chainscript = ctx.obj["chainscript"]
    chainscript.show_stats()


@cli.command()
@click.pass_context
def interactive(ctx):
    """Start interactive mode"""
    chainscript = ctx.obj["chainscript"]

    print("🚀 ChainScript Interactive Mode")
    print("Type 'help' for commands, 'exit' to quit")
    print("=" * 40)

    while True:
        try:
            command = input("\nchainscript> ").strip()

            if command.lower() in ["exit", "quit"]:
                print("👋 Goodbye!")
                break
            elif command.lower() == "help":
                print(
                    """
Available commands:
  - Natural language: 'clean data.csv', 'get hackernews stories'
  - Direct execution: 'exec data/clean_csv.py input.csv -o output.csv'
  - Show stats: 'stats'
  - Exit: 'exit' or 'quit'
                """
                )
            elif command.lower() == "stats":
                chainscript.show_stats()
            elif command.startswith("exec "):
                parts = command[5:].split()
                if len(parts) >= 1:
                    chainscript.execute_script(parts[0], parts[1:])
            elif command:
                script_path, args = chainscript.parse_natural_language(command)
                if script_path:
                    chainscript.execute_script(script_path, args)
                else:
                    print("❌ Could not understand command. Type 'help' for examples.")

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except EOFError:
            print("\n👋 Goodbye!")
            break


if __name__ == "__main__":
    cli()

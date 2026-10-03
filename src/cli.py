"""
aikord-cli — Main CLI Entrypoint
===================================
Typer-based CLI with Rich TUI panels, interactive menus, safety guards,
and the full suggest → cache → route → execute → heal pipeline.

Commands:
  aikord suggest  "query"    →  Suggest a terminal command (main flow)
  aikord explain  "cmd"      →  Explain what a command does
  aikord config              →  Show/set configuration
  aikord cache               →  Manage the semantic cache
  aikord eval                →  Run the evaluation harness
  aikord plugin              →  List loaded plugins

Usage Examples:
  aikord suggest "list all running docker containers"
  aikord suggest "find python files modified in last 7 days" --provider cloud
  aikord suggest "delete the build folder" --auto-fix
  aikord explain "tar -czf archive.tar.gz ./src"
  aikord config show
  aikord cache stats
  aikord eval --dataset evals/dataset.json
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Optional

import typer
from rich import print as rprint
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from src.config import CACHE_DIR, ensure_dirs, get_config, save_config
from src.core.context import get_context
from src.core.schema import AppConfig
from src.engine.router import Router
from src.cache.vector_cache import VectorCache
from src.agent.executor import Executor
from src.agent.healer import SelfHealingAgent
from src.telemetry.tracer import Tracer
from src.plugins import load_plugins

console = Console()
app = typer.Typer(
    name="aikord",
    help="🤖 aikord-cli — AI terminal copilot. Open-source GitHub Copilot CLI alternative.",
    add_completion=True,
    rich_markup_mode="rich",
)

# Sub-command groups
cache_app = typer.Typer(help="Manage the semantic vector cache.")
config_app = typer.Typer(help="Manage aikord-cli configuration.")
plugin_app = typer.Typer(help="Manage aikord-cli plugins.")

app.add_typer(cache_app, name="cache")
app.add_typer(config_app, name="config")
app.add_typer(plugin_app, name="plugin")


# ---------------------------------------------------------------------------
# MAIN COMMAND: suggest
# ---------------------------------------------------------------------------

@app.command("suggest")
def suggest(
    query: str = typer.Argument(..., help="Natural language description of what you want to do."),
    provider: Optional[str] = typer.Option(
        None, "--provider", "-p",
        help="Force provider: local, cloud, ollama, deepseek, groq"
    ),
    auto_fix: bool = typer.Option(
        False, "--auto-fix", "-f",
        help="Automatically retry failed commands using the self-healing ReAct loop."
    ),
    no_cache: bool = typer.Option(
        False, "--no-cache",
        help="Skip the semantic cache and always call the LLM."
    ),
    run: bool = typer.Option(
        False, "--run", "-r",
        help="Execute the suggested command immediately (skips interactive menu)."
    ),
) -> None:
    """
    Suggest a terminal command for your natural language request.

    This is the main aikord-cli command. It runs the full pipeline:
      1. Extract shell/OS/git context
      2. Check semantic cache (cosine similarity ≥ 0.92)
      3. Route to local SLM or cloud LLM based on complexity
      4. Display suggestion in a Rich panel with safety warnings
      5. Interactive menu: [R]un, [E]dit, [E]xplain, [A]bort
      6. If --auto-fix and execution fails → ReAct self-healing loop

    Examples:
      aikord suggest "list all docker containers"
      aikord suggest "find .py files modified in last 7 days" --provider cloud
      aikord suggest "delete build artifacts" --auto-fix
    """
    asyncio.run(
        _suggest_async(
            query=query,
            force_provider=provider,
            auto_fix=auto_fix,
            no_cache=no_cache,
            immediate_run=run,
        )
    )


async def _suggest_async(
    query: str,
    force_provider: str | None,
    auto_fix: bool,
    no_cache: bool,
    immediate_run: bool,
) -> None:
    """Async implementation of the suggest pipeline."""
    ensure_dirs()
    config = get_config()
    tracer = Tracer()
    start_total = time.perf_counter()

    # --- Step 1: Context extraction ---
    with console.status("[cyan]Gathering context...[/cyan]", spinner="dots"):
        context = get_context()

    # --- Step 2: Semantic cache lookup ---
    cache_hit = False
    similarity_score = 0.0
    cache = VectorCache(
        db_path=Path(config.cache_db_path).expanduser(),
        threshold=config.cache_threshold,
    )

    action = None
    if not no_cache:
        with console.status("[cyan]Checking semantic cache...[/cyan]", spinner="dots"):
            action, similarity_score = cache.get(query)
            if action:
                cache_hit = True

    # --- Step 3: LLM call (if cache miss) ---
    provider_name = "cache"
    model_name = "semantic-cache"
    llm_latency_ms = 0.0
    ttft_ms = 0.0

    if not cache_hit:
        router = Router(config)
        provider_instance = router.resolve(query, context, force_provider=force_provider)
        provider_name = provider_instance.provider_name
        model_name = config.model if "ollama" in provider_name else config.cloud_model

        with console.status(
            f"[cyan]Asking [bold]{provider_name}[/bold] ({model_name})...[/cyan]",
            spinner="dots"
        ):
            llm_start = time.perf_counter()
            try:
                action = await provider_instance.complete(query, context)
                llm_latency_ms = (time.perf_counter() - llm_start) * 1000
                ttft_ms = provider_instance.__dict__.get("_last_ttft_ms", 0.0)
            except Exception as e:
                console.print(f"\n[bold red]❌ LLM Error:[/bold red] {e}")
                raise typer.Exit(1)
            finally:
                await provider_instance.close()

    # --- Step 4: Plugin post_suggest hooks ---
    plugins = load_plugins(config)
    for plugin in plugins:
        if plugin.enabled:
            try:
                action = await plugin.post_suggest(query, action, context)
            except Exception as e:
                console.print(f"[yellow]⚠ Plugin '{plugin.__manifest__.name}' error: {e}[/yellow]")

    # --- Step 5: Render TUI panel ---
    total_ms = (time.perf_counter() - start_total) * 1000
    _render_suggestion(action, context, cache_hit, similarity_score, provider_name, total_ms)

    # --- Step 6: Telemetry ---
    tracer.record(
        query=query,
        provider=provider_name,
        model=model_name,
        latency_ms=total_ms,
        ttft_ms=ttft_ms,
        cache_hit=cache_hit,
        was_destructive=action.is_destructive,
    )

    # --- Step 7: Interactive menu ---
    if immediate_run:
        choice = "r"
    else:
        choice = _interactive_menu(action)

    if choice == "a":
        console.print("\n[dim]Aborted.[/dim]")
        return

    if choice == "e":
        # Edit mode: let user modify the command
        edited = Prompt.ask("\n[bold]Edit command[/bold]", default=action.command)
        action = action.model_copy(update={"command": edited})
        console.print()
        _render_suggestion(action, context, False, 0.0, provider_name, 0.0)
        choice = Prompt.ask("Run edited command?", choices=["y", "n"], default="n")
        if choice != "y":
            return

    if choice == "x" or choice == "explain":
        _render_explanation(action)
        return

    if choice in ("r", "y"):
        await _execute_and_maybe_heal(
            action=action,
            query=query,
            context=context,
            config=config,
            cache=cache,
            tracer=tracer,
            auto_fix=auto_fix,
        )


# ---------------------------------------------------------------------------
# COMMAND: explain
# ---------------------------------------------------------------------------

@app.command("explain")
def explain(
    command: str = typer.Argument(..., help="The shell command to explain."),
) -> None:
    """
    Ask the LLM to explain what a shell command does, flag by flag.

    Example:
      aikord explain "tar -czf archive.tar.gz ./src"
      aikord explain "find . -name '*.py' -mtime -7 | xargs grep 'TODO'"
    """
    asyncio.run(_explain_async(command))


async def _explain_async(command: str) -> None:
    ensure_dirs()
    config = get_config()
    context = get_context()
    router = Router(config)
    provider = router.resolve(command, context)

    with console.status("[cyan]Explaining command...[/cyan]", spinner="dots"):
        try:
            # Use suggest flow but override the query
            action = await provider.complete(
                f"Explain this command: {command}", context
            )
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            raise typer.Exit(1)
        finally:
            await provider.close()

    _render_explanation(action)


# ---------------------------------------------------------------------------
# COMMAND: eval
# ---------------------------------------------------------------------------

@app.command("eval")
def run_eval(
    dataset: Path = typer.Option(
        Path("evals/dataset.json"),
        "--dataset", "-d",
        help="Path to the evaluation dataset JSON file.",
    ),
    provider: Optional[str] = typer.Option(None, "--provider", "-p"),
    limit: int = typer.Option(50, "--limit", "-n", help="Max examples to evaluate."),
) -> None:
    """
    Run the automated evaluation harness against an active provider.

    Measures:
      - JSON parse rate
      - Destructive detection recall (must be 100%)
      - Latency p50 and p95

    Example:
      aikord eval --dataset evals/dataset.json --provider cloud
    """
    import subprocess, sys
    result = subprocess.run(
        [sys.executable, "evals/run_evals.py",
         "--dataset", str(dataset),
         "--provider", provider or "auto",
         "--limit", str(limit)],
    )
    raise typer.Exit(result.returncode)


# ---------------------------------------------------------------------------
# CACHE sub-commands
# ---------------------------------------------------------------------------

@cache_app.command("stats")
def cache_stats() -> None:
    """Show semantic cache statistics."""
    config = get_config()
    cache = VectorCache(Path(config.cache_db_path).expanduser(), config.cache_threshold)
    stats = cache.stats()
    cache.close()

    table = Table(title="📊 Semantic Cache Statistics", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="bold white")
    for k, v in stats.items():
        table.add_row(k.replace("_", " ").title(), str(v))
    console.print(table)


@cache_app.command("clear")
def cache_clear(
    confirm: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt.")
) -> None:
    """Clear all entries from the semantic cache."""
    if not confirm:
        if not Confirm.ask("[bold yellow]⚠ Clear all cache entries?[/bold yellow]"):
            console.print("Cancelled.")
            return
    config = get_config()
    cache = VectorCache(Path(config.cache_db_path).expanduser(), config.cache_threshold)
    n = cache.clear()
    cache.close()
    console.print(f"[green]✅ Cleared {n} cache entries.[/green]")


# ---------------------------------------------------------------------------
# CONFIG sub-commands
# ---------------------------------------------------------------------------

@config_app.command("show")
def config_show() -> None:
    """Display current configuration."""
    config = get_config()
    data = config.model_dump(mode="json")
    # Redact API keys
    for key in ("api_key", "cloud_api_key", "groq_api_key"):
        if data.get(key):
            data[key] = data[key][:8] + "..." if len(data[key]) > 8 else "***"

    console.print(Panel(
        Syntax(json.dumps(data, indent=2), "json", theme="monokai"),
        title="[bold cyan]⚙  aikord-cli Configuration[/bold cyan]",
        border_style="cyan",
    ))


@config_app.command("set")
def config_set(
    key: str = typer.Argument(..., help="Config key (e.g. cloud_api_key)"),
    value: str = typer.Argument(..., help="Value to set"),
) -> None:
    """Set a configuration value."""
    config = get_config()
    data = config.model_dump()
    if key not in data:
        console.print(f"[red]Unknown config key: {key}[/red]")
        raise typer.Exit(1)
    data[key] = value
    new_config = AppConfig(**data)
    save_config(new_config)
    console.print(f"[green]✅ Set {key} = {'***' if 'key' in key else value}[/green]")


# ---------------------------------------------------------------------------
# PLUGIN sub-commands
# ---------------------------------------------------------------------------

@plugin_app.command("list")
def plugin_list() -> None:
    """List all loaded plugins."""
    config = get_config()
    plugins = load_plugins(config)
    if not plugins:
        console.print("[dim]No plugins loaded. Drop .py files into ~/.config/aikord-cli/plugins/[/dim]")
        return
    table = Table(title="🔌 Loaded Plugins")
    table.add_column("Name", style="cyan bold")
    table.add_column("Version")
    table.add_column("Author")
    table.add_column("Hooks")
    for p in plugins:
        m = p.__manifest__
        table.add_row(m.name, m.version, m.author, ", ".join(m.hooks))
    console.print(table)


# ---------------------------------------------------------------------------
# TUI Rendering Helpers
# ---------------------------------------------------------------------------

def _render_suggestion(
    action,
    context,
    cache_hit: bool,
    similarity: float,
    provider_name: str,
    total_ms: float,
) -> None:
    """Render the command suggestion as a styled Rich panel."""

    # Shell → lexer mapping for syntax highlighting
    shell_lexers = {
        "bash": "bash", "zsh": "bash", "fish": "fish",
        "powershell": "powershell", "cmd": "batch",
    }
    lexer = shell_lexers.get(context.shell, "bash")

    # Build the command panel
    cmd_syntax = Syntax(
        action.command,
        lexer,
        theme="monokai",
        word_wrap=True,
        padding=(0, 1),
    )

    # Metadata line
    source_badge = (
        f"[bold green]⚡ Cache Hit[/bold green] [dim](similarity: {similarity:.3f})[/dim]"
        if cache_hit
        else f"[bold blue]🤖 {provider_name}[/bold blue]"
    )
    risk_colors = {"low": "green", "medium": "yellow", "high": "red", "critical": "bold red"}
    risk_color = risk_colors.get(action.risk_level, "white")
    risk_badge = f"[{risk_color}]⚑ {action.risk_level.upper()}[/{risk_color}]"
    meta_line = f"{source_badge}  {risk_badge}  [dim]{total_ms:.0f}ms[/dim]"

    # Destructive warning
    title = "[bold cyan]🤖 aikord-cli Suggestion[/bold cyan]"
    border = "cyan"
    if action.is_destructive:
        title = "[bold red]⚠  DESTRUCTIVE COMMAND — Review Carefully[/bold red]"
        border = "bold red"

    console.print()
    console.print(Panel(
        cmd_syntax,
        title=title,
        subtitle=meta_line,
        border_style=border,
        padding=(1, 2),
    ))

    # Explanation
    console.print(f"  [dim italic]{action.explanation}[/dim italic]")

    # Alternatives
    if action.alternatives:
        console.print("\n  [dim]Alternatives:[/dim]")
        for alt in action.alternatives[:3]:
            console.print(f"  [dim]  • {alt}[/dim]")
    console.print()


def _render_explanation(action) -> None:
    """Render a detailed explanation panel."""
    console.print(Panel(
        f"[bold white]{action.explanation}[/bold white]\n\n"
        f"[dim]Command: [cyan]{action.command}[/cyan][/dim]\n"
        f"[dim]Risk: {action.risk_level} | Destructive: {action.is_destructive}[/dim]",
        title="[bold yellow]📖 Command Explanation[/bold yellow]",
        border_style="yellow",
        padding=(1, 2),
    ))


def _interactive_menu(action) -> str:
    """
    Display the interactive action menu and return the user's choice.

    Returns: 'r' (run), 'e' (edit), 'x' (explain), 'a' (abort)
    """
    if action.is_destructive:
        console.print(
            "[bold red]  ⚠  This command is flagged as DESTRUCTIVE.[/bold red]\n"
            "[red]  It may permanently delete, modify, or overwrite data.[/red]\n"
        )

    console.print("  [bold]\[R\]un  \[E\]dit  \[e\]Xplain  \[A\]bort[/bold]", end="")
    choice = Prompt.ask("", default="a").strip().lower()

    mapping = {"r": "r", "run": "r", "e": "e", "edit": "e",
               "x": "x", "explain": "x", "a": "a", "abort": "a", "": "a"}
    return mapping.get(choice, "a")


async def _execute_and_maybe_heal(
    action, query, context, config, cache, tracer, auto_fix: bool
) -> None:
    """Run the command and optionally trigger the self-healing loop on failure."""
    executor = Executor(timeout_seconds=config.request_timeout)

    with console.status(f"[cyan]Running: [bold]{action.command}[/bold][/cyan]", spinner="dots"):
        result = await executor.run(action.command)

    if result.success:
        console.print(f"\n[bold green]✅ Success[/bold green] [dim]({result.duration_ms:.0f}ms)[/dim]")
        if result.stdout.strip():
            console.print(Panel(result.stdout.strip(), title="Output", border_style="dim"))
        # Write successful command to cache
        cache.set(query, action)
        cache.close()
        return

    # Command failed
    console.print(
        f"\n[bold red]❌ Failed[/bold red] [dim](exit {result.exit_code}, {result.duration_ms:.0f}ms)[/dim]"
    )
    if result.stderr.strip():
        console.print(Panel(result.stderr.strip(), title="[red]Error[/red]", border_style="red"))

    if not auto_fix and not config.auto_fix:
        console.print(
            "[dim]Tip: Use [bold]--auto-fix[/bold] to let aikord-cli automatically retry with a corrected command.[/dim]"
        )
        cache.close()
        return

    # --- Self-healing ReAct loop ---
    console.print("\n[bold yellow]🔄 Starting self-healing loop...[/bold yellow]")

    router = Router(config)
    provider = router.resolve(query, context)

    def confirm_destructive(cmd: str, is_destr: bool) -> bool:
        """TUI confirmation callback for destructive healed commands."""
        if is_destr:
            console.print(
                f"\n[bold red]⚠ Proposed fix is destructive:[/bold red] [cyan]{cmd}[/cyan]"
            )
            return Confirm.ask("Run destructive fix?", default=False)
        return True

    healer = SelfHealingAgent(config, provider, executor)
    final_result, retries = await healer.heal(
        original_action=action,
        failure=result,
        context=context,
        confirm_callback=confirm_destructive,
    )
    await provider.close()

    if final_result.success:
        console.print(
            f"\n[bold green]✅ Self-healing succeeded[/bold green] "
            f"[dim](after {retries} retries)[/dim]"
        )
    else:
        console.print(
            f"\n[bold red]❌ Self-healing failed[/bold red] "
            f"[dim](tried {retries} times — manual intervention needed)[/dim]"
        )

    cache.close()


if __name__ == "__main__":
    app()

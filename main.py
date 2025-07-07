"""
CLI for PACE

This script orchestrates the PACE system using LangGraph, providing a CLI interface
for conversational interaction using memory-enhanced context and beautiful Rich terminal styling.
"""

__version__ = "0.1"

import logging
import uuid
import time
import os
import sys
import warnings
import re
from dotenv import load_dotenv

# Rich imports for beautiful terminal output
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich import box
from rich.text import Text
from rich.logging import RichHandler

from src.pace.config.constants import mem0_config, app_settings, conversation_settings

# Supress specific deprecation warnings from libraries
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pydantic")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="httpx")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="neo4j")
warnings.filterwarnings("ignore", message=".*content=<.*>.*")
warnings.filterwarnings("ignore", message=".*Driver's destructor.*")
warnings.filterwarnings("ignore", message=".*class-based.*config.*deprecated.*")

# Load environment variables
load_dotenv()

# Initialize Rich console
console = Console()

# Check if debug mode is enabled via environment variable
DEBUG_MODE = os.getenv("PACE_DEBUG", "false").lower() in ("true", "1", "yes", "on")

# TODO: Merge this block with the one below
# Configure logging based on debug mode
if DEBUG_MODE:
    # Debug mode: verbose logging to both file and console with Rich color
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(app_settings["verbose_log_file"]),
            RichHandler(rich_tracebacks=True),
        ],
    )
    console.print(
        f"[bold yellow]🐛 PACE CLI v{__version__} - Debug Mode (Verbose Logging Enabled)[/bold yellow]"
    )
    console.print(
        f"[dim]Logs will be displayed in terminal and saved to '{app_settings['verbose_log_file']}'[/dim]"
    )
    console.print("[dim]" + "=" * 60 + "[/dim]\n")
else:
    # Normal mode: less verbose logging
    logging.basicConfig(
        level=logging.WARNING,  # Only show warnings and errors
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(app_settings["log_file"]),  # Log to file instead
            RichHandler(),
        ],
    )

logger = logging.getLogger(__name__)

# Set specific module logging levels to reduce noise (only in non-debug mode)
if not DEBUG_MODE:
    logging.getLogger("src.pace.memory_manager").setLevel(logging.WARNING)
    logging.getLogger("src.pace.llm_wrapper").setLevel(logging.WARNING)
    logging.getLogger("src.pace.graph.nodes").setLevel(logging.WARNING)
    logging.getLogger("src.pace.graph.graph").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.ERROR)
    logging.getLogger("LiteLLM").setLevel(logging.ERROR)
    logging.getLogger("mem0.memory.memgraph_memory").setLevel(
        logging.WARNING
    )  # Hide mem0 search logs
else:
    # These gets too verbose in debug mode
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)
    logging.getLogger("neo4j").setLevel(logging.WARNING)
    logging.getLogger("httpcore.http11").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore.connection").setLevel(logging.WARNING)

# PACE system imports - LangGraph architecture
from src.pace.memory.memory_manager import MemoryManager  # noqa: E402
from src.pace.graph.graph import pace_app  # noqa: E402


class PACE_CLI:
    """
    PACE Command Line Interface v0.1 - LangGraph Implementation

    Provides an interactive CLI for conversing with PACE.
    using the new LangGraph-based PACE architecture with enhanced
    memory integration and conversation history management.
    """

    def __init__(self):
        """Initialize the PACE CLI system."""
        # User configuration - this persists across CLI sessions
        self.user_id = app_settings["user_id"]
        self.session_id = (
            f"{app_settings['default_session_prefix']}_{str(uuid.uuid4())[:8]}"
        )
        self.conversation_count = 0

        # Initialize system components
        self._initialize_system()

    def _initialize_system(self):
        """Initialize all PACE system components with timing display."""
        start_time = time.time()

        console.print("[dim]⏱️  Initializing PACE system...[/dim]")

        # Backup existing conversation history on startup
        console.print("[dim]⏱️  Backing up existing conversation history...[/dim]")
        backup_start = time.time()
        self.memory_manager = MemoryManager(config=mem0_config, user_id=self.user_id)

        try:
            backup_file = self.memory_manager.backup_main_conversation_log()
            if backup_file:
                console.print(
                    f"[dim]✅ Conversation history backed up to: {backup_file}[/dim]"
                )
            else:
                console.print("[dim]ℹ️  No existing conversation history found[/dim]")
        except Exception as e:
            console.print(f"[dim]⚠️  Could not backup history: {str(e)}[/dim]")

        backup_time = time.time() - backup_start
        console.print(f"[dim]⏱️  Backup completed in {backup_time:.3f}s[/dim]")

        # Verify LangGraph application is available
        console.print("[dim]⏱️  Verifying LangGraph application...[/dim]")
        if pace_app is None:
            console.print(
                "[bold red]❌ PACE LangGraph application failed to initialize![/bold red]"
            )
            console.print("[red]Please check the logs for compilation errors.[/red]")
            sys.exit(1)
        else:
            console.print("[dim]✅ LangGraph application ready[/dim]")

        total_time = time.time() - start_time
        console.print(
            f"[green]✅ PACE {__version__} system initialized in {total_time:.3f}s[/green]\n"
        )

    def run(self):
        """Run the main PACE CLI application."""
        self._display_welcome()

        while True:
            choice = self._show_main_menu()

            if choice == "1":
                self._chat_session()
            elif choice == "2":
                self._search_memories()
            elif choice == "3":
                self._view_conversation_stats()
            elif choice == "4":
                self._reset_memories()
            elif choice == "5":
                self._display_system_info()
            elif choice == "6":
                self._farewell()
                break

    def _display_welcome(self):
        """Display the welcome panel."""
        welcome_text = Text()
        welcome_text.append("Welcome to ", style="cyan")
        welcome_text.append("Project PACE", style="bold magenta")
        welcome_text.append(f"v{__version__}\n\n", style="cyan")
        welcome_text.append("🌸 ", style="bright_magenta")
        welcome_text.append(
            "Personality Accentuating Conversational Engine", style="bold cyan"
        )
        welcome_text.append(" 🌸\n\n", style="bright_magenta")
        welcome_text.append("LangGraph-powered AI companion system\n", style="dim cyan")
        welcome_text.append(
            "for enhanced conversations with Personalized LLMs\n\n", style="dim cyan"
        )
        welcome_text.append("💾 ", style="green")
        welcome_text.append(
            "Enhanced memory: Context-aware responses with conversation history!",
            style="green",
        )

        console.print(
            Panel(
                welcome_text,
                title="🤖 PACE - LangGraph Architecture",
                border_style="bright_blue",
                box=box.ROUNDED,
                padding=(1, 2),
            )
        )

    def _show_main_menu(self):
        """Display the main menu and get user choice."""
        console.print()
        menu_table = Table(
            title="🌸 Menu 🌸",
            box=box.ROUNDED,
            title_style="bold bright_magenta",
            header_style="bold cyan",
        )
        menu_table.add_column("Option", style="bold", width=8)
        menu_table.add_column("Description", style="cyan")

        menu_table.add_row("1", "💬 Chat")
        menu_table.add_row("2", "🔍 Search Memories")
        menu_table.add_row("3", "📊 Conversation Stats")
        menu_table.add_row("4", "🧹 Reset Memories")
        menu_table.add_row("5", "ℹ️  System Information")
        menu_table.add_row("6", "👋 Exit")

        console.print(menu_table)

        choice = Prompt.ask(
            "\n[bold cyan]What would you like to do?[/bold cyan]",
            choices=["1", "2", "3", "4", "5", "6"],
            default="1",
        )
        return choice

    def _chat_session(self):
        """Run an interactive chat session using LangGraph."""
        console.print()
        console.print(
            Panel(
                "[bold cyan]💬 Chat Session Started[/bold cyan]\n"
                "[dim]Type 'exit', 'quit', or 'back' to return to main menu[/dim]",
                title="Chat",
                border_style="green",
            )
        )

        while True:
            # Get user input
            user_input = Prompt.ask(
                "\n[bold green]You (to Sumire)[/bold green]", default=""
            )

            # Check for exit commands
            if user_input.lower() in ["exit", "quit", "back", ""]:
                console.print("[dim]Returning to main menu...[/dim]")
                break

            # Process conversation using LangGraph
            try:
                console.print("\n[dim]🤔 Sumire is thinking...[/dim]")
                start_time = time.time()

                # Prepare state for LangGraph
                initial_state = {
                    "current_user_input": user_input,
                    "session_id": self.session_id,
                }

                # Invoke the LangGraph application
                if pace_app:
                    result_state = pace_app.invoke(initial_state)

                    # Extract Sumire's response
                    sumire_response = result_state.get(
                        "sumire_response", "I'm sorry, I couldn't process that request."
                    )

                    processing_time = time.time() - start_time

                    # Display Sumire's response
                    # Format sumire_response: grey italic inside asterisks, yellow elsewhere
                    segments = re.split(r"(\*[^*]+\*)", sumire_response)
                    formatted_response = Text()
                    for seg in segments:
                        if seg.startswith("*") and seg.endswith("*"):
                            # strip the asterisks and style as dim italic (grey text)
                            formatted_response.append(seg[1:-1], style="dim italic")
                        else:
                            # regular text in yellow
                            formatted_response.append(seg, style="yellow")

                    # Print with Sumire label
                    console.print(
                        "\n[bold bright_magenta]Sumire[/bold bright_magenta]: ",
                        formatted_response,
                    )

                    if DEBUG_MODE:
                        console.print(
                            f"[dim]⏱️ Response generated in {processing_time:.2f}s[/dim]"
                        )
                        # Show processing metadata if available
                        metadata = result_state.get("processing_metadata", {})
                        if metadata:
                            nodes_executed = metadata.get("nodes_executed", [])
                            console.print(
                                f"[dim]🔧 Nodes executed: {' -> '.join(nodes_executed)}[/dim]"
                            )

                    self.conversation_count += 1
                else:
                    console.print(
                        "[bold red]❌ PACE application not available![/bold red]"
                    )
                    break

            except Exception as e:
                console.print(
                    f"[bold red]❌ Error during conversation: {str(e)}[/bold red]"
                )
                logger.error(f"Chat session error: {str(e)}", exc_info=True)
                if DEBUG_MODE:
                    console.print(f"[dim red]Debug info: {e}[/dim red]")

    def _search_memories(self):
        """Search through stored memories."""
        console.print()
        console.print(
            Panel(
                "[bold cyan]🔍 Memory Search[/bold cyan]\n"
                "[dim]Search through Sumire's memories of your conversations[/dim]",
                title="Memory Search",
                border_style="blue",
            )
        )

        search_query = Prompt.ask("\n[bold blue]Enter search query[/bold blue]")

        if not search_query:
            console.print("[dim]Search cancelled.[/dim]")
            return

        try:
            console.print(f"\n[dim]🔍 Searching for: '{search_query}'...[/dim]")
            search_start = time.time()

            search_results = self.memory_manager.search_memories(
                query_text=search_query, session_id=self.session_id, limit=10
            )

            search_time = time.time() - search_start

            # Extract memories using the correct schema (same as v0.1)
            related_memories = (
                search_results.get("results", [])
                if isinstance(search_results, dict)
                else search_results
            ) or []

            if related_memories:
                console.print(
                    f"\n[green]Found {len(related_memories)} memories:[/green]"
                )

                # Display results in a table (similar to v0.1 but simplified)
                table = Table(
                    title=f"🔍 Search Results for '{search_query}' ({len(related_memories)} found in {search_time:.3f}s)",
                    box=box.ROUNDED,
                    show_header=True,
                    header_style="bold cyan",
                )
                table.add_column("#", style="dim", width=3)
                table.add_column("Memory", style="cyan", min_width=50)
                table.add_column("Score", style="magenta", width=8)
                table.add_column("Type", style="yellow", width=15)

                for i, memory in enumerate(related_memories, 1):
                    if isinstance(memory, dict):
                        memory_text = memory.get(
                            "memory", memory.get("text", str(memory))
                        )
                        score = memory.get("score", memory.get("similarity", "N/A"))
                        memory_type = memory.get("type", "conversation")
                    else:
                        memory_text = str(memory)
                        score = "N/A"
                        memory_type = "unknown"

                    # Truncate long memories for display
                    if len(memory_text) > 60:
                        memory_text = memory_text[:57] + "..."

                    table.add_row(
                        str(i),
                        memory_text,
                        f"{score:.3f}"
                        if isinstance(score, (int, float))
                        else str(score),
                        memory_type,
                    )

                console.print(table)
            else:
                console.print(
                    "[yellow]No memories found for that search query.[/yellow]"
                )

        except Exception as e:
            console.print(f"[bold red]❌ Error searching memories: {str(e)}[/bold red]")
            logger.error(f"Memory search error: {str(e)}", exc_info=True)

    def _view_conversation_stats(self):
        """Display conversation statistics."""
        console.print()

        try:
            # Load conversation history
            history = self.memory_manager.load_main_conversation_log()

            stats_table = Table(
                title="📊 Conversation Statistics",
                box=box.ROUNDED,
                title_style="bold cyan",
            )
            stats_table.add_column("Metric", style="bold")
            stats_table.add_column("Value", style="green")

            stats_table.add_row("Current Session ID", self.session_id)
            stats_table.add_row(
                "Current Session Conversations", str(self.conversation_count)
            )
            stats_table.add_row("Total Conversation History", str(len(history)))
            stats_table.add_row("User ID", self.user_id)

            # Show recent conversation info if available
            if history:
                latest = history[-1]
                latest_time = latest.get("timestamp", "Unknown")
                stats_table.add_row(
                    "Last Conversation",
                    latest_time[:19] if len(latest_time) > 19 else latest_time,
                )

            console.print(stats_table)

        except Exception as e:
            console.print(f"[bold red]❌ Error loading statistics: {str(e)}[/bold red]")

    def _reset_memories(self):
        """Reset all stored memories with confirmation."""
        console.print()
        console.print(
            Panel(
                "[bold red]⚠️  WARNING[/bold red]\n"
                "This will permanently delete ALL of Sumire's memories!\n"
                "This action cannot be undone.",
                title="Reset Memories",
                border_style="red",
            )
        )

        confirm = Prompt.ask(
            "\n[bold red]Are you absolutely sure? Type 'DELETE' to confirm[/bold red]",
            default="cancel",
        )

        if confirm == "DELETE":
            try:
                console.print("\n[dim]🧹 Resetting memories...[/dim]")

                # Reset Mem0 memories
                self.memory_manager.reset_memories()

                # Backup and clear conversation log
                backup_file = self.memory_manager.backup_main_conversation_log()
                if backup_file:
                    # Clear the main log by writing empty list
                    import json

                    with open(
                        conversation_settings["main_log_filename"],
                        "w",
                        encoding="utf-8",
                    ) as f:
                        json.dump([], f)

                console.print("[green]✅ All memories have been reset.[/green]")
                if backup_file:
                    console.print(
                        f"[dim]📁 Previous conversations backed up to: {backup_file}[/dim]"
                    )

            except Exception as e:
                console.print(
                    f"[bold red]❌ Error resetting memories: {str(e)}[/bold red]"
                )
        else:
            console.print("[green]Memory reset cancelled.[/green]")

    def _display_system_info(self):
        """Display system information."""
        console.print()

        info_table = Table(
            title="ℹ️ System Information", box=box.ROUNDED, title_style="bold cyan"
        )
        info_table.add_column("Component", style="bold")
        info_table.add_column("Status/Info", style="green")

        info_table.add_row("PACE Version", __version__)
        info_table.add_row("Architecture", "LangGraph-based MVS 0.1")
        info_table.add_row("LangGraph App Status", "Ready" if pace_app else "Error")
        info_table.add_row("Memory Backend", "Mem0 + ChromaDB")
        info_table.add_row("Debug Mode", "Enabled" if DEBUG_MODE else "Disabled")
        info_table.add_row("Session ID", self.session_id)

        # Show persona directives count
        # info_table.add_row("Persona Directives", str(len(core_persona_directives)))

        console.print(info_table)

    def _farewell(self):
        """Display farewell message."""
        console.print()
        farewell_text = Text()
        farewell_text.append("🌸 ", style="bright_magenta")
        farewell_text.append("Thank you for using PACE!", style="bold cyan")
        farewell_text.append(" 🌸\n\n", style="bright_magenta")
        farewell_text.append(
            "PACE will remember our conversations for next time. ", style="cyan"
        )
        farewell_text.append("See you soon! 💖", style="bright_magenta")

        console.print(
            Panel(
                farewell_text,
                title="👋 Farewell",
                border_style="bright_magenta",
                box=box.ROUNDED,
                padding=(1, 2),
            )
        )


def main():
    """Main entry point for the PACE CLI application."""
    try:
        cli = PACE_CLI()
        cli.run()
    except KeyboardInterrupt:
        console.print("\n\n[yellow]👋 Goodbye! Interrupted by user.[/yellow]")
    except Exception as e:
        console.print(f"\n[bold red]❌ Fatal error: {str(e)}[/bold red]")
        logger.error(f"Fatal application error: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

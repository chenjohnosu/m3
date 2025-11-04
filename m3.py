import click
import shlex
import sys
import uuid
import contextlib
import traceback

from cli.project_commands import project
from cli.corpus_commands import corpus
from cli.vector_commands import vector
from cli.analyze_commands import analyze
# Import the new session manager
from core.session_manager import M3Session

@click.group(invoke_without_command=True)
@click.option('--go', is_flag=True, help='Enters interactive mode.')
@click.option('--batch', 'batch_file', type=click.Path(exists=True), help='Executes commands from a file.')
@click.pass_context
def cli(ctx, go, batch_file):
    """
    m3: A tool for qualitative data analysis powered by local LLMs.
    """
    if ctx.invoked_subcommand is None:
        if go:
            # Pass a new M3Session into the context object
            # This is the core of the optimization
            interactive_mode(M3Session())
        elif batch_file:
            # Batch mode does not get the optimization, as it runs
            # commands as separate processes anyway.
            batch_mode(batch_file)
        else:
            click.echo(ctx.get_help())


def show_interactive_help():
    """Displays the help message for interactive mode."""
    click.echo("Available commands:")
    click.echo("  /project  (alias: /p) - Manage projects")
    click.echo("  /corpus   (alias: /c) - Manage a project's corpus")
    click.echo("  /vector   (alias: /v) - Manage a project's vector store")
    click.echo("  /analyze  (alias: /a) - Analyze project data")
    click.echo("  /help                 - Show this help message")
    click.echo("  /quit     (alias: /q) - Exit interactive mode")


def show_subcommand_help(command_path):
    """Displays the help message for a given command or subcommand."""
    args = command_path.split()
    cmd_obj = cli
    ctx = click.Context(cli)
    for arg in args:
        if isinstance(cmd_obj, click.Group):
            # Attempt to get the subcommand
            cmd_obj = cmd_obj.get_command(ctx, arg)
            if cmd_obj is None:
                click.echo(f"Error: No such command '{arg}'")
                return
            ctx = click.Context(cmd_obj, info_name=arg, parent=ctx)
        else:
            click.echo(f"Error: '{ctx.command.name}' does not have subcommands.")
            return

    if cmd_obj:
        click.echo(cmd_obj.get_help(ctx))


def interactive_mode(session: M3Session):
    """Starts a clean, simplified interactive REPL session."""
    click.echo("Entering interactive mode. Use '/quit' or '/q' to exit.")

    # --- NEW: Alias mapping ---
    command_aliases = {
        'c': 'corpus',
        'p': 'project',
        'v': 'vector',
        'q': 'quit',
        'a': 'analyze'
    }

    while True:
        # Get prompt from the session
        prompt = session.get_project_prompt()

        try:
            command = input(prompt).strip()

            if not command:
                continue

            if not command.startswith('/'):
                click.echo("Error: Invalid command format. Commands must start with a '/'.")
                continue

            command_to_process = command[1:]
            args = shlex.split(command_to_process)

            if not args:
                continue

            cmd = args[0].lower()

            # --- NEW: Resolve alias ---
            if cmd in command_aliases:
                cmd = command_aliases[cmd]
                args[0] = cmd  # Update the args list with the resolved command

            if cmd == 'quit':
                break

            if cmd == 'help':
                show_interactive_help()
                continue

            valid_commands = cli.list_commands(click.Context(cli))
            if cmd not in valid_commands:
                click.echo(f"Error: Command '{cmd}' unknown.")
                show_interactive_help()
                continue

            # If a group command is entered without a subcommand, show its help.
            cmd_obj = cli.get_command(click.Context(cli), cmd)
            if isinstance(cmd_obj, click.Group) and len(args) == 1:
                # Pass the session object to the help context
                ctx = click.Context(cmd_obj, info_name=cmd)
                ctx.obj = session
                show_subcommand_help(cmd)
                continue

            # --- MODIFIED: Pass the session object (ctx.obj) to the command ---
            with cli.make_context(cli.name, args, resilient_parsing=True, obj=session) as ctx:
                cli.invoke(ctx)

        except (EOFError, KeyboardInterrupt):
            break
        except click.exceptions.UsageError as e:
            click.echo(f"Error: {e.format_message()}", err=True)
            # When a usage error occurs, show the help for the command that was attempted.
            show_subcommand_help(" ".join(args))
        except Exception as e:
            click.echo(f"An unexpected error occurred: {e}", err=True)

    click.echo("\nExiting interactive mode.")


def batch_mode(filename):
    """
    Executes commands from a batch file.
    NOTE: This mode does NOT use the persistent session and will be slower,
    as each command is invoked in a clean context.
    """
    click.echo(f"Executing commands from '{filename}'...")
    click.secho("  > Batch mode runs without a persistent session.", dim=True)
    with open(filename, 'r') as f:
        for line in f:
            command = line.strip()
            if command and not command.startswith('#'):
                args = shlex.split(command)
                click.echo(f"==> {command}")
                # We do *not* pass ctx.obj here.
                with cli.make_context(cli.name, args, resilient_parsing=True) as ctx:
                    try:
                        cli.invoke(ctx)
                    except click.exceptions.UsageError as e:
                        click.echo(f"Error executing command '{command}': {e}", err=True)
                    except Exception as e:
                        click.echo(f"An unexpected error occurred while executing '{command}': {e}", err=True)


# Add the command groups to the main CLI tool
cli.add_command(project)
cli.add_command(corpus)
cli.add_command(vector)
cli.add_command(analyze)

if __name__ == '__main__':
    cli(obj={}) # obj={} is the default for non-interactive mode
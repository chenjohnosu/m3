import click

def start_dialogue(project_name):
    """Core logic to start a dialogue session."""
    click.echo(f"CORE: Starting dialogue for project '{project_name}'...")
    click.echo("Entering dialogue mode. Press Ctrl+C to exit.")
    # This is where you would start a REPL (Read-Eval-Print Loop)
    while True:
        try:
            prompt = input(f"({project_name}) > ")
            if prompt.lower() in ['exit', 'quit']:
                break
            click.echo(f"QUERY: {prompt}")
        except KeyboardInterrupt:
            break
    click.echo("\nExiting dialogue mode.")

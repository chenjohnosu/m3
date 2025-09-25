import click

def load_config():
    """Placeholder for loading application configuration."""
    click.echo("==> Util: Loading configuration")
    return {"llm_model": "llama3", "embed_model": "mxbai-embed-large"}

def save_config(config):
    """Placeholder for saving application configuration."""
    click.echo(f"==> Util: Saving configuration: {config}")

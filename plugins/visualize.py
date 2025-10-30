#
# plugins/visualize.py
#
import click
import os
from plugins.base_plugin import BaseAnalyzerPlugin

# Forward-declare AnalyzeManager
if "AnalyzeManager" not in globals():
    from typing import TypeVar

    AnalyzeManager = TypeVar("AnalyzeManager")

# --- Visualization Dependencies ---
try:
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.manifold import TSNE

    VIZ_AVAILABLE = True
except ImportError:
    VIZ_AVAILABLE = False


# ---------------------------------


class VisualizePlugin(BaseAnalyzerPlugin):
    """
    Generates a 2D knowledge map using t-SNE.
    """
    key: str = "visualize"
    description: str = "Generates a 2D t-SNE map of all chunks, colored by theme."

    def analyze(self, manager: AnalyzeManager, **kwargs):
        """
        Runs the t-SNE dimensionality reduction and saves a plot.
        """

        # --- 1. Check for Dependencies ---
        if not VIZ_AVAILABLE:
            click.secho("🔥 Error: Missing dependencies for the visualize plugin.", fg="red")
            click.echo("  > Please run: pip install pandas matplotlib seaborn")
            return

        click.secho(f"==> Running: {self.key} Plugin (t-SNE Knowledge Map)", fg="cyan")
        click.echo("  > This may take a few minutes for large datasets...")

        # --- 2. Get All Data from Vector Store ---
        try:
            # We need embeddings for coordinates and metadata for color-coding
            all_data = manager.collection.get(
                include=["embeddings", "metadatas", "documents"]
            )
        except Exception as e:
            click.secho(f"🔥 Error: Could not retrieve data from vector store: {e}", fg="red")
            return

        embeddings = all_data.get('embeddings')
        metadatas = all_data.get('metadatas')
        documents = all_data.get('documents')

        if embeddings is None or len(embeddings) == 0:
            click.secho("  > No embeddings found. Nothing to visualize.", fg="yellow")
            return

        n_samples = len(embeddings)
        click.echo(f"  > Found {n_samples} chunks to map.")

        if n_samples < 5:
            click.secho("  > Not enough data to create a map (requires >= 5 chunks).", fg="yellow")
            return

        # --- 3. Run t-SNE (Step 1 from your process) ---
        click.echo("  > Running t-SNE... (This is the slow part)")
        try:
            # t-SNE perplexity must be less than n_samples
            perplexity_value = min(30.0, float(n_samples - 1))

            tsne = TSNE(
                n_components=2,
                perplexity=perplexity_value,
                random_state=42,
                init='pca',
                learning_rate='auto'
            )
            embeddings_np = np.array(embeddings)
            coords = tsne.fit_transform(embeddings_np)
            click.echo("  > t-SNE complete.")

        except Exception as e:
            click.secho(f"🔥 Error during t-SNE reduction: {e}", fg="red")
            return

        # --- 4. Prepare Data for Plotting (Step 2 & 3) ---
        click.echo("  > Preparing data for plotting...")

        # Use 'axial_theme' if available (from clustering), otherwise 'doc_type'
        themes = [
            m.get('axial_theme', m.get('doc_type', 'Uncategorized'))
            for m in metadatas
        ]

        # Create a snippet for hover-text (though not used in this static plot)
        snippets = [
            (doc.replace('\n', ' ').strip()[:75] + '...') if len(doc) > 78 else doc
            for doc in documents
        ]

        df = pd.DataFrame({
            'x': coords[:, 0],
            'y': coords[:, 1],
            'theme': themes,
            'snippet': snippets
        })

        # --- 5. Generate and Save Plot (Step 3) ---
        try:
            plt.figure(figsize=(16, 10))
            ax = sns.scatterplot(
                data=df,
                x='x',
                y='y',
                hue='theme',
                s=50,
                alpha=0.7,
                palette='viridis'  # You can change this color palette
            )

            plt.title('Knowledge Map (t-SNE Visualization of Document Chunks)', fontsize=16)
            plt.xlabel('t-SNE Component 1')
            plt.ylabel('t-SNE Component 2')

            # Move the legend outside the plot
            ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

            # Get the project path from the manager
            save_path = os.path.join(manager.project_path, "knowledge_map.png")

            # Save the figure
            plt.savefig(save_path, bbox_inches='tight')

            click.secho(f"\n✅ Success! Knowledge map saved to:", fg="green", bold=True)
            click.echo(f"   {save_path}")

        except Exception as e:
            click.secho(f"🔥 Error during plot generation: {e}", fg="red")
import click
from plugins.base_plugin import BaseAnalyzerPlugin

# Forward-declare AnalyzeManager
if "AnalyzeManager" not in globals():
    from typing import TypeVar

    AnalyzeManager = TypeVar("AnalyzeManager")

# --- Anomaly Dependencies ---
# Attempt to import scikit-learn.
try:
    import numpy as np
    from sklearn.ensemble import IsolationForest

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


# -----------------------------


class AnomalyPlugin(BaseAnalyzerPlugin):
    """
    Detects outlier document chunks using IsolationForest.
    """
    key: str = "anomaly"
    description: str = "Finds document chunks that are semantic outliers."

    def analyze(self, manager: AnalyzeManager, **kwargs):
        """
        Runs the anomaly detection analysis.

        It fetches all embeddings, runs IsolationForest, and then
        displays the Top-K chunks with the lowest (most anomalous) scores.
        """

        # --- 1. Check for Dependencies ---
        if not SKLEARN_AVAILABLE:
            click.secho("🔥 Error: 'scikit-learn' is required for the anomaly plugin.", fg="red")
            click.echo("  > Please install it by running: pip install scikit-learn")
            return

        click.secho(f"==> Running: {self.key} Plugin", fg="cyan")
        click.echo("  > This plugin will find the most unique/isolated chunks.")

        # --- 2. Define Parameters ---
        # Get 'k' (number of outliers to show) from kwargs
        top_k = kwargs.get('k', 5)  # <-- MODIFIED

        # Contamination is the "expected" percentage of outliers.
        # 'auto' works well.
        contamination = 'auto'

        # --- 3. Get All Data from Vector Store ---
        try:
            all_data = manager.collection.get(
                include=["embeddings", "metadatas"]
            )
        except Exception as e:
            click.secho(f"🔥 Error: Could not retrieve data from vector store: {e}", fg="red")
            return

        embeddings = all_data.get('embeddings')
        metadatas = all_data.get('metadatas')

        if embeddings is None or len(embeddings) < 1:
            click.secho("  > No chunks found. Nothing to analyze.", fg="yellow")
            click.secho("==> Anomaly detection aborted.", fg="yellow")
            return

        click.echo(f"  > Analyzing {len(embeddings)} chunks for outliers.")

        # --- 4. Run IsolationForest ---
        try:
            embeddings_np = np.array(embeddings)

            # Initialize the model
            model = IsolationForest(
                contamination=contamination,
                random_state=42
            )

            # Fit the model (this step just learns the data structure)
            model.fit(embeddings_np)

            # Get the anomaly score for each chunk.
            # Lower scores are *more* anomalous.
            scores = model.decision_function(embeddings_np)

            # Get the indices of the chunks, sorted by score (lowest to highest)
            sorted_indices = np.argsort(scores)

            # Get the top K most anomalous chunk indices
            outlier_indices = sorted_indices[:top_k]  # <-- MODIFIED

        except Exception as e:
            click.secho(f"🔥 Error during anomaly detection: {e}", fg="red")
            return

        click.secho(f"\n==> Anomaly detection complete.", bold=True)
        click.secho(f"--- Top {top_k} Most Anomalous Chunks (Outliers) ---", bold=True)  # <-- MODIFIED

        # --- 5. Print the Results ---
        for i, chunk_index in enumerate(outlier_indices):
            score = scores[chunk_index]
            meta = metadatas[chunk_index]
            filename = meta.get('original_filename', 'Unknown')

            # Get the clean, original text for display
            original_text = meta.get('original_text', 'No original text found.')

            # Create a clean, one-line snippet
            snippet = original_text.replace('\n', ' ').strip()
            snippet = (snippet[:75] + '...') if len(snippet) > 78 else snippet

            click.secho(f"\n[Rank {i + 1}] (Score: {score:.4f})", fg="yellow", bold=True)
            click.echo(f"  > Source:  {filename}")
            click.echo(f"  > Content: \"{snippet}\"")

        click.secho("\n--- End of Results ---", bold=True)
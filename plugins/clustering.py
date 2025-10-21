import click
from plugins.base_plugin import BaseAnalyzerPlugin
from collections import defaultdict

# Forward-declare AnalyzeManager
if "AnalyzeManager" not in globals():
    from typing import TypeVar

    AnalyzeManager = TypeVar("AnalyzeManager")

# --- Clustering Dependencies ---
# Attempt to import scikit-learn.
try:
    import numpy as np
    from sklearn.cluster import KMeans
    from sklearn.feature_extraction.text import TfidfVectorizer

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


# -----------------------------


class ClusteringPlugin(BaseAnalyzerPlugin):
    """
    Performs K-Means clustering on document chunks.
    """
    key: str = "clustering"
    description: str = "Groups all document chunks into N clusters using K-Means."

    def analyze(self, manager: AnalyzeManager, **kwargs):
        """
        Runs the clustering analysis.

        It fetches all embeddings, runs K-Means, and then for each cluster,
        it runs TF-IDF to find the most representative terms.
        """

        # --- 1. Check for Dependencies ---
        if not SKLEARN_AVAILABLE:
            click.secho("🔥 Error: 'scikit-learn' is required for the clustering plugin.", fg="red")
            click.echo("  > Please install it by running: pip install scikit-learn")
            return

        click.secho(f"==> Running: {self.key} Plugin", fg="cyan")
        click.echo("  > This plugin will find all document vectors and group them.")

        # --- 2. Define Parameters ---
        # Get 'k' from kwargs, with a default of 5 if not provided
        k = kwargs.get('k', 5)  # <-- MODIFIED

        # --- 3. Get All Data from Vector Store ---
        try:
            all_data = manager.collection.get(
                include=["embeddings", "metadatas", "documents"]
            )
        except Exception as e:
            click.secho(f"🔥 Error: Could not retrieve data from vector store: {e}", fg="red")
            return

        embeddings = all_data.get('embeddings')
        metadatas = all_data.get('metadatas')
        documents = all_data.get('documents')  # 'documents' has the full text used for embedding

        # We must check for 'is None' explicitly, as 'if not numpy_array' is ambiguous
        if embeddings is None or len(embeddings) < k:
            click.secho(
                f"  > Found {len(embeddings) if embeddings is not None else 0} chunks. Not enough data to cluster with k={k}.",
                fg="yellow")
            click.secho("==> Clustering analysis aborted.", fg="yellow")
            return

        click.echo(f"  > Found {len(embeddings)} chunks to analyze...")
        click.echo(f"  > Running K-Means clustering (k={k})...")  # <-- MODIFIED

        # --- 4. Run K-Means Clustering ---
        try:
            # Convert to numpy array just in case it's a list (KMeans prefers numpy)
            embeddings_np = np.array(embeddings)

            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(embeddings_np)
        except Exception as e:
            click.secho(f"🔥 Error during clustering: {e}", fg="red")
            return

        # --- 5. Group Chunks by Cluster ---
        clustered_chunks = defaultdict(list)
        for i, label in enumerate(labels):
            clustered_chunks[label].append({
                "metadata": metadatas[i],
                "document": documents[i]  # The full text for TF-IDF
            })

        click.secho(f"\n==> Clustering analysis complete. Found {k} clusters.", bold=True)

        # --- 6. Analyze and Print Each Cluster ---
        for cluster_id in range(k):
            chunks_in_cluster = clustered_chunks.get(cluster_id, [])
            if not chunks_in_cluster:
                continue

            click.secho(f"\n--- Cluster {cluster_id + 1} ({len(chunks_in_cluster)} Chunks) ---", bold=True)

            # --- 7. Find Representative Terms using TF-IDF ---
            try:
                # Get the text for this cluster. We use 'document' (the embedded text)
                # as it's what the vector represents.
                cluster_texts = [c['document'] for c in chunks_in_cluster]

                # Use TF-IDF to find the top 5 most important words
                vectorizer = TfidfVectorizer(
                    max_features=5,
                    stop_words='english',
                    ngram_range=(1, 2)  # Allow single words and two-word phrases
                )
                vectorizer.fit(cluster_texts)
                terms = vectorizer.get_feature_names_out()

                click.secho(f"  > Representative Terms: ", nl=False)
                click.echo(", ".join(terms))

            except Exception as e:
                # This can fail if a cluster has empty strings or only stop words
                click.secho(f"  > Could not determine representative terms: {e}", fg="yellow")

            # --- 8. Print Chunk Samples ---
            click.echo("  > Chunks:")

            # Show the first 3 chunks as samples
            for chunk in chunks_in_cluster[:3]:
                meta = chunk['metadata']
                filename = meta.get('original_filename', 'Unknown')

                # Get the clean, original text for display
                original_text = meta.get('original_text', 'No original text found.')

                # Create a clean, one-line snippet
                snippet = original_text.replace('\n', ' ').strip()
                snippet = (snippet[:75] + '...') if len(snippet) > 78 else snippet

                click.echo(f"    - [{filename}] \"{snippet}\"")

            if len(chunks_in_cluster) > 3:
                click.echo(f"    - (and {len(chunks_in_cluster) - 3} more chunks...)")
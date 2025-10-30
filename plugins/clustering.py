# plugins/clustering.py

import click
import json
import re
from plugins.base_plugin import BaseAnalyzerPlugin
from collections import defaultdict
from llama_index.core.llms import ChatMessage

# (Imports for AnalyzeManager, sklearn, and AXIAL_CODING_PROMPT remain the same)
# ...
# Forward-declare AnalyzeManager
if "AnalyzeManager" not in globals():
    from typing import TypeVar

    AnalyzeManager = TypeVar("AnalyzeManager")

# --- Clustering Dependencies ---
try:
    import numpy as np
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.feature_extraction.text import TfidfVectorizer

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
# -----------------------------

# --- NEW: System prompt for Axial Coding ---
AXIAL_CODING_PROMPT = """
You are an expert qualitative data analyst. You will be given a list of "open codes" or "initial themes" identified in a set of related data chunks.
Your task is to perform "axial coding" by synthesizing these initial themes into a single, more abstract "core theme" (3-7 words) that represents the central concept of the cluster.
The output must be a single, valid JSON object with one key: "axial_theme".

Example Input:
["Difficulty finding information", "Website navigation issues", "Confusing help articles", "Unclear instructions"]

Example Output:
{"axial_theme": "User frustration with information access"}
"""


# -----------------------------------------


class ClusteringPlugin(BaseAnalyzerPlugin):
    """
    Performs Hierarchical clustering and Axial Coding on document chunks.
    """
    key: str = "clustering"
    description: str = "Groups chunks via Hierarchical Clustering & synthesizes themes (Axial Coding)."

    def analyze(self, manager: AnalyzeManager, **kwargs):
        """
        Runs the full clustering and axial coding analysis.
        Checks for '--save' flag in kwargs to persist metadata.
        """

        # --- 1. Check for Dependencies ---
        if not SKLEARN_AVAILABLE:
            click.secho("🔥 Error: 'scikit-learn' is required for the clustering plugin.", fg="red")
            click.echo("  > Please install it by running: pip install scikit-learn")
            return

        click.secho(f"==> Running: {self.key} Plugin (Hierarchical + Axial Coding)", fg="cyan")

        # --- 2. Define Parameters & Check Save Flag ---
        k = kwargs.get('k', 5)
        # --- MODIFIED: Check for the --save flag ---
        save_results = kwargs.get('save', False)

        if save_results:
            click.secho("  > Mode: 'Save'. This will group all chunks and *save* new metadata.", bold=True)
        else:
            click.secho("  > Mode: 'Preview'. This will group chunks and *display* themes.", bold=True)
            click.echo("    (Run with --save to persist these results to metadata)")
        # ------------------------------------------

        # --- 3. Get LLM from manager ---
        try:
            llm = manager.llm
            if not llm:
                raise ValueError("LLM is not available via AnalyzeManager.")
        except Exception as e:
            click.secho(f"🔥 Error: Could not load LLM for theme synthesis: {e}", fg="red")
            return

        # --- 4. Get All Data from Vector Store ---
        try:
            all_data = manager.collection.get(
                include=["embeddings", "metadatas", "documents", "ids"]
            )
        except Exception as e:
            click.secho(f"🔥 Error: Could not retrieve data from vector store: {e}", fg="red")
            return

        ids = all_data.get('ids')
        embeddings = all_data.get('embeddings')
        metadatas = all_data.get('metadatas')
        documents = all_data.get('documents')

        if embeddings is None or len(embeddings) < k:
            click.secho(f"  > Not enough data to cluster with k={k}.", fg="yellow")
            return

        click.echo(f"  > Found {len(embeddings)} chunks to analyze...")
        click.echo(f"  > Running Hierarchical Clustering (n_clusters={k})...")

        # --- 5. Run Hierarchical Clustering ---
        try:
            embeddings_np = np.array(embeddings)
            clusterer = AgglomerativeClustering(n_clusters=k)
            labels = clusterer.fit_predict(embeddings_np)
        except Exception as e:
            click.secho(f"🔥 Error during clustering: {e}", fg="red")
            return

        # --- 6. Group Chunks by Cluster ---
        clustered_chunks = defaultdict(list)
        for i, label in enumerate(labels):
            clustered_chunks[label].append({
                "id": ids[i],
                "metadata": metadatas[i],
                "document": documents[i]
            })

        click.secho(f"\n==> Clustering analysis complete. Found {k} clusters.", bold=True)

        # --- 7. Analyze, Synthesize, and (Optionally) Update Each Cluster ---
        for cluster_id in range(k):
            chunks_in_cluster = clustered_chunks.get(cluster_id, [])
            if not chunks_in_cluster:
                continue

            cluster_label_str = f"cluster_{cluster_id + 1}"
            click.secho(f"\n--- Cluster {cluster_id + 1} ({len(chunks_in_cluster)} Chunks) ---", bold=True)

            # --- 8. Perform Axial Coding (Theme Synthesis) ---
            axial_theme = "N/A"  # Default theme
            try:
                # (This logic remains the same as before)
                cluster_themes = []
                for chunk in chunks_in_cluster:
                    if 'themes' in chunk['metadata']:
                        themes_str = chunk['metadata'].get('themes', '')
                        themes_list = themes_str.split(',')
                        cluster_themes.extend([t.strip() for t in themes_list if t.strip()])

                if cluster_themes:
                    unique_themes = sorted(list(set(cluster_themes)))
                    messages = [
                        ChatMessage(role="system", content=AXIAL_CODING_PROMPT),
                        ChatMessage(role="user", content=json.dumps(unique_themes))
                    ]
                    click.echo(f"  > Synthesizing {len(unique_themes)} unique themes for Axial Code...")
                    response = llm.chat(messages)
                    response_text = response.message.content.strip()

                    json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                    if not json_match:
                        raise ValueError("No JSON object found in LLM response.")

                    theme_json = json.loads(json_match.group(0))
                    axial_theme = theme_json.get('axial_theme', 'N/A')

                    click.secho(f"  > Axial Theme: ", nl=False, fg="cyan")
                    click.secho(f"{axial_theme}", bold=True)
                else:
                    click.secho(f"  > No 'themes' metadata found to synthesize.", fg="yellow")
                    axial_theme = "Uncategorized"

            except Exception as e:
                click.secho(f"  > Warning: Could not synthesize axial theme: {e}", fg="yellow")

            # --- 9. MODIFIED: Update Metadata in Vector Store (Only if --save is used) ---
            if save_results:
                click.echo(f"  > Writing new metadata to {len(chunks_in_cluster)} chunks...")
                try:
                    cluster_ids_to_update = [c['id'] for c in chunks_in_cluster]
                    new_metadatas_for_update = []

                    for chunk in chunks_in_cluster:
                        new_meta = chunk['metadata'].copy()
                        new_meta['axial_theme'] = axial_theme
                        new_meta['cluster_id'] = cluster_label_str
                        new_metadatas_for_update.append(new_meta)

                    manager.collection.update(
                        ids=cluster_ids_to_update,
                        metadatas=new_metadatas_for_update
                    )
                    click.secho(f"  > Successfully updated metadata for {len(cluster_ids_to_update)} chunks.",
                                fg="green")

                except Exception as e:
                    click.secho(f"  > 🔥 Error updating metadata in vector store: {e}", fg="red")
            # -----------------------------------------------------------------

            # --- 10. Find Representative Terms (for display) ---
            # (This logic remains the same as before)
            try:
                cluster_texts = [c['document'] for c in chunks_in_cluster]
                vectorizer = TfidfVectorizer(max_features=5, stop_words='english', ngram_range=(1, 2))
                vectorizer.fit(cluster_texts)
                terms = vectorizer.get_feature_names_out()
                click.secho(f"  > Representative Terms: ", nl=False)
                click.echo(", ".join(terms))
            except Exception as e:
                click.secho(f"  > Could not determine representative terms: {e}", fg="yellow")

            # --- 11. Print Chunk Samples (for display) ---
            # (This logic remains the same as before)
            click.echo("  > Sample Chunks:")
            for chunk in chunks_in_cluster[:3]:
                meta = chunk['metadata']
                filename = meta.get('original_filename', 'Unknown')
                original_text = meta.get('original_text', meta.get('text', 'No original text.'))
                snippet = original_text.replace('\n', ' ').strip()
                snippet = (snippet[:75] + '...') if len(snippet) > 78 else snippet
                click.echo(f"    - [{filename}] \"{snippet}\"")
            if len(chunks_in_cluster) > 3:
                click.echo(f"    - (and {len(chunks_in_cluster) - 3} more chunks...)")
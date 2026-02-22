import yaml
import os
import shutil

_config = None
_prompts = None

def get_config_dir():
    """Returns the path to the ~/.monkey3 configuration directory."""
    return os.path.expanduser("~/.monkey3")

def get_config_path():
    """Returns the full path to the config.yaml file."""
    return os.path.join(get_config_dir(), "config.yaml")

def get_prompts_path():
    """Returns the full path to the prompts.yaml file."""
    return os.path.join(get_config_dir(), "prompts.yaml")

def create_default_config_if_not_exists():
    """Creates a default config.yaml by copying from the project root if it doesn't exist."""
    config_path = get_config_path()
    if not os.path.exists(config_path):
        print("No config.yaml found. Creating a default one in ~/.monkey3/")
        os.makedirs(os.path.dirname(config_path), exist_ok=True)

        # Path to the config.yaml in the project's root directory
        source_config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'config.yaml'))

        if os.path.exists(source_config_path):
            shutil.copy(source_config_path, config_path)
            print(f"Default config copied to: {config_path}")
        else:
            # Fallback to creating a default config if the source file is not found
            print("Source config.yaml not found, creating a default one.")
            default_config = {
                'project_settings': {
                    'projects_directory': '~/.monkey3/projects'
                },
                'llm_providers': {
                    'ollama_client': {
                        'provider': 'ollama',
                        'base_url': 'http://localhost:11434',
                        'models': {
                            'synthesis_model': {'model_name': 'llama3', 'request_timeout': 120.0},
                            'enrichment_model': {'model_name': 'mistral', 'request_timeout': 60.0}
                        }
                    }
                },
                'ingestion_config': {
                    'known_doc_types': [
                        'document', 'interview', 'paper', 'data',
                        'observation', 'ethnographic_notes'
                    ],
                    'default_doc_type': 'document',
                    'cogarc_settings': {
                        'stage_0_model': 'synthesis_model',
                        'stage_1_model': 'synthesis_model',
                        'stage_2_model': 'enrichment_model',
                        'stage_3_model': 'synthesis_model'
                    }
                }
            }
            with open(config_path, 'w') as f:
                yaml.dump(default_config, f, sort_keys=False)
            print(f"Default config created at: {config_path}")

def create_default_prompts_if_not_exists():
    """Creates a default prompts.yaml by copying from the project root if it doesn't exist."""
    prompts_path = get_prompts_path()
    if not os.path.exists(prompts_path):
        print("No prompts.yaml found. Creating a default one in ~/.monkey3/")
        os.makedirs(os.path.dirname(prompts_path), exist_ok=True)

        # Path to the prompts.yaml in the project's root directory
        source_prompts_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'prompts.yaml'))

        if os.path.exists(source_prompts_path):
            shutil.copy(source_prompts_path, prompts_path)
            print(f"Default prompts copied to: {prompts_path}")
        else:
            # Fallback to creating a default prompts file if source is missing
            print("Source prompts.yaml not found, creating a default one with built-in prompts.")
            default_prompts = {
                'prompts': {
                    'dialogue_system': "You are a helpful research assistant specialized in qualitative analysis. Answer the user's questions based ONLY on the context provided from the documents. If the answer is not in the context, say so explicitly. Do not hallucinate information. When applicable, cite the specific documents (by filename) you used to form your answer.",
                    'ingestion_stratify': "You are an expert qualitative data analysis assistant. Your task is to read an interview transcript and stratify it into a series of questions and their corresponding answers.\nAnalyze the provided transcript and perform the following actions:\n1.  Identify each question asked by the \"Interviewer\".\n2.  Identify the block of text that constitutes the \"Interviewee's\" answer to that question.\n3.  Structure your output as a single, valid JSON array of objects.\n4.  Each object in the array must represent a single answer and contain two keys:\n    - \"question\": A string containing the full text of the question.\n    - \"answer\": A string containing the full, corresponding block of text for the answer.\nImportant Rules:\n-   Ignore any introductory text or metadata at the beginning of the transcript.\n-   Combine multi-part answers into a single \"answer\" block for the most recent question.\n-   Ensure the final output is only the JSON array, with no explanations or conversational text.",
                    'ingestion_structure': "You are an expert qualitative data analyst. Your task is to read a piece of text and identify its core underlying themes or topics.\nAnalyze the provided text and perform the following actions:\n1.  Read the text to understand its main points and arguments.\n2.  Identify 2-4 distinct, high-level themes that capture the essence of the text. A theme should be a short phrase (3-5 words).\n3.  Structure your output as a single, valid JSON array of strings.\n4.  Each string in the array must be a single theme.\n\nExample Output: `[\"Online learning experiences\", \"Connection to the university\", \"Career preparation and skills\"]`\n\nImportant Rules:\n-   Focus on the conceptual topics, not just keywords.\n-   Ensure the final output is only the JSON array, with no explanations or conversational text.",
                    'ingestion_enrich': "You are an expert in synthesizing information. Your task is to read the following text chunk and generate a single, concise, and relevant question that this text could answer.\nThe question should be a natural-language query that a user might ask to find this specific information.\n\n---\nIMPORTANT RULES:\n1.  Your output MUST be only the question itself.\n2.  Do NOT include any preamble like \"Here is the question:\".\n3.  Do NOT copy any part of this system prompt.\n4.  Generate ONE question ONLY.\n5.  The output must be a single string, not a JSON object.\n---\n\nRead the text below and provide only the question.",
                    'ingestion_synthesis': "You are an expert qualitative data analyst. Your task is to synthesize a collection of text chunks from a single document into a concise, abstractive summary.\nAnalyze the provided text and perform the following actions:\n1.  Read all the text chunks to understand the document's main points, arguments, and narrative flow.\n2.  Generate a single, holistic summary (3-5 sentences) that captures the core essence and key takeaways of the entire document.\n3.  Ensure the summary is abstractive, meaning you should synthesize ideas in your own words rather than just extracting and combining sentences.\nhr\nImportant Rules:\n-   The final output should be only the summary text, with no explanations, conversational text, or preamble like \"Here is the summary:\".\n-   Focus on the overarching themes and conclusions from the text.",
                    'analysis_interpret': "You are an expert qualitative data analyst. You will be provided with a list of individual document summaries from a research corpus.\nYour task is to read all of them and synthesize them into a single, overarching \"meta-summary\" (3-5 paragraphs) that describes the entire collection as a whole.\nIdentify the key, high-level themes, patterns, and any potential contradictions that emerge from the corpus.\nDo not just list the summaries; synthesize them.",
                    'analysis_clustering_axial': "You are an expert qualitative data analyst. You will be given a list of \"open codes\" or \"initial themes\" identified in a set of related data chunks.\nYour task is to perform \"axial coding\" by synthesizing these initial themes into a single, more abstract \"core theme\" (3-7 words) that represents the central concept of the cluster.\nThe output must be a single, valid JSON object with one key: \"axial_theme\".\n\nExample Input:\n[\"Difficulty finding information\", \"Website navigation issues\", \"Confusing help articles\", \"Unclear instructions\"]\n\nExample Output:\n{\"axial_theme\": \"User frustration with information access\"}",
                    'plugin_summarize': "You are an expert summarization assistant. Based *only* on the context provided by the user, write a concise, multi-paragraph summary that directly answers the user's query.\n\nDo not use any information other than the context provided.\n\nUSER QUERY: \"{query}\"",
                    'plugin_sentiment': "You are an expert sentiment analyst. Read the provided context chunks, which are all relevant to the user's query, and perform a sentiment analysis based *only* on that context.\n\nUSER QUERY: \"{query}\"\n\nFirst, provide an *overall* sentiment (Positive, Negative, Neutral, or Mixed) for the topic as a whole. \n\nThen, list any individual chunks that show particularly strong sentiment, explaining your reasoning.",
                    'plugin_categorize': "You are an expert text categorization assistant. Analyze the provided context chunks and categorize each chunk in relation to the user's query: \"{query}\"\n\nUse *only* the following categories: {options}\n\nList each chunk's source file and its assigned category. Provide a brief justification for each categorization.",
                    'plugin_categorize_error': "You are a text categorization assistant. The user failed to provide categories. Please tell them to use the --options flag.\n\nExample: /a run categorize \"feedback\" --options='Positive,Negative,Neutral'",
                    'plugin_entity': "You are an expert entity extraction assistant. Read the provided context chunks and extract all entities of the following types: {options}\n\nThe analysis should focus on information relevant to the user's query: \"{query}\"\n\nList the extracted entities, grouped by type. Only list entities explicitly found in the text.",
                    'plugin_entity_error': "You are an entity extraction assistant. The user failed to provide entity types to extract. Please tell them to use the --options flag.\n\nExample: /a run entity \"safety concerns\" --options='People,Locations,Equipment'"
                }
            }
            with open(prompts_path, 'w') as f:
                yaml.dump(default_prompts, f, sort_keys=False)
            print(f"Default prompts created at: {prompts_path}")

def get_config():
    """Loads and returns the configuration from ~/.monkey3/config.yaml."""
    global _config
    if _config is None:
        create_default_config_if_not_exists()
        config_path = get_config_path()
        with open(config_path, 'r') as f:
            _config = yaml.safe_load(f)
    return _config

def get_prompts():
    """Loads and returns the prompts from ~/.monkey3/prompts.yaml."""
    global _prompts
    if _prompts is None:
        create_default_prompts_if_not_exists()
        prompts_path = get_prompts_path()
        with open(prompts_path, 'r') as f:
            data = yaml.safe_load(f)
            _prompts = data.get('prompts', {})
    return _prompts
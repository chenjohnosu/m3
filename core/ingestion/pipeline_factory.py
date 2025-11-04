from core.ingestion.cognitive_architect_pipeline import CognitiveArchitectPipeline

PIPELINES = {
    "cogarc": CognitiveArchitectPipeline,
}

def get_pipeline(pipeline_name, config, llm_manager):
    pipeline_class = PIPELINES.get(pipeline_name)
    if not pipeline_class:
        raise ValueError(f"Unknown pipeline: '{pipeline_name}'.")
    # Pass the llm_manager to the constructor
    return pipeline_class(config, llm_manager)
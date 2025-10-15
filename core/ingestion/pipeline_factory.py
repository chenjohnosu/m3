from core.ingestion.cognitive_architect_pipeline import CognitiveArchitectPipeline

PIPELINES = {
    "cogarc": CognitiveArchitectPipeline,
}

def get_pipeline(pipeline_name, config):
    pipeline_class = PIPELINES.get(pipeline_name)
    if not pipeline_class:
        raise ValueError(f"Unknown pipeline: '{pipeline_name}'.")
    return pipeline_class(config)
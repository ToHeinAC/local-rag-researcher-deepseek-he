import os
from dataclasses import dataclass, fields
from typing import Any, Optional
from langchain_core.runnables import RunnableConfig
from dataclasses import dataclass

DEFAULT_REPORT_STRUCTURE = """
# Introduction
- Brief overview of the research topic or question.
- Purpose and scope of the report.

# Main Body
- For each section (1-4 sections):
  - Subheading: Provide a relevant subheading to the section's key aspect.
  - Explanation: A detailed explanation of the concept or topic being discussed in the section.
  - Findings/Details: Support the explanation with research findings, statistics, examples, or case studies.

# Key Takeaways
- Bullet points summarizing the most important insights or findings.

# Conclusion
- Final summary of the research.
- Implications or relevance of the findings.   
"""

@dataclass(kw_only=True)
class Configuration:
    """The configurable fields for the chatbot."""
    report_structure: str = DEFAULT_REPORT_STRUCTURE
    max_search_queries: int = 5
    enable_web_search: bool = False
    enable_quality_checker: bool = True
    quality_check_loops: int = 1
    llm_model: str = "mistral-small:latest"  # Default general purpose LLM model
    report_llm: str = "deepseek-r1:latest"  # Default report writing LLM model
    summarization_llm: str = "llama3.2"  # Default summarization LLM model
    #embedding_model: str = "sentence-transformers/all-mpnet-base-v2"
    embedding_model: str = "jinaai/jina-embeddings-v2-base-de"
    
    def update_embedding_model(self, model_name: str) -> None:
        """Update the embedding model at runtime."""
        self.embedding_model = model_name
        
    def items(self):
        """Make the Configuration class compatible with dictionary operations.
        Returns key-value pairs of all configuration attributes."""
        return {field.name: getattr(self, field.name) for field in fields(self)}.items()
        
    def __getitem__(self, key):
        """Allow dictionary-like access to configuration attributes."""
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(f"Configuration has no attribute {key}")
        
    def get(self, key, default=None):
        """Dictionary-like get method that returns default if key doesn't exist."""
        if hasattr(self, key):
            return getattr(self, key)
        return default

# Global configuration instance
_config_instance = None

def get_config_instance() -> Configuration:
    """Get the global configuration instance."""
    global _config_instance
    if _config_instance is None:
        _config_instance = Configuration()
    return _config_instance

def update_embedding_model(model_name: str) -> None:
    """Update the embedding model in the global configuration."""
    config = get_config_instance()
    config.embedding_model = model_name
    print(f"Updated global embedding model to: {model_name}")


    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        """Create a Configuration instance from a RunnableConfig."""
        configurable = (
            config["configurable"] if config and "configurable" in config else {}
        )
        values: dict[str, Any] = {
            f.name: os.environ.get(f.name.upper(), configurable.get(f.name))
            for f in fields(cls)
            if f.init
        }
        values['embedding_model'] = os.environ.get('EMBEDDING_MODEL', configurable.get('embedding_model'))
        return cls(**{k: v for k, v in values.items() if v})
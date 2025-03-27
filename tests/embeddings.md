Here's a comparison of six popular embedding models compatible with Ollama, focusing on German text performance:

| Model | Size | License | Benchmark Score | German Text Strength |
|-------|------|---------|-----------------|----------------------|
| **nomic-embed-text** | 137M | Apache-2.0 | MTEB: 62.39LoCo: 54.90[13] | 8192-token context handles complex German documents[4][13] |
| **sentence-transformers/all-mpnet-base-v2** | 110M | typically Apache-2.0 | STS-B: 87.9MTEB: 76.8[5] | Requires multilingual variant for German support[5][12] |
| **BGE-M3** | 568M | MIT | MTEB: 56.9LoCo: 53.2[6] | Supports 100+ languages including German through multilingual training[6] |
| **T-Systems-onsite/cross-en-de-roberta** | 355M | MIT | Spearman DE: 0.855Cross-lingual: 0.852[14] | Specialized for German-English bilingual tasks[9][14] |
| **jina-embeddings-v2-base-de** | 161M | Apache-2.0 | German STS: 85.3Cross-lingual: 84.1[7] | Native German support with 8192-token context[7][12] |
| **multilingual-e5-large** | 335M | MIT | MrTyDi DE: 72.1Cross-lingual: 68.9[8] | Explicit German support in 100-language model[8][12] |

**Key differentiators for German text:**
- **jina-embeddings-v2-base-de** leads in native German performance with minimal English bias[7][12]
- **T-Systems-onsite** model excels in cross-lingual DE<>EN tasks[14]
- **nomic-embed-text** offers best long-context (8192 token) handling[11][13]
- **multilingual-e5-large** provides broad 100-language coverage including German[8]
- **BGE-M3** balances size with multilingual capabilities[6]
- **all-mpnet-base-v2** requires multilingual variant for optimal German performance[5]

All models integrate with Ollama through API endpoints or LangChain/LlamaIndex tooling[10]. For pure German applications, jina-embeddings-v2-base-de and T-Systems-onsite models show specialized capabilities, while nomic-embed-text offers the best general-purpose performance with long German documents[7][13][14].

Citations:
[1] https://klu.ai/blog/open-source-llm-models
[2] https://huggingface.co/sentence-transformers/all-mpnet-base-v2
[3] https://www.restack.io/p/ollama-knowledge-best-model-german-cat-ai
[4] https://the-decoder.de/open-source-modell-von-nomic-ai-zur-texteinbettung-uebertrifft-ada-002-von-openai/
[5] https://www.aimodels.fyi/models/huggingFace/all-mpnet-base-v2-sentence-transformers
[6] https://www.linkedin.com/pulse/guidebook-state-of-the-art-embeddings-information-aapo-tanskanen-pc3mf
[7] https://jina.ai/de/news/ich-bin-ein-berliner-german-english-bilingual-embeddings-with-8k-token-length/
[8] https://deepinfra.com/intfloat/multilingual-e5-large/api
[9] https://www.aimodels.fyi/models/huggingFace/cross-en-de-roberta-sentence-transformer-t-systems-onsite
[10] https://ollama.com/blog/embedding-models
[11] https://ollama.com/library/nomic-embed-text
[12] https://www.reddit.com/r/LangChain/comments/1aqh168/best_german_embedding_model/
[13] https://www.nomic.ai/blog/posts/nomic-embed-text-v1
[14] https://huggingface.co/T-Systems-onsite/cross-en-de-roberta-sentence-transformer
[15] https://www.reddit.com/r/LangChain/comments/1aqh168/best_german_embedding_model/?tl=de
[16] https://ollama.com/models
[17] https://ollama.com/library/bge-m3
[18] https://huggingface.co/BAAI/bge-m3
[19] https://ollama.com/jina/jina-embeddings-v2-base-de
[20] https://dataloop.ai/library/model/intfloat_multilingual-e5-large/
[21] http://arxiv.org/pdf/2402.01613.pdf
[22] https://www.youtube.com/watch?v=f4tXwCNP1Ac
[23] https://www.sbert.net/docs/sentence_transformer/pretrained_models.html

---
Answer from Perplexity: pplx.ai/share
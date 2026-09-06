# Awesome Rerankers [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

> A curated list of reranking models, libraries, and resources for RAG applications.

Rerankers take a query and retrieved documents and reorder them by relevance. They use cross-encoders to jointly encode query-document pairs, which is slower than vector search but more accurate. Typical pipeline: retrieve 50-100 candidates with vector search, rerank to top 3-5.

## Contents

- [What are Rerankers?](#what-are-rerankers)
- [Top Models Comparison](#top-models-comparison)
- [Quick Start](#quick-start)
- [Open Source Models](#open-source-models)
  - [Cross-Encoder Models](#cross-encoder-models)
  - [T5-Based Models](#t5-based-models)
  - [LLM-Based Models](#llm-based-models)
- [Commercial APIs](#commercial-apis)
- [Libraries & Frameworks](#libraries--frameworks)
- [RAG Framework Integrations](#rag-framework-integrations)
  - [LangChain](#langchain)
  - [LlamaIndex](#llamaindex)
  - [Haystack](#haystack)
- [Datasets & Benchmarks](#datasets--benchmarks)
- [Evaluation Metrics](#evaluation-metrics)
- [Research Papers](#research-papers)
- [Tutorials & Resources](#tutorials--resources)
- [Tools & Utilities](#tools--utilities)
- [Reranker Leaderboard](#reranker-leaderboard)
- [Related Awesome Lists](#related-awesome-lists)

## What are Rerankers?

Rerankers refine search results by re-scoring query-document pairs. Key differences from vector search:

**Vector search (bi-encoders):**
- Encodes query and documents separately
- Fast (pre-computed embeddings)
- Returns 50-100 candidates

**Reranking (cross-encoders):**
- Jointly encodes query + document
- Slower but more accurate
- Refines to top 3-5 results

**Types:** Pointwise (score each doc independently), pairwise (compare pairs), listwise (score entire list)

## Top Models Comparison

| Model | Type | Multilingual | Deployment | Best For |
|-------|------|--------------|------------|----------|
| [Cohere Rerank](https://docs.cohere.com/docs/reranking) | API | 100+ languages | Cloud | Production, easy start |
| [Voyage Rerank 2.5](https://docs.voyageai.com/docs/reranker) | API | English-focused | Cloud | Highest accuracy |
| [Jina Reranker v2](https://jina.ai/reranker/) | API/OSS | 100+ languages | Cloud/Self-host | Balance cost/quality |
| [BGE-Reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3) | Open Source | 100+ languages | Self-host | Free, multilingual |
| [mxbai-rerank-large-v2](https://huggingface.co/mixedbread-ai/mxbai-rerank-large-v2) | Open Source | English | Self-host | Best OSS accuracy |
| [FlashRank](https://github.com/PrithivirajDamodaran/FlashRank) | Open Source | Limited | Self-host | Lightweight, CPU-only |

**→ [View Full Benchmarks & Leaderboard](https://agentset.ai/rerankers)** - Live comparison of rerankers on production benchmarks including NDCG@10, latency, and cost metrics. Updated regularly with new models and real-world performance data.

## Quick Start

**5-Minute Setup:**

```python
# Option 1: Cohere API (easiest)
from cohere import Client
client = Client("your-api-key")
results = client.rerank(
    query="What is deep learning?",
    documents=["Doc 1...", "Doc 2..."],
    model="rerank-v3.5",
    top_n=3
)

# Option 2: Self-hosted (free)
from sentence_transformers import CrossEncoder
model = CrossEncoder('BAAI/bge-reranker-v2-m3')
scores = model.predict([
    ["What is deep learning?", "Doc 1..."],
    ["What is deep learning?", "Doc 2..."]
])
```

**Choosing a Reranker:** For help selecting the best reranker for your use case, check out [Best Reranker for RAG: We tested the top models](https://agentset.ai/blog/best-reranker) where we break down consistency, accuracy, and performance across top models.

## Open Source Models

### Cross-Encoder Models

Cross-encoders jointly encode query and document pairs for accurate relevance scoring.

**BGE-Reranker** ([GitHub](https://github.com/FlagOpen/FlagEmbedding))
- [bge-reranker-base](https://huggingface.co/BAAI/bge-reranker-base) - 278M params, fast
- [bge-reranker-large](https://huggingface.co/BAAI/bge-reranker-large) - 560M params, high accuracy
- [bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3) - 568M params, multilingual (100+ languages)
- [bge-reranker-v2-gemma](https://huggingface.co/BAAI/bge-reranker-v2-gemma) - Gemma architecture

**Jina Reranker v2** ([HuggingFace](https://huggingface.co/jinaai/jina-reranker-v2-base-multilingual))
- 1024 token context, 100+ languages, code search support

**Mixedbread AI**
- [mxbai-rerank-base-v2](https://huggingface.co/mixedbread-ai/mxbai-rerank-base-v2) - 0.5B params (Qwen-2.5)
- [mxbai-rerank-large-v2](https://huggingface.co/mixedbread-ai/mxbai-rerank-large-v2) - 1.5B params, top BEIR scores

**MS MARCO Models**
- [ms-marco-MiniLM-L-12-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-12-v2) - Efficient
- [ms-marco-TinyBERT-L-6](https://huggingface.co/cross-encoder/ms-marco-TinyBERT-L-6) - Ultra-lightweight

### T5-Based Models

Sequence-to-sequence models leveraging T5 architecture for text ranking.

- **[MonoT5](https://huggingface.co/castorini/monot5-base-msmarco)** - Pointwise T5-base reranker fine-tuned on MS MARCO, scores documents independently.
- **[DuoT5](https://huggingface.co/castorini/duot5-3b-msmarco)** - Pairwise T5-3B reranker for comparing document pairs with O(n²) complexity.
- **[RankT5](https://github.com/castorini/rank_llm)** - T5 variant fine-tuned with specialized ranking losses for improved performance.
- **[PyTerrier T5](https://github.com/terrierteam/pyterrier_t5)** - T5-based reranking models integrated with PyTerrier IR platform.

### LLM-Based Models

Large language models adapted for reranking tasks with zero-shot or few-shot capabilities.

- **[RankLLM](https://github.com/castorini/rank_llm)** - Unified framework supporting RankVicuna, RankZephyr, and RankGPT with vLLM/SGLang/TensorRT-LLM integration.
- **[RankGPT](https://github.com/sunnweiwei/RankGPT)** - Zero-shot listwise reranking using GPT-3.5/GPT-4 with permutation generation.
- **[LiT5](https://github.com/castorini/rank_llm)** - Listwise reranking model based on T5 architecture.
- **[RankVicuna](https://github.com/castorini/rank_llm)** - Vicuna LLM fine-tuned for ranking tasks.
- **[RankZephyr](https://github.com/castorini/rank_llm)** - Zephyr-based model optimized for reranking.

## Commercial APIs

Production-ready reranking services with enterprise support and scalability.

- **[Cohere Rerank](https://docs.cohere.com/docs/reranking)** - Leading reranking API with multilingual support (100+ languages) and "Nimble" variant for low latency.
- **[Voyage AI Rerank](https://docs.voyageai.com/docs/reranker)** - Instruction-following rerankers (rerank-2.5/rerank-2.5-lite) with 200M free tokens.
- **[Jina AI Reranker API](https://jina.ai/reranker/)** - Cloud-hosted Jina reranker models with pay-as-you-go pricing.
- **[Pinecone Rerank](https://docs.pinecone.io/guides/rerank)** - Integrated reranking service within Pinecone's vector database platform.
- **[Mixedbread AI Reranker API](https://www.mixedbread.ai/api-reference/endpoints/reranking)** - API access to mxbai-rerank models with competitive pricing.
- **[NVIDIA NeMo Retriever](https://www.nvidia.com/en-us/ai/ai-enterprise-suite/nemo-retriever/)** - Enterprise-grade reranking optimized for NVIDIA hardware.

## Libraries & Frameworks

### Unified Reranking Libraries

- **[rerankers](https://github.com/AnswerDotAI/rerankers)** - Lightweight Python library providing unified API for all major reranking models (FlashRank, Cohere, RankGPT, cross-encoders).
- **[FlashRank](https://github.com/PrithivirajDamodaran/FlashRank)** - Ultra-lite (~4MB) reranking library with zero torch/transformers dependencies, supports CPU inference.
- **[Sentence-Transformers](https://www.sbert.net/docs/package_reference/cross_encoder/cross_encoder.html)** - Popular library for training and using cross-encoder reranking models.
- **[rank-llm](https://pypi.org/project/rank-llm/)** - Python package for listwise and pairwise reranking with LLMs.

### Specialized Tools

- **[FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding)** - BAAI's comprehensive toolkit for embeddings and reranking, includes BGE models and training code.
- **[PyTerrier](https://github.com/terrier-org/pyterrier)** - Information retrieval platform with extensive reranking support and experimentation tools.

## RAG Framework Integrations

### LangChain

Node postprocessors and document transformers for reranking in LangChain pipelines.

- **[Cohere Reranker](https://python.langchain.com/docs/integrations/retrievers/cohere-reranker/)** - Official Cohere integration using ContextualCompressionRetriever.
- **[FlashRank Reranker](https://python.langchain.com/docs/integrations/retrievers/flashrank-reranker/)** - Lightweight reranking without heavy dependencies.
- **[RankLLM Reranker](https://python.langchain.com/docs/integrations/document_transformers/rankllm-reranker/)** - LLM-based listwise reranking for LangChain.
- **[Cross Encoder Reranker](https://python.langchain.com/docs/integrations/document_transformers/cross_encoder_reranker/)** - Hugging Face cross-encoder models integration.
- **[Pinecone Rerank](https://python.langchain.com/docs/integrations/retrievers/pinecone_rerank/)** - Native Pinecone reranking support.
- **[VoyageAI Reranker](https://python.langchain.com/docs/integrations/document_transformers/voyageai-reranker/)** - Voyage AI models for document reranking.

### LlamaIndex

Postprocessor modules for enhancing retrieval in LlamaIndex query engines.

- **[CohereRerank](https://docs.llamaindex.ai/en/stable/examples/node_postprocessor/CohereRerank/)** - Top-N reranking using Cohere's API.
- **[SentenceTransformerRerank](https://docs.llamaindex.ai/en/stable/module_guides/querying/node_postprocessors/node_postprocessors/)** - Cross-encoder reranking from sentence-transformers.
- **[LLMRerank](https://docs.llamaindex.ai/en/latest/api_reference/postprocessor/llm_rerank/)** - Uses LLMs to score and reorder retrieved nodes.
- **[JinaRerank](https://docs.llamaindex.ai/en/stable/module_guides/querying/node_postprocessors/node_postprocessors/)** - Jina AI reranker integration.
- **[RankLLM Rerank](https://pypi.org/project/llama-index-postprocessor-rankllm-rerank/)** - RankLLM models as postprocessors.
- **[NVIDIA Rerank](https://pypi.org/project/llama-index-postprocessor-nvidia-rerank/)** - NVIDIA NeMo Retriever integration.

### Haystack

Ranker components for deepset's Haystack framework.

- **[CohereRanker](https://docs.haystack.deepset.ai/docs/cohereranker)** - Semantic reranking with Cohere models.
- **[SentenceTransformersRanker](https://docs.haystack.deepset.ai/docs/rankers)** - Cross-encoder based reranking.
- **[JinaRanker](https://haystack.deepset.ai/integrations/jina)** - Jina reranker models for Haystack pipelines.
- **[MixedbreadAIRanker](https://haystack.deepset.ai/integrations/mixedbread-ai)** - Mixedbread AI reranker integration.
- **[LostInTheMiddleRanker](https://docs.haystack.deepset.ai/docs/rankers)** - Optimizes document ordering to combat the "lost in the middle" phenomenon.

## Datasets & Benchmarks

### Training & Evaluation Datasets

- **[MS MARCO](https://microsoft.github.io/msmarco/)** - Large-scale passage and document ranking datasets with real Bing queries.
- **[MS MARCO Passage Ranking](https://microsoft.github.io/msmarco/Datasets)** - 8.8M passages with 500k+ training queries for passage retrieval.
- **[MS MARCO Document Ranking](https://microsoft.github.io/msmarco/Datasets)** - 3.2M documents for full document ranking tasks.
- **[BEIR](https://github.com/beir-cellar/beir)** - Heterogeneous benchmark with 18 diverse datasets for zero-shot evaluation.
- **[TREC Deep Learning Track](https://trec.nist.gov/data/deep.html)** - High-quality test collections (TREC-DL-2019, TREC-DL-2020) for passage/document ranking.
- **[TREC-DL-2019](https://trec.nist.gov/data/deep/2019.html)** - 200 queries with dense relevance judgments.
- **[TREC-DL-2020](https://trec.nist.gov/data/deep/2020.html)** - 200 queries with expanded corpus coverage.
- **[Natural Questions](https://ai.google.com/research/NaturalQuestions)** - Google's dataset of real user questions for QA and retrieval.
- **[SciRerankBench](https://arxiv.org/abs/2508.08742)** - Specialized benchmark for scientific document reranking.

### Benchmark Suites

- **[BEIR Benchmark](https://github.com/beir-cellar/beir)** - Zero-shot evaluation across 18 retrieval tasks (NQ, HotpotQA, FiQA, ArguAna, etc.).
- **[MTEB Reranking](https://github.com/embeddings-benchmark/mteb)** - Massive Text Embedding Benchmark including reranking tasks.

## Evaluation Metrics

Key metrics for assessing reranker performance:

- **[NDCG (Normalized Discounted Cumulative Gain)](https://en.wikipedia.org/wiki/Discounted_cumulative_gain)** - Standard metric emphasizing top results, commonly reported as NDCG@10.
- **[MRR (Mean Reciprocal Rank)](https://en.wikipedia.org/wiki/Mean_reciprocal_rank)** - Measures average inverse rank of first relevant result, used by MS MARCO (MRR@10).
- **[MAP (Mean Average Precision)](https://en.wikipedia.org/wiki/Evaluation_measures_%28information_retrieval%29#Mean_average_precision)** - Average precision across all relevant documents.
- **[Recall@K](https://en.wikipedia.org/wiki/Evaluation_measures_%28information_retrieval%29#Recall)** - Percentage of relevant documents in top-K results.
- **[Precision@K](https://en.wikipedia.org/wiki/Evaluation_measures_%28information_retrieval%29#Precision)** - Proportion of relevant documents in top-K results.

## Research Papers

### Foundational Papers

- **[Document Ranking with a Pretrained Sequence-to-Sequence Model](https://arxiv.org/abs/2003.06713)** (2020) - Introduces MonoT5 and DuoT5 for text ranking.
- **[BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models](https://arxiv.org/abs/2104.08663)** (2021) - Establishes BEIR benchmark suite.
- **[Is ChatGPT Good at Search? Investigating Large Language Models as Re-Ranking Agents](https://arxiv.org/abs/2304.09542)** (2023) - Introduces RankGPT for zero-shot LLM reranking.
- **[RankRAG: Unifying Context Ranking with Retrieval-Augmented Generation in LLMs](https://arxiv.org/abs/2407.02485)** (2024) - Unified framework for context ranking and answer generation.

### Recent Advances (2024-2026)

#### Cross-Encoder Innovations

- **[A Thorough Comparison of Cross-Encoders and LLMs for Reranking SPLADE](https://arxiv.org/abs/2403.10407)** (March 2024) - Comprehensive evaluation on TREC-DL and BEIR showing traditional cross-encoders remain competitive against GPT-4 while being more efficient.
- **[Set-Encoder: Permutation-Invariant Inter-Passage Attention for Listwise Passage Re-Ranking with Cross-Encoders](https://arxiv.org/abs/2404.06912)** (April 2024, ECIR 2025) - Novel cross-encoder architecture with inter-passage attention for efficient listwise reranking, achieving state-of-the-art results while maintaining permutation invariance.
- **[Don't Forget to Connect! Improving RAG with Graph-based Reranking](https://arxiv.org/abs/2405.18414)** (May 2024) - Introduces G-RAG, a GNN-based reranker that leverages document connections and semantic graphs, outperforming state-of-the-art approaches with smaller computational footprint.
- **[CROSS-JEM: Accurate and Efficient Cross-encoders for Short-text Ranking Tasks](https://arxiv.org/abs/2409.09795)** (September 2024) - Novel joint ranking approach achieving 4x lower latency than standard cross-encoders while maintaining state-of-the-art accuracy through Ranking Probability Loss.
- **[Efficient Re-ranking with Cross-encoders via Early Exit](https://dl.acm.org/doi/10.1145/3726302.3729962)** (2024, SIGIR 2025) - Introduces early exit mechanisms for cross-encoders to improve inference efficiency without sacrificing accuracy.

#### LLM-Based Reranking

- **[CompRank: Efficient LLM Reranking via Token-Level Compression and Decoding-Free Scoring](https://arxiv.org/abs/2606.11700)** (June 2026) - Combines reusable document-side representations, segment-wise token compression, and attention-based scoring to reduce computation on long candidate lists without autoregressive decoding.
- **[FIRST: Faster Improved Listwise Reranking with Single Token Decoding](https://arxiv.org/abs/2406.15657)** (June 2024) - Accelerates LLM reranking inference by 50% using output logits of first generated identifier while maintaining robust performance across BEIR benchmark.
- **[InsertRank: LLMs can reason over BM25 scores to Improve Listwise Reranking](https://arxiv.org/abs/2506.14086)** (June 2025) - Demonstrates consistent gains by injecting BM25 scores into zero-shot listwise prompts across Gemini, GPT-4, and Deepseek models.
- **[JudgeRank: Leveraging Large Language Models for Reasoning-Intensive Reranking](https://arxiv.org/abs/2411.00142)** (October 2024) - Agentic reranker using Chain-of-Thought reasoning with query analysis, document analysis, and relevance judgment steps, excelling on BRIGHT benchmark.
- **[Do Large Language Models Favor Recent Content? A Study on Recency Bias in LLM-Based Reranking](https://arxiv.org/abs/2509.11353)** (September 2024, SIGIR-AP 2025) - Reveals significant recency bias across GPT and LLaMA models, with fresh passages promoted by up to 95 ranks and date injection reversing 25% of preferences.

#### RAG & Production Systems

- **[HyperRAG: Enhancing Quality-Efficiency Tradeoffs in Retrieval-Augmented Generation with Reranker KV-Cache Reuse](https://arxiv.org/abs/2504.02921)** (April 2025) - Achieves 2-3x throughput improvement for decoder-only rerankers through KV-cache reuse while maintaining high generation quality.
- **[DynamicRAG: Leveraging Outputs of Large Language Model as Feedback for Dynamic Reranking in Retrieval-Augmented Generation](https://arxiv.org/abs/2505.07233)** (May 2025, NeurIPS 2025) - RL-based agent that dynamically adjusts both order and number of retrieved documents, achieving state-of-the-art results across seven knowledge-intensive datasets.
- **[SciRerankBench: Benchmarking Rerankers Towards Scientific RAG-LLMs](https://arxiv.org/abs/2508.08742)** (August 2025) - Specialized benchmark for scientific document reranking with emphasis on effectiveness-efficiency tradeoffs.

#### Test-Time Compute & Advanced Techniques

- **[Rank1: Test-Time Compute for Reranking in Information Retrieval](https://arxiv.org/abs/2502.18418)** (February 2025, CoLM 2025) - First reranking model leveraging test-time compute with reasoning traces, distilled from R1/o1 models with 600K+ examples, achieving state-of-the-art on reasoning tasks.
- **[How Good are LLM-based Rerankers? An Empirical Analysis](https://arxiv.org/abs/2508.16757)** (August 2025) - Comprehensive empirical evaluation comparing state-of-the-art LLM reranking approaches across multiple benchmarks and dimensions.

#### Surveys & Analysis

- **[The Evolution of Reranking Models in Information Retrieval: From Heuristic Methods to Large Language Models](https://arxiv.org/abs/2512.16236)** (December 2024) - Comprehensive survey tracing reranking evolution from cross-encoders to LLM-based approaches, covering architectures and training objectives.
- **[C-Pack: Packaged Resources To Advance General Chinese Embedding](https://arxiv.org/abs/2309.07597)** (2023) - Introduces BGE reranking model family and training methodologies.

### Survey Papers

- **[Pretrained Transformers for Text Ranking: BERT and Beyond](https://arxiv.org/abs/2010.06467)** (2020) - Survey of neural ranking models.
- **[Neural Models for Information Retrieval](https://arxiv.org/abs/1705.01509)** (2017) - Foundational survey of neural IR approaches.

## Tutorials & Resources

### Comprehensive Guides

- **[Top 7 Rerankers for RAG](https://www.analyticsvidhya.com/blog/2025/06/top-rerankers-for-rag/)** - Analytics Vidhya's comparison of leading reranking models.
- **[Comprehensive Guide on Reranker for RAG](https://www.analyticsvidhya.com/blog/2025/03/reranker-for-rag/)** - In-depth tutorial on implementing rerankers in RAG systems.
- **[Improving RAG Accuracy with Rerankers](https://www.infracloud.io/blogs/improving-rag-accuracy-with-rerankers/)** - Practical guide with implementation examples.
- **[Mastering RAG: How to Select A Reranking Model](https://galileo.ai/blog/mastering-rag-how-to-select-a-reranking-model)** - Selection criteria and comparison framework.

### Implementation Tutorials

- **[Boosting RAG: Picking the Best Embedding & Reranker Models](https://www.llamaindex.ai/blog/boosting-rag-picking-the-best-embedding-reranker-models-42d079022e83)** - LlamaIndex guide with benchmarks.
- **[Advanced RAG: Evaluating Reranker Models using LlamaIndex](https://akash-mathur.medium.com/advanced-rag-enhancing-retrieval-efficiency-through-evaluating-reranker-models-using-llamaindex-3f104f24607e)** - Step-by-step evaluation tutorial.
- **[Enhancing Advanced RAG Systems Using Reranking with LangChain](https://medium.com/@myscale/enhancing-advanced-rag-systems-using-reranking-with-langchain-523a0b840311)** - LangChain implementation patterns.
- **[Training and Finetuning Reranker Models with Sentence Transformers v4](https://huggingface.co/blog/train-reranker)** - Official Hugging Face training guide.
- **[Fine-Tuning Re-Ranking Models for LLM-Based Search](https://www.rohan-paul.com/p/fine-tuning-re-ranking-models-for)** - Domain-specific fine-tuning techniques.

### Video Tutorials

- **[Implementing Rerankers in Your AI Workflows](https://blog.n8n.io/implementing-rerankers-in-your-ai-workflows/)** - n8n's practical workflow tutorial.
- **[Cohere Rerank on LangChain Integration Guide](https://docs.cohere.com/docs/rerank-on-langchain)** - Official Cohere tutorial.

### Blog Posts & Articles

- **[Rerankers in RAG](https://medium.com/@avd.sjsu/rerankers-in-rag-2f784fc977f3)** - Conceptual overview of reranking in RAG pipelines.
- **[Sentence Embeddings: Cross-encoders and Re-ranking](https://osanseviero.github.io/hackerllama/blog/posts/sentence_embeddings2/)** - Technical deep-dive into cross-encoder architectures.
- **[The Four Types of Passage Reranker in RAG](https://medium.com/@autorag/the-four-types-of-passage-reranker-in-rag-02c907b4f747)** - Classification and comparison of reranker types.
- **[RAG in 2025: From Quick Fix to Core Architecture](https://medium.com/@hrk84ya/rag-in-2025-from-quick-fix-to-core-architecture-9a9eb0a42493)** - Industry trends and best practices.
- **[Boosting Your Search and RAG with Voyage's Rerankers](https://blog.voyageai.com/2024/03/15/boosting-your-search-and-rag-with-voyages-rerankers/)** - Voyage AI's technical blog.

## Tools & Utilities

### Evaluation Tools

- **[ranx](https://github.com/AmenRa/ranx)** - Fast IR evaluation library supporting NDCG, MAP, MRR, and more.
- **[ir-measures](https://github.com/terrierteam/ir_measures)** - Comprehensive IR metrics library with TREC integration.
- **[MTEB](https://github.com/embeddings-benchmark/mteb)** - Massive Text Embedding Benchmark for systematic evaluation.

### Development Tools

- **[Haystack Studio](https://haystack.deepset.ai/overview/haystack-studio)** - Visual pipeline builder with reranking components.
- **[LangSmith](https://www.langchain.com/langsmith)** - Debugging and monitoring for LangChain pipelines including rerankers.
- **[AutoRAG](https://github.com/Marker-Inc-Korea/AutoRAG)** - Automated RAG optimization including reranker selection.

### Visualization Tools

- **[Text Embeddings Visualization](https://projector.tensorflow.org/)** - TensorFlow's embedding projector for understanding model behavior.
- **[Phoenix](https://github.com/Arize-ai/phoenix)** - LLM observability platform with retrieval tracing.

## Reranker Leaderboard

📊 **[View Live Leaderboard](https://agentset.ai/rerankers)** - Compare rerankers using ELO scoring, nDCG@10, latency, and cost

Models ranked by head-to-head GPT-5 comparisons across financial, scientific, and essay datasets.

**Current Leaders (as of Nov 2025):**
- **[Zerank 2](https://agentset.ai/rerankers/zerank-2)** - Wins most head-to-head matchups, highest consistency across domains
- **[Cohere Rerank 4 Pro](https://agentset.ai/rerankers/cohere-rerank-4-pro)** - Second best win rate, strong performance on complex queries
- **[Voyage AI Rerank 2.5](https://agentset.ai/rerankers/voyage-ai-rerank-25)** - Balanced accuracy and response times

Rankings update as new models are evaluated.

## Related Awesome Lists

- **[Awesome RAG](https://github.com/tholman/awesome-rag)** - Comprehensive RAG resources and frameworks.
- **[Awesome LLM](https://github.com/Hannibal046/Awesome-LLM)** - Large Language Models resources and tools.
- **[Awesome Information Retrieval](https://github.com/harpribot/awesome-information-retrieval)** - IR papers, datasets, and tools.
- **[Awesome Embedding Models](https://github.com/Hannibal046/Awesome-Embedding-Models)** - Vector embeddings and similarity search.
- **[Awesome Neural Search](https://github.com/currentslab/awesome-neural-search)** - Neural search and dense retrieval resources.
- **[Awesome Vector Search](https://github.com/currentslab/awesome-vector-search)** - Vector databases and search engines.

## Contributing

Contributions are welcome! Please read the [contribution guidelines](CONTRIBUTING.md) first.

To add a new item:
1. Search previous suggestions before making a new one
2. Make an individual pull request for each suggestion
3. Use the following format: `**[Name](link)** - Description.`
4. New categories or improvements to the existing categorization are welcome
5. Keep descriptions concise and informative
6. Check your spelling and grammar
7. Make sure your text editor is set to remove trailing whitespace

## License

[![CC0](https://mirrors.creativecommons.org/presskit/buttons/88x31/svg/cc-zero.svg)](https://creativecommons.org/publicdomain/zero/1.0/)

To the extent possible under law, the contributors have waived all copyright and related rights to this work.

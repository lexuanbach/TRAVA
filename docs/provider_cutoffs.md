# Provider date evidence

Verified 15 November 2026. Provider dates describe different concepts and do not establish membership of particular benchmark items in training.

| Model | Date and meaning | Primary record |
|---|---|---|
| Haiku 4.5 | Overall training data July 2025, reliable knowledge February 2025 | [Model specifications](https://platform.claude.com/docs/en/models/haiku-4-5/overview) |
| Haiku 4.5 | Public-Internet training data February 2025 | [October 2025 system card, p.5](https://assets.anthropic.com/m/99128ddd009bdcb/original/Claude-Haiku-4-5-System-Card.pdf) |
| Sonnet 4.5 | Training data July 2025, reliable knowledge January 2025 | [Anthropic transparency](https://www.anthropic.com/transparency) |
| GPT-4o-mini | Knowledge cutoff 1 October 2023 | [Model documentation](https://platform.openai.com/docs/models/gpt-4o-mini) |
| Llama-3-8B | Pretraining data March 2023 | [Meta card](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) |
| Llama-3.1 controls | Pretraining data December 2023 | [Meta card](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) |
| Llama-3.2 controls | Pretraining data December 2023 | [Meta card](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) |

Qwen2.5 and Mistral-Large-2 have no disclosed cutoff in the records used here. The original nine-model cutoff panel contains seven such models. Llama and GPT-4o-mini are additional controls. `pipeline/verified_slice.py` records the main-panel evidence, while `pipeline/controlled_cutoff.py` records the Llama controls. `pipeline/cutoff_relative.py` uses the later Haiku overall date and preserves the February Internet-data date as a separate sensitivity result. The analyst sensitivity band is not a provider-certified confidence interval.

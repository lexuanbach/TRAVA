# Data attribution and terms

The MIT license in `LICENSE` applies to TRAVA's code and original documentation. It does not grant rights to third-party datasets, article excerpts, model checkpoints or provider-generated outputs.

## RealTimeQA

Source: [realtimeqa/realtimeqa_public](https://github.com/realtimeqa/realtimeqa_public). The authors' [NeurIPS 2023 datasheet](https://papers.nips.cc/paper_files/paper/2023/file/9941624ef7f867a502732b5154d30cb7-Supplemental-Datasets_and_Benchmarks.pdf) states CC BY-SA 4.0 for the dataset. Retain upstream attribution and applicable share-alike terms for question/annotation derivatives.

News article bodies and excerpts are separate third-party content. The datasheet describes fair-use collection; that is not a blanket license from the article publishers. The artifact contains article text and excerpts for reproducibility, including corpus files, annotation CSVs and model-label inputs. Redistribution permission for this text has not been independently established by this package. Public availability does not establish additional redistribution rights; downstream users must assess the terms applicable to their intended use. Do not describe the whole repository as MIT-licensed data.

## TRAVA annotations and cached model outputs

Anonymous annotator IDs A/B/C identify response columns, not named individuals. Blinding metadata is supplied to reproduce estimates, and must be withheld from raters in any future blinded study. Existing annotations are not a new 600-character annotation collection.

Model responses and classifier outputs are frozen scientific observations. The code license does not override the original model or service terms. Checkpoint names and provider snapshots are recorded in `configs/` and `results/environment_manifest.json`; weights and credentials are not distributed.

## Citation

Cite TRAVA using `CITATION.cff`, and cite the upstream dataset:

Jungo Kasai et al. *RealTime QA: What's the Answer Right Now?* Advances in Neural Information Processing Systems, 2023. [Dataset repository](https://github.com/realtimeqa/realtimeqa_public).

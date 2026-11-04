# CodeEffiJudge analysis package

This archive contains the 2,700 labeled program pairs, their displayed code, the model judgments and the data and scripts used to analyze those judgments. All analyses below start from the saved execution-cost labels. The test inputs, expected program outputs, per-run measurements and native timing audit are outside the scope of this archive. Independently rerunning the benchmark programs or deriving the labels from repeated executions requires those materials. We plan to make the full replication package public upon acceptance.

## Reproduce the reported analyses

Extract this ZIP and open the directory containing this README. Use Python 3.11 or later. To check the included records, run `python scripts/data_tools.py verify`. To reproduce the main results from the saved judgments, run `python scripts/paper_tables.py`. Install the dependencies for the statistical analyses, plots and optional feature extraction with `python -m pip install -r scripts/requirements.txt`. The analysis commands read local files.

The root `index.json` lists the files, SHA-256 checksums, record keys and links to the paper's results.

| Paper result | Command or source |
| --- | --- |
| Table I and Figure 1 | Related studies and an illustrative example in the manuscript. |
| Table II and Figure 2 | The paper describes the measurement protocols. The pair table supplies the resulting labels, and `scripts/filter_and_sample.py` implements selection from a candidate pool. |
| Table III | `python scripts/paper_tables.py` → `dataset` counts the displayed sources and measured implementations separately. |
| Table IV | `data/model_settings.json` and `data/prompts.json` describe the inference settings. |
| Tables V and VII | `python scripts/bootstrap.py` computes point estimates and paired intervals for models and prompts. `python scripts/paper_tables.py` also prints the point estimates. |
| Table VI | `python scripts/data_tools.py baseline` reconstructs pair-level rule predictions. `python scripts/bootstrap.py` computes their scores and intervals. |
| Figure 3 and RQ2 | `python scripts/plot_model_orders.py --output ../models.pdf` draws the two radar panels from the 54 model/language/prompt cells. `paper_tables.py` also returns `language_by_prompt` and `language_by_ratio`. |
| Table VIII and RQ3 | `python scripts/perturbation_statistics.py` computes the five edits. `python scripts/paper_tables.py` → `unedited_repeat` reproduces the separately discussed control with identical pair/side weights. |
| Figure 4 | `python scripts/plot_ppl.py --output ../ppl.pdf` renders the perplexity results. `python scripts/recompute_ppl.py --output ../associations.json` recomputes the associations from saved PPL scores. |
| Table IX | `python scripts/paper_tables.py` → `complexity_by_model` and `complexity`. |
| Table X and Figure 5 | `python scripts/paper_tables.py` → `error_categories` and `caller_guard_result_id`. Look up the case ID in `data/main_judgments.jsonl.gz` and `data/error_analysis.jsonl.gz`. |
| Full integrity and linkage check | `python scripts/data_tools.py verify` checks file hashes, primary keys, links, both orders and coverage. |

To reconstruct a single prompt, run `python scripts/data_tools.py prompt --language cpp --pair pie_41431 --mode cot --direction forward`. `filter_and_sample.py` implements selection given a candidate pool. The distributed pairs are the final evaluation set. `extract_features.py` implements the static features used by the rules.

## Identifiers and units

- A task is identified by `(language, problem_id)`. A pair is identified by `(language, pair_id)`. The source catalog uses `(language, code_id)` where `code_id` is the SHA-256 of the exact source bytes. Each pair links to two displayed sources and two measured sources. Each measured implementation within a task is identified by `implementation_id`.
- Main judgments use one `result_id` per model, language, pair, prompt `mode` and presentation `direction`. In the forward order A is the slower program. The reverse order swaps A and B. An unparsed or missing main judgment counts as incorrect. Accuracy counts ordered answers. Both-Correct counts pairs whose two orders are both correct.
- The main judgment field `inference_route` records local or hosted execution. Qwen3.8-27B used both routes for CoT (3,505 local and 1,895 hosted slots), and its perplexity was scored locally. The other five judges used hosted APIs.
- Each unedited repeat links to exactly one original main judgment through `result_id`. Each perturbation has one `variant_id` within its language and two ordered judgments. The Table VIII comparison set contains one row per `(language, pair_id, site)`. Several eligible edit sites can give the same base pair a weight greater than one.
- The pair table records the measured slow-to-fast cost ratio in `speedup`. The label metric is CPU instruction count for Python, elapsed time for Java, and the PIE cost for C++.
- The familiarity analysis uses `code_key` in the form `language:code_id`. Error categories C1–C6 follow the order in Table X. U denotes an unresolved response. Displayed source and available response text are kept as observed. All 4,455 displayed programs are present. For 251 additional measured-source IDs, the review data retain the SHA-256 and omit source text for anonymity. The pair table references these IDs in its measured-source fields.

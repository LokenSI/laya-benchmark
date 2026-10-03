# Local Laya benchmark — business findings

Run status: **complete**. Measured 2026-09-26T07:41:44.086858+00:00.

These are local measurements, not the model publisher's scores. Public test-set accuracy estimates performance on these tasks; it does not establish accuracy on your company's data.

| Test | Model | Examples | Accuracy | 95% interval* | Macro F1 | Single request p50 |
|---|---|---:|---:|---|---:|---:|
| English request routing | laya | 2948 | 66.6% | 64.9%–68.3% | 0.690 | 30.0 ms |
| Norwegian Bokmål request routing | laya | 2948 | 21.1% | 19.5%–22.5% | 0.256 | 30.7 ms |
| Norwegian review sentiment | laya | 1173 | 53.4% | 50.5%–56.4% | 0.470 | 30.3 ms |
| English request routing | laya-multilingual | 2948 | 61.6% | 59.8%–63.4% | 0.613 | 26.2 ms |
| Norwegian Bokmål request routing | laya-multilingual | 2948 | 51.3% | 49.5%–53.2% | 0.525 | 28.4 ms |
| Norwegian review sentiment | laya-multilingual | 1173 | 59.3% | 56.3%–62.2% | 0.534 | 29.5 ms |

*Source-document cluster bootstrap, 2,000 resamples. NoReC sentences from one review are resampled together. Intervals cover sampling uncertainty, not training-data contamination or deployment drift.

## Does this beat a simpler solution?

The comparison below is a supervised word/character TF-IDF + linear SVM trained on official training data. Laya receives no task-specific training. This tests the practical alternative when labelled training data is available, not equal training conditions.

| Test | Majority baseline | Simple trained classifier |
|---|---:|---:|
| English request routing | 13.6% | 89.7% |
| Norwegian Bokmål request routing | 13.6% | 89.1% |
| Norwegian review sentiment | 50.3% | 64.4% |

## Norwegian benefit from the multilingual checkpoint

Both checkpoints receive the same texts and prompts. Positive differences favor Laya Multilingual.

| Test | Difference in accuracy | Paired 95% interval |
|---|---:|---|
| English request routing | -5.0 percentage points | -6.9 to -3.1 percentage points |
| Norwegian Bokmål request routing | +30.2 percentage points | +28.1 to +32.3 percentage points |
| Norwegian review sentiment | +6.0 percentage points | +2.0 to +9.4 percentage points |

## Can it safely take work away from people?

A policy is selected on a separate validation subset after calibration. It must have at least 30 accepted validation cases and a 95% Wilson lower accuracy bound of at least 95%. It is then evaluated once on the test set. This is evidence for a pilot, not a production guarantee.

| Model / test | Selected threshold | Automated share of test | Accuracy on automated cases | 95% interval |
|---|---:|---:|---:|---|
| laya / English request routing | None qualified | 0.0% | n/a | n/a |
| laya / Norwegian Bokmål request routing | None qualified | 0.0% | n/a | n/a |
| laya / Norwegian review sentiment | None qualified | 0.0% | n/a | n/a |
| laya-multilingual / English request routing | None qualified | 0.0% | n/a | n/a |
| laya-multilingual / Norwegian Bokmål request routing | None qualified | 0.0% | n/a | n/a |
| laya-multilingual / Norwegian review sentiment | None qualified | 0.0% | n/a | n/a |

## What a software product can do with this

Route incoming messages, tag feedback, or propose a queue to a support agent. Local inference can keep message content inside the organization's environment after the initial downloads. A user interface can show the proposed category, allow correction, and save that correction as future labelled training data.

Start with suggestions that a person reviews. Enable automatic actions only on a separately validated in-domain workflow. A classifier's output should not itself trigger payments, account deletion, or other irreversible actions.

Value comes from reduced handling time minus the cost of errors, hosting, monitoring, integration, and ongoing maintenance. The HTML report contains an editable NOK calculator. Its amounts are scenarios, not measured savings; local software still has infrastructure and staffing costs.

## Limits that affect the decision

- MASSIVE tests 18 assistant-service categories, not support tickets or all 60 intents. English and Bokmål test IDs are matched. This is a routing proxy.
- NoReC tests three-way sentiment on published reviews. It contains Norwegian varieties but this version has no row-level variety tag, so no separate Bokmål/Nynorsk public score is claimed.
- Separate authored diagnostics cover Bokmål, Nynorsk, negation, quotes, typos, code switching, binary decisions, ordinal urgency, option order, and long inputs. They need native-speaker review and are not pooled into headline accuracy.
- Public benchmarks may have appeared during pretraining or fine-tuning. No contamination-free claim is possible without private, newly labelled data.
- Calibration uses the SDK's four-decimal probabilities, renormalized. NLL clips zero probabilities at 1e-8. ECE uses ten equal-width bins; low ECE alone does not mean good accuracy.
- Prompts are frozen, English by default, with no test-set tuning. Native checkpoint token budgets are retained and all truncation is counted. SDK-shipped/clamped temperatures remain in the raw results.
- Request latency is warm, batch-one inference on this machine. Batch throughput, model load time, downloads, and network/service overhead are separate.

## Evidence

See `summary.json` for per-class precision/recall/F1, confusion matrices, calibration, thresholds, sampling checks, and paired comparisons. `predictions.jsonl` records every measured decision; `errors.csv` lists mistakes. Both raw exports contain source text and should stay local, especially NoReC's CC-BY-NC material.

Sources: [Laya](https://huggingface.co/convaiinnovations/laya), [Laya Multilingual](https://huggingface.co/convaiinnovations/laya-multilingual), [MASSIVE](https://huggingface.co/datasets/AmazonScience/massive), [NoReC sentence](https://huggingface.co/datasets/ltg/norec_sentence). Exact revisions and file hashes are saved with the results.

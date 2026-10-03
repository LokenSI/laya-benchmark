# Evaluation protocol, version 1

## Primary questions, fixed before the scored run

1. How accurate are the English and multilingual checkpoints on all 18 MASSIVE service categories in English and Bokmål?
2. How accurate are they on three-class Norwegian sentence sentiment?
3. Does multilingual improve Norwegian accuracy on the same examples?
4. Do probabilities support a useful high-accuracy subset after independent calibration?
5. Is either checkpoint better than a majority baseline and a simple supervised text classifier?
6. What latency and throughput does this local machine actually achieve?

English Laya on Norwegian is an explicit out-of-language control, not an endorsed deployment. The multilingual model also runs English to quantify the trade-off. Checkpoints are always selected directly.

## Sources, splits and leakage

Data is fetched as pinned Parquet files without executing remote dataset scripts. `sources.json` records immutable revisions, including the MASSIVE Parquet conversion revision and its upstream main revision. File SHA-256 digests are recorded after download. Model weights are pinned and hashed too.

For each suite, deduplicate normalized text (casefolded, collapsed whitespace), first within test, then remove validation overlaps with test, then remove training overlaps with both. Test data has priority. All removed counts are reported. This removes exact normalized duplicates, not semantic near-duplicates. Test labels do not guide removal or sample selection.

MASSIVE en-US and nb-NO test and validation IDs are intersected after cleaning. The two languages therefore use matched cases with identical labels. All 18 labels remain available at inference; no gold-conditioned distractor sampling is used. The source task has 60 intents; those fine-grained intents are outside this benchmark's scope.

NoReC ternary mapping is explicitly `0=negative`, `1=positive`, `2=neutral`. The prefix of each sentence ID identifies the source review. Review-group overlap across training, validation and test raises an error. Sentences are not tagged as Bokmål or Nynorsk in this export; the public score is labelled Norwegian, not either variety specifically.

`--test-size 0` and `--validation-size 0` use the complete cleaned pools. Capped runs use SHA-256 of seed plus ID to order and sample without reference to labels. Different model runs read the same prepared data file. Raw text is the only state sent to Laya: IDs, source labels, split names and group IDs are never supplied as model features.

## Frozen inference conditions

SDK: Laya 0.3.20. Native context/head token budgets, eager PyTorch execution, one checkpoint at a time. No model fine-tuning, no test-set prompt choice, no SDK router, no external judge. Default English questions are identical across models and languages. Full prompts are in `laya_bench/questions.py` and hashed with the code.

The SDK's shipped calibration settings remain applied. In particular, the English checkpoint's choice:11+ temperature is clamped by SDK 0.3.20 to its supported minimum. “Raw” in the report means **SDK output before this benchmark's extra validation-fitted temperature**, not unmodified network logits. Applied temperatures and full checkpoint configuration are recorded.

The SDK returns probabilities rounded to four decimals; these are renormalized. Choice predictions preserve the SDK label to avoid changing a prediction when rounding creates a tie. Binary `noul` is interpreted as P(true), threshold 0.5. `score` diagnostic class is the modal ordinal level, not the rounded expected score; expected-score MAE is reported separately. Entropy-based `confidence` and `act_probability` are never used as correctness probabilities.

All input truncation, shortened option descriptions and shortened instructions are audited using the pinned SDK's sequence builder and tokenizer. Native budgets are part of the default deployment comparison. More generous settings appear only in labelled long-input diagnostics.

## Metrics and uncertainty

- Accuracy: correct / evaluated examples, with no silent omission of mistakes.
- Macro F1: unweighted average over the declared class set, including zero for a class with no predicted positives.
- Balanced accuracy: mean per-class recall over observed gold classes.
- Confusion matrix: actual classes in rows, predicted classes in columns.
- Multiclass Brier: mean sum of squared probability errors across all classes; range 0–2.
- NLL: mean negative natural logarithm of gold probability, clipped at 1e-8 for rounded zero outputs.
- ECE: ten equal-width confidence bins weighted by example count; confidence is probability of the predicted class, not entropy. Confidence 1 belongs in the final bin.
- Primary accuracy interval: percentile cluster bootstrap, 2,000 resamples of source review / example groups, seed 20260926. Every sentence of a sampled review is retained together. Wilson intervals are also reported for transparency but assume independent Bernoulli observations.
- Paired differences: per-ID correctness difference, bootstrapped by source group. Positive means multilingual beats English checkpoint; for language gaps positive means Bokmål beats English input. Exact McNemar p-values are exploratory, unadjusted, and assume independence; clustered intervals take priority on NoReC.

Intervals on a complete public split express uncertainty about analogous underlying examples; the observed score on that finite split is exact. They do not cover pretraining contamination, human-label ambiguity, alternate prompts, unsupported dialects, or future distribution shift. No single score pools tasks or languages.

## Calibration and human-review policy

Validation source groups are hash-ordered and divided into two disjoint halves. A scalar temperature in [0.25, 10] is fitted by NLL on the first half, per model and suite. It rescales the logarithm of SDK probabilities and preserves the category decision except numerical ties, for which the original SDK decision is retained.

The second validation half selects a threshold from the predeclared grid 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99. Choose maximum coverage among thresholds with at least 30 accepted examples and a 95% Wilson lower accuracy bound of at least 0.95. If none qualifies, accept zero cases and report accuracy as unavailable. This screening condition is not a simultaneous statistical guarantee across searched thresholds or correlated reviews. The untouched test set supplies the independent follow-up check.

The selected threshold is then fixed for the test set; report coverage, accuracy, interval, accepted mistakes, review volume and errors per 1,000 total requests. Fixed threshold curves are descriptive diagnostics, not thresholds retroactively selected on test. Even a passing public policy requires independent private-workflow validation.

## Baselines

Majority class is determined only from cleaned training labels. Uniform-random expected accuracy is 1/K. The supervised comparator combines word TF-IDF (1–2 grams, max 40,000 features) and character TF-IDF (3–5 grams, max 60,000), both min_df=2 and sublinear TF, with a linear SVM (C=1, max_iter=5000). It is trained independently per language/task using only the official training split. Hyperparameters are fixed, not tuned on final outcomes. It answers the practical question of whether a cheap task-specific classifier is sufficient.

## Diagnostics and business meaning

Thirty authored support scenarios each have English, Bokmål and Nynorsk versions. Questions measure five-way routing, refund detection as `noul` and neutral-key binary `choice`, and three-level urgency. Six additional contrast pairs per language balance positive/negative refund intent. The same Norwegian texts are also tested with Norwegian question wording. Primary public prompts are not changed based on diagnostic outcomes.

The first 100 public examples in deterministic order are rerun with reversed choice order. Long-input diagnostics place a refund request before or after 0/40/160 repetitions of neutral background, using native limits and an additional multilingual 2,048-token condition. Repetitions and translated cases are correlated and do not expand the independent headline sample size. Authored gold labels and Norwegian wording require human review; they are diagnostic starting points, not certified linguistic coverage.

Business interpretation uses routing as a proxy for queue assignment and sentiment as a proxy for feedback tagging. It does not assert these datasets match real support queues. Financial inputs in the HTML report are explicitly hypothetical. Error cost must reflect the actual consequence of a wrong decision. Saved time must be measured in a pilot; the formula must not double-count the same labor as both saved and correction work.

## Performance and reproducibility

Warm up before scoring. Synchronize CUDA before and after each measured call. Batched timing covers tokenization, transfer, forward pass, and decoding within the SDK. Separate 50-example batch-one measurements provide p50/p95 latency; amortized batch duration is not reported as request latency. Logging and metric calculation are outside timed inference. Download time is excluded; load time and peak allocated GPU memory are separate. Shared machine activity and first-use effects can change latency.

Seeds, CPU thread count, package versions, GPU, OS, data hashes, model hashes, and code hashes are saved. Deterministic algorithms are requested with warnings for unsupported operations; small hardware-dependent numerical changes remain possible. Any SDK device fallback aborts the run so GPU and CPU measurements cannot be mixed. A partial or failed run is not a completed evaluation.

## Dataset terms and attribution

- [MASSIVE](https://huggingface.co/datasets/AmazonScience/massive): Amazon Science; CC-BY-4.0. Paired multilingual assistant requests.
- [NoReC sentence](https://huggingface.co/datasets/ltg/norec_sentence): Language Technology Group, University of Oslo, SANT project; CC-BY-NC-4.0. Norwegian review sentence sentiment.
- [Laya](https://huggingface.co/convaiinnovations/laya) and [Laya Multilingual](https://huggingface.co/convaiinnovations/laya-multilingual): Convai Innovations; model cards list Apache-2.0.

Raw datasets and prediction/error logs contain source text. Keep those files local unless the relevant redistribution terms are satisfied. Aggregate reports contain no source utterances.

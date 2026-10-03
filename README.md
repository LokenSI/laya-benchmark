# Decision-model benchmark: English and Norwegian

A local, reproducible benchmark that began with **convaiinnovations/laya** and **convaiinnovations/laya-multilingual**, then expanded to local decision models and a live Jev API reference. It measures classification quality, confidence reliability, speed, and the practical alternative of a simple trained classifier. It does not equate a publisher's confidence score with correctness.

**Start with [the published business summary and reports](https://lokensi.github.io/laya-benchmark/).** The latest update includes four Decision 2.0 models, 70,304 saved predictions and 9,600 timed concurrent HTTP requests. [Accuracy](https://lokensi.github.io/laya-benchmark/results/alternatives/decision2.html), [stress testing](https://lokensi.github.io/laya-benchmark/results/decision2-stress/report.html), and [the local-versus-API comparison](https://lokensi.github.io/laya-benchmark/results/jev_live/report.html) retain their distinct scopes.

For sharing: [one-page business PNG](https://lokensi.github.io/laya-benchmark/results/share/decision2-2026-10-03/business-summary.png), [English draft](results/share/decision2-2026-10-03/sharing-text-en.txt), and [Norwegian draft](results/share/decision2-2026-10-03/sharing-text-no.txt). On 2,948 public Norwegian routing-proxy cases, Nox 4B reached 80.3%, the tested hosted API 81.8%, and Decider 4B 82.0%. A supervised classifier reached 89.1% using labelled training data. These are measurements on public tests, not measured business savings or production acceptance criteria.

The original [interactive Laya business report](https://lokensi.github.io/laya-benchmark/results/full/report.html) and [written findings](results/full/business_report.md) remain available. Reports also work locally without a web service. [Publication scope](docs/PUBLICATION.md) explains which aggregate evidence is published; raw datasets, per-case payloads, weights and credentials remain local.

## What is tested

| Suite | Task | Scope |
|---|---|---|
| MASSIVE English | Route requests to 18 assistant services | Human-labelled official test split |
| MASSIVE Norwegian Bokmål | Same task, same example IDs | Direct comparison of both checkpoints and language gap |
| NoReC sentence | Negative / neutral / positive review sentiment | Human-labelled Norwegian official test split; intervals grouped by source review |
| Authored diagnostics | Support routing, refund detection, urgency | English, Bokmål, Nynorsk; separate from headline accuracy |
| Robustness checks | Negation, quotes, typos, code switching, option order, long input | Diagnostic evidence, not a production accuracy estimate |

The full run evaluates all retained public test examples: **2,948 English + 2,948 Bokmål + 1,173 Norwegian review sentences per model**. Another 5,356 validation examples per model are split by source group between calibration and threshold selection. Duplicates are removed before evaluation. Baselines use only the official, cleaned training splits.

These models return decisions such as categories and probabilities; this is not a speech-recognition, translation, question-answering, or text-generation benchmark. MASSIVE tests **18 scenarios, not all 60 intents**. Norwegian review data does not expose a per-row language-variety tag. Nynorsk results therefore come only from authored diagnostics, pending independent native-speaker review.

## Local setup (Windows / PowerShell)

Python 3.11 and a project-local `.venv` are used. The setup script installs CUDA 12.8 PyTorch for NVIDIA GPUs, including the RTX 50 series. It does not change global Python packages.

```powershell
cd D:\git\laya_benchmark
.\scripts\setup.ps1
.\scripts\benchmark.ps1
```

If your PowerShell policy blocks local scripts, use these direct commands instead:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe -m laya_bench prepare --test-size 0 --validation-size 0 --output data/prepared/full.json
.venv\Scripts\python.exe -m laya_bench run --data data/prepared/full.json --output results/my-run --device cuda
```

For CPU: `setup.ps1 -Cpu` and `benchmark.ps1 -Cpu -BatchSize 2`. A quicker functional run is `benchmark.ps1 -Quick`; its smaller sample **must not replace the full evaluation for a business decision**. Close other GPU-heavy workloads if necessary; the runner fails if the SDK silently switches from CUDA to CPU, so reported timing never mixes devices.

Model weights and data download once into `.cache/huggingface`. The English checkpoint uses only its root files; the separate multilingual repository is loaded explicitly. The automatic router is intentionally bypassed so it cannot conceal the checkpoint being evaluated. No Hugging Face account, hosted API, or inference key is required.

After downloading, run inference without network access:

```powershell
$env:HF_HUB_OFFLINE = '1'
.venv\Scripts\python.exe -m laya_bench run --data data/prepared/full.json --output results/offline-run --offline
```

Every run requires a new output folder to preserve evidence. If a run is interrupted, its partial summary is marked `running` or `failed`, never complete. Start a new output folder to retry. Regenerate presentation without rerunning models:

```powershell
.venv\Scripts\python.exe -m laya_bench report --input results/full/summary.json
.venv\Scripts\python.exe -m laya_bench verify --input results/full/summary.json
```

## Results and interpretation

### Decision 2.0 (October 3)

The vLLM Semantic Router Decision 2.0 release was not in the earlier results.
Four native variants are now integrated: Kai 0.6B, Eos 0.8B, Sol 2B and Nox 4B.
Their pinned release manifests and every packaged file hash are verified before
loading. The publisher's exact inference path, prompts, temperature and heads
are retained. A Windows portability shim canonicalizes fingerprint path names
to forward slashes and still checks the exact scored identity. Native overflow
rejections remain visible; score classes use modal levels and retain the native
expected scores. Lux 9B and Vega 27B are deferred for the local 16 GB GPU budget.

The sequential runner first evaluates all 231 public JevBench cases for each
model, then the exact 300-case pilot, fresh shared English/Norwegian cases, and
the full existing fixtures. All four native models have now completed all five
fixtures. Read [the final verification](results/alternatives/decision2-verification.json)
and [the Decision 2.0 report](https://lokensi.github.io/laya-benchmark/results/alternatives/decision2.html)
for completed coverage. The original supervisor retains its interrupted attempts;
`queue-decision2-retry.json` records the finished resumable retries.
This Windows/NVIDIA deployment does not reproduce the
publisher's ROCm latency measurements. Use a single GPU supervisor:

```powershell
.venv-decision\Scripts\python.exe -X utf8 -m laya_bench.decision2_queue
```

The [Decision 2.0 stress report](results/decision2-stress/report.html) separately
records direct compatibility probes in the pinned official vLLM 0.30.0 Linux
container, concurrent native HTTP requests at 1/8/32/128/512 clients, and
1/4/16/64/128/512 questions in one request. The native baseline has one FIFO GPU
executor and no continuous batching. It measures queueing and stability;
SDK submission capacity is different from simultaneous GPU forwards. All
timed request records, serial references, model revisions and errors are saved.
Run stress at a standard supervisor job boundary, with only one GPU owner:

```powershell
.venv\Scripts\python.exe -m laya_bench.decision2_stress prepare
.venv\Scripts\python.exe -m laya_bench.decision2_stress_queue --supervisor-pid <actual-supervisor-pid>
.venv\Scripts\python.exe -m laya_bench.decision2_stress_report --charts
.venv\Scripts\python.exe -m laya_bench.decision2_verify
```

The completed October run contains 70,304 saved predictions (17,576 cases per
native model) plus 9,600 timed concurrent HTTP responses. The
[independent verification](results/alternatives/decision2-verification.json)
checks case IDs, input hashes, answer keys, output schemas, saved prediction
hashes, model revisions and stress response consistency. Windows allocation
and kernel crashes required exact-runtime resumable retries; the original
events remain in the verification and supervisor logs. Many-question capacity
failures are retained separately from concurrent-request success.

### October local-model comparison against Jev

The [open-source leaderboard expansion](results/alternatives/leader-expansion.html) audits the union of the top ten overall and top ten Jev-class capability entries in JevBench v1.5.4. Decision 4B v1.2 and Winnow-12B Q8 have native integrations. Cygnet and Jev-Omni have separately named NF4/Transformers variants for the 16 GB GPU; these cannot verify or refute the published BF16 configurations. JevK5 v0.3, Plumb, Imajev and Decider are already covered. Rune v3 and djev are deferred for hardware/runtime/access constraints, with publisher license evidence retained.

The expansion checks publisher file hashes, runs separate integration probes, then inserts `jev_verified` and `jev_fresh` at an existing GPU job boundary. Full shared suites are queued after the existing follow-up work. The prior supervisor resumes automatically. Unsupported option counts and overlength inputs remain visible as rejections. No closed-source candidate is newly added, and no hosted GPU is provisioned. See `results/alternatives/queue-leaders-priority.json`, `leaders-execution-jobs.json` and the detailed license audit for execution evidence.

The [live Jev business comparison](results/jev_live/report.html) adds current API calls to the historical evidence below. The frozen run covers 17,576 shared requests, including 1,264 new records: four public task groups with 100 development and 200 test examples each, plus 32 paired English/Norwegian business scenarios. Whole tasks were selected from earlier local strengths; individual local successes were never used to choose cases. Spam and phishing are reported Laya training tasks, so results are not proof of unseen-domain security performance.

API requests use the resolved `jev-1.13.0` version. Stable and preview aliases resolved to the same version in integration probes. The report includes complete task groups only, paired uncertainty, failures, confusion matrices, calibration, and thresholds fitted on development data before application to test data. Raw responses and ordered request hashes are retained. The temporary credential is encrypted with Windows DPAPI under the ignored cache, or can be supplied through `TYPESAFE_API_KEY`; it is never written into source or reports. Billable token usage and conservative in-flight reservations enforce the authorized $25 ledger budget at the dated public price.

```powershell
.venv\Scripts\python.exe -X utf8 -m laya_bench.jev_live run --workers 4
.venv\Scripts\python.exe -X utf8 -m laya_bench.jev_live_report --charts
```

The runner resumes saved IDs. Do not start a second live runner: its ledger is protected by a process lock. Use the existing GPU supervisor for local runs. PNG/SVG graphics are in `results/jev_live/figures`; their footer states whether the comparison is complete or interim.

The [graphics gallery](results/jev_live/figures/index.html) includes the matched accuracy figures, technology designs, confidence gates and an explicitly illustrative NOK business-value scenario. A separate [response-time experiment](results/latency/protocol.json) uses 80 short messages (20 per task), three serial passes, eight warm-up calls, local batch size one and a persistent HTTPS API session. Its median and 95th-percentile timings describe this local deployment; they do not reproduce vendor H100 latency ratios. PNG and editable SVG exports share the same clean visual style.

The [claim audit](results/alternatives/claim_audit.html) distinguishes matching scores, near reproductions, incompatible checkpoint versions and inaccessible test sets. [Execution verification](results/jev_live/verification.json) reconciles ordered API requests, pinned versions and the shared usage ledger. Native abstentions remain unanswered in scores and review coverage.

The [verified Jev comparison](results/alternatives/jev_comparison.html) contains two separate checks:

- **231 public JevBench cases:** pinned source-file hashes, exact questions and option order, and published per-case outcomes for Jev 1.13.0. Local outputs use JevBench's original scoring. Paired accuracy differences keep supplied paraphrase groups together.
- **300 exact pilot cases:** 100 AG News, 100 DAIR Emotion and 100 Banking77/BTZSC examples, matching the published manifest hash. Original instructions, opaque option IDs, descriptions and order are preserved, including all 72 banking candidates. Published Jev accuracy is 91%, 48% and 87%, respectively. Per-case Jev pilot predictions are not public, so this comparison reports aggregate differences.

In this separate historical report, Jev is a **published API reference**. Current calls are in the live report above. Public case verification establishes source integrity and input equality; answer keys have not been independently relabelled. The main `jev_native` suite uses simpler label wording and is a separate prompt diagnostic. The sealed JevBench leaderboard and hosted/local latency are not directly comparable here.

```powershell
.venv\Scripts\python.exe -X utf8 -m laya_bench.alternatives_jev prepare
.venv-decision\Scripts\python.exe -X utf8 -m laya_bench.alternatives_run decider-2b --fixture jev_verified --batch-size 8
.venv\Scripts\python.exe -X utf8 -m laya_bench.alternatives_report --charts
```

Use the model's assigned venv and let the sequential supervisor own the GPU. The updated follow-up queue starts each model with `jev_fresh` and `jev_verified`, and completed prediction IDs are resumed without duplication. [Source verification](results/alternatives/jev_verification.json), [the exact-pilot protocol](results/alternatives/jev_verified-protocol.json), and [paired case outcomes](results/alternatives/jev_paired_cases.jsonl) make the comparison auditable. New cases are stored separately so the running frozen fixtures remain unchanged.

- `report.html`: interactive business report, category comparisons, reliability, diagnostics, and an editable NOK cost scenario.
- `business_report.md`: written findings and caveats for decision makers.
- `summary.json`: complete metrics, confusion matrices, per-class precision/recall/F1, confidence checks, paired comparisons, hardware, versions, hashes, and token-budget audits.
- `predictions.jsonl`: every individual model decision, probabilities, source ID, text, expected answer, and truncation audit.
- `errors.csv`: mistakes for manual inspection; spreadsheet formula-leading text is escaped.
- `verification.json`: independent reconciliation of saved predictions with frozen inputs and reported metrics.

Raw data and exports stay local and are excluded by `.gitignore`. The aggregate reports contain no dataset utterances. **Do not publish the raw NoReC texts as unrestricted commercial data**; its dataset card specifies CC-BY-NC-4.0. The benchmark tooling and model licensing do not replace dataset terms.

Accuracy answers “how often was the decision correct?” Macro F1 gives every category equal weight so frequent categories cannot conceal weak rare categories. Confidence calibration asks whether a purported 90% confidence actually corresponds to about 90% correctness. Confidence selection reports **both accuracy and coverage**; sending everything to human review is not counted as 100% model accuracy.

The simple TF-IDF + linear SVM comparator is supervised and has task-specific labels. Laya is zero-shot. This is a practical cost/quality comparison, not a claim of equal training conditions. Laya could still offer value when labels are unavailable, categories change, or multiple decisions are needed; that benefit must be established in the intended workflow.

## How to measure actual business value

Start by showing a suggested category to staff. Log corrections, time spent, and error consequences. For a production acceptance test, collect newly labelled private messages, split by customer/conversation and time, keep a private test set untouched, and have two Norwegian reviewers adjudicate disagreements. Include enough examples in every important category and each language variety. Use development data to tune; rerun final acceptance once.

Use the report's cost calculator to model:

`automated cases × time saved × staff cost − error cost − hosting/operations − integration amortization`

The example NOK amounts are assumptions, not market prices or measured savings. Local inference does not mean zero cost. Compare this with the existing manual workflow and simple classifier, then run a controlled pilot to measure actual savings.

No public benchmark can reveal a universal “true accuracy.” Public training contamination is unknown, confidence intervals capture only sampling variability, and domain shift can be large. The benchmark gives reproducible evidence and a way to reject bad assumptions rather than a production guarantee.

See [the evaluation protocol](docs/METHODOLOGY.md) for exact definitions and predeclared choices. Checkpoint and dataset revisions are in [sources.json](sources.json). The tested environment is recorded in [requirements-lock.txt](requirements-lock.txt) and each run's manifest.

Sources: [Laya](https://huggingface.co/convaiinnovations/laya), [Laya Multilingual](https://huggingface.co/convaiinnovations/laya-multilingual), [MASSIVE](https://huggingface.co/datasets/AmazonScience/massive), [NoReC sentence](https://huggingface.co/datasets/ltg/norec_sentence).

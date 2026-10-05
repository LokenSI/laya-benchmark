# Public benchmark package

**Unofficial, independent testing. Use as is, without warranty.** These results are not vendor-endorsed. Validate suitability on your own data before production use.

**Revision `completion-2026-10-04` — updated 5 October 2026.** The completion pass retains the original five frozen fixtures, prompts, model revisions and scoring. It resumes missing local results in fresh workers, retries only recorded infrastructure failures, and extends controlled timing coverage on the original fixed samples. All 31 local models have 17,576 recorded cases each. The dated completion report states answered coverage and remaining failures; a saved rejection is not a successful answer. Historical API attempts and earlier timing evidence remain preserved.

Repository: https://github.com/LokenSI/laya-benchmark

**Presentation revision `overview-2026-10-05` — 5 October 2026.** The opening page now leads with task-specific local accuracy leaders against Jev and an explorer covering every model and published scored group. It retains exact ties, includes unsuccessful requests in accuracy denominators, and supports sorting by controlled median response time and measured added VRAM. Timing is shown only for complete groups with at least 90% answered calls; missing values stay unavailable. Fastest and lowest-VRAM callouts state the model's accuracy and the number of measured candidates, without an accuracy filter. These descriptive rankings do not establish statistical superiority or a universal winner. The selected four-model sections and model-family spotlight are removed from the opening pages; the all-model comparison is the main entry point. No predictions, scoring, aggregate measurements or benchmark evidence hashes change in this presentation revision.

Published reports: https://lokensi.github.io/laya-benchmark/

`public-files.json` is the explicit allowlist for the Pages site. It contains aggregate reports, protocols, source/model hashes, verification summaries, historical process failures, and graphics. The historical supervisor's failure status is retained; the final verification establishes completed prediction coverage. Results and model/data caches are ignored by default. Only the reviewed publication files are committed from `results/`.

Raw dataset utterances, NoReC review texts, model weights, API request/response payloads, per-case prediction files, environment folders, credentials and cache files remain local. Report demos contain assistant-authored diagnostics, not private customer messages. The original report files are preserved; the Pages build labels links to unpublished raw evidence as local-only and adds navigation. Raw benchmark evidence can be reconstructed by following the fixture preparation and benchmark commands, subject to source availability and dataset/model terms.

The public metrics are not a production acceptance test or measured ROI. Accuracy on 18 MASSIVE assistant-service scenarios is a routing proxy, not an estimate for a company's support inbox. The supervised classifier uses labelled training data; zero-shot models do not use task-specific fitting in this benchmark. Similar scores do not establish statistical equivalence. Native FIFO load testing is distinct from vLLM compatibility and simultaneous GPU execution.

Build the public site without running inference:

```powershell
.venv\Scripts\python.exe -m laya_bench.business_summary
.venv\Scripts\python.exe scripts/build_pages.py --output .cache/pages-site
```

The historical share-card generator verifies complete source reports and records their SHA-256 hashes in the share package. It exports a 1080×1350 PNG, a 2160×2700 master, editable SVG and alt text. `build_pages.py` requires every public source to exist, builds the all-model landing page from the committed accuracy and controlled-timing aggregates, checks matching case counts and timing accuracy, checks local report links and writes an asset checksum manifest. The page works without JavaScript as a complete Norwegian routing table; JavaScript enables task selection and sorting. GitHub Actions repeats this check and deploys the static site on pushes to `main`.

Before regeneration, `scripts/verify_completion.py` checks protected predictions, archived infrastructure retries, at least three retained attempts for each remaining memory failure, unchanged historical timing evidence, the pinned Jev reference and removal of temporary restored weights. Its [aggregate verification](../results/completion/final-verification.json) publishes counts and hashes, without case payloads. The Pages build rejects mismatches between those verified prediction hashes and the completion report.

On a clean clone, the committed share card and aggregate results are sufficient to build the site with standard Python. Regenerating results requires datasets and models to be prepared locally. Model and dataset licences apply separately; this repository does not redistribute them.

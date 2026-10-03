# Public benchmark package

Repository: https://github.com/LokenSI/laya-benchmark

Published reports: https://lokensi.github.io/laya-benchmark/

`public-files.json` is the explicit allowlist for the Pages site. It contains aggregate reports, protocols, source/model hashes, verification summaries, historical process failures, and graphics. The historical supervisor's failure status is retained; the final verification establishes completed prediction coverage. Results and model/data caches are ignored by default. Only the reviewed publication files are committed from `results/`.

Raw dataset utterances, NoReC review texts, model weights, API request/response payloads, per-case prediction files, environment folders, credentials and cache files remain local. Report demos contain assistant-authored diagnostics, not private customer messages. The original report files are preserved; the Pages build labels links to unpublished raw evidence as local-only and adds navigation. Raw benchmark evidence can be reconstructed by following the fixture preparation and benchmark commands, subject to source availability and dataset/model terms.

The public metrics are not a production acceptance test or measured ROI. Accuracy on 18 MASSIVE assistant-service scenarios is a routing proxy, not an estimate for a company's support inbox. The supervised classifier uses labelled training data; zero-shot models do not use task-specific fitting in this benchmark. Similar scores do not establish statistical equivalence. Native FIFO load testing is distinct from vLLM compatibility and simultaneous GPU execution.

Build the public site without running inference:

```powershell
.venv\Scripts\python.exe -m laya_bench.business_summary
.venv\Scripts\python.exe scripts/build_pages.py --output .cache/pages-site
```

The share-card generator verifies complete source reports and records their SHA-256 hashes in the share package. It exports a 1080×1350 PNG, a 2160×2700 master, editable SVG, alt text, and English/Norwegian sharing drafts. `build_pages.py` requires every public source to exist, builds the landing page from the recorded metrics, checks local report links and writes an asset checksum manifest. GitHub Actions repeats this check and deploys the static site on pushes to `main`.

On a clean clone, the committed share card and aggregate results are sufficient to build the site with standard Python. Regenerating results requires datasets and models to be prepared locally. Model and dataset licences apply separately; this repository does not redistribute them.

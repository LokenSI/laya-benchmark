"""Business assessment generated from saved evidence; no model calls."""
import html
from pathlib import Path

from .common import ROOT,read_json,write_json,digest


def pct(v):
    return f"{100*v:.1f}%"


def table(headers,rows):
    return '<div class="table"><table><thead><tr>'+''.join('<th>'+html.escape(str(v))+'</th>' for v in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table></div>'


def build():
    c=read_json(ROOT/"results/claims/summary.json")
    j=read_json(ROOT/"results/jev/summary.json")
    i=read_json(ROOT/"results/industry/summary.json")
    out=ROOT/"results/assessment"
    out.mkdir(parents=True,exist_ok=True)
    claims=[]
    names={"news":"News topics","emotion":"Emotion","spam":"Email spam","phishing":"Phishing","jailbreaking":"Jailbreak detection","toxicity":"Toxicity","intent":"English intent (20 candidates)","scenario":"English scenario (18 labels)"}
    for model in ["laya","laya-multilingual"]:
        for task in ["news.replication","emotion.replication","spam.replication","phishing.replication","jailbreaking","toxicity","intent","scenario"]:
            r=c["results"][f"{model}/{task}"]
            claims.append([model,names[task.split('.')[0]],r["n"],pct(r["published_accuracy"]),pct(r["accuracy"]),f'{r["difference_pp"]:+.2f} pp',r["verdict"]])
    reproduced=sum('not reproduced' not in row[-1] for row in claims)
    jev=[]
    for dataset,label in [("agnews","News (4 labels)"),("emotiondair","Emotion (6 labels)"),("banking77","Banking intent (72 labels)")]:
        rows=[j["results"][f"{m}/{dataset}/default"] for m in ["laya","laya-multilingual"]]
        jev.append([label,100,pct(rows[0]["accuracy"]),pct(rows[1]["accuracy"]),pct(rows[0]["published_jev_accuracy"])])
    industry=[]
    for task in ["documents","routing"]:
        for lang,model in [("en","laya"),("nb","laya-multilingual")]:
            r=i["results"][f"{model}/{task}/{lang}"]
            baseline=i["results"][f"rules/{task}/{lang}"]
            interval='–'.join(pct(v) for v in r["accuracy_wilson95"])
            industry.append([task,"English" if lang=="en" else "Norwegian Bokmål",f'{r["correct"]}/{r["n"]}',pct(r["accuracy"]),interval,pct(baseline["accuracy"]),pct(r["option_order_flip_rate"])])
    extension=[]
    for task in ["news","emotion","spam","phishing"]:
        a=c["results"][f"laya/{task}.extension"]
        b=c["results"][f"laya-multilingual/{task}.extension"]
        extension.append([names[task],1000,pct(a["accuracy"]),pct(b["accuracy"])])
    en=i["results"]["laya/documents/en"]
    nb=i["results"]["laya-multilingual/documents/nb"]
    body=f'''<div class="eyebrow">Independent local evaluation · 26 September 2026</div>
<h1>Where Laya can earn its place</h1><p class="lead">A candidate for reviewed English document filing. Strong public email classification results. Insufficient evidence for autonomous industrial decisions or dependable Norwegian routing.</p>
<div class="stats"><div><b>{reproduced}/16</b><span>published figures reproduced within 2 percentage points</span></div><div><b>25/28</b><span>English engineering document cases classified correctly</span></div><div><b>21/28</b><span>Norwegian engineering document cases classified correctly</span></div></div>
<nav><a href="#claims">Claims audit</a><a href="#jev">Jev comparison</a><a href="#industry">Industry tests</a><a href="#value">Business value</a><a href="#method">Evidence & limits</a></nav>
<h2 id="claims">The headline accuracy figures mostly reproduce</h2><p>The earlier 66.6% English result measured 18-way scenario classification on 2,948 examples. The advertised 78.3% concerns a different intent task with 20 candidates, including the known correct label plus 19 randomly sampled alternatives. Rebuilding that protocol gave 78.0%. The protocols measure different capabilities.</p>
{table(["Checkpoint","Task","N","Published","Our run","Difference","Outcome"],claims)}
<p class="note">Predeclared ±2 percentage-point repeatability tolerance, not a statistical significance test. We reconstructed the publisher's sample order and questions. Our SDK is 0.3.20 on an RTX 5070 Ti; the application reference used SDK 0.2.1 on CPU, and MASSIVE used a T4. Upstream data revisions were not published. Our dataset and model revisions are pinned. The multilingual jailbreak result was 5.0 points higher; the precise cause is unresolved.</p>
<h3>Additional examples beyond the advertised slices</h3>{table(["Task","Additional N per model","Laya","Multilingual"],extension)}
<p>These additional slices exclude the reproduced row IDs. They are from the same datasets, so they are not proof of performance on a new domain. The publisher marks news, spam and phishing as training tasks; that does not establish exact test-example leakage. Training-data overlap could not be independently audited. Emotion and ToxicChat are described by the publisher as held out.</p>
<h3>Which statements hold up?</h3>
{table(["Statement","Assessment"],[
('High news and email accuracy','Supported on the tested public benchmark protocols. The extra slices remain strong. Transfer to current industrial mail is unverified.'),
('Mathematically calibrated probabilities','Not a guarantee on new data. English document ECE was '+f'{en["calibration"]["ece"]:.3f}'+', Norwegian '+f'{nb["calibration"]["ece"]:.3f}'+'. Proper scoring rules encourage calibration at an optimum; they do not establish it for a deployed model.'),
('Nothing to hallucinate','Typed outputs prevent invented answer labels and free-text fabrication. They still produce confidently wrong decisions. A fixed output type is not a correctness guarantee.'),
('100+ languages / usable languages','Broad language coverage is not business-grade accuracy. Norwegian scenario accuracy was 51.3%; our authored Norwegian request routing was 45.0%. We did not test 100 languages.'),
('Roughly 33 ms inference','Plausible on our GPU: short industry requests took about 20–26 ms at warm batch size one. Hardware, sequence length, batch size and load matter. This does not reproduce the T4 or hosted-service latency conditions.'),
('Free self-hosting','No per-call vendor fee in this offline setup. Compute, integration, monitoring and review still cost money.'),
('76.6% typed decisions','Belongs to the separate fine-tuned checkpoint. It is not a score for either base checkpoint tested here; we did not reproduce that checkpoint.'),
('Superior to Jev in general','Not supported by the matched pilot below. The ranking depends on the task and input schema.'),
('8,192-token input support','Capacity claim not validated end to end here. Our industrial tests use short excerpts; long-document accuracy and PDF extraction remain outside this evaluation.')])}
<p>The publisher's current <a href="https://huggingface.co/convaiinnovations/laya#honest-limits">limitations section</a> already acknowledges overconfidence and weak general decision performance. The issue is interpreting broad introductory claims as guarantees.</p>
<h2 id="jev">Jev: the same 300-example fixture</h2><p>Our reconstruction matches the independent pilot's published SHA-256 hash exactly, covering text, targets, labels, label order and example order. We also matched its question and state format. Both Laya checkpoints ran locally. Jev 1.13.0 figures are the earlier <a href="https://github.com/AbdelStark/jev-benchmarks">independent published run</a>; we did not call the Jev API.</p>
{table(["Task","N","Laya local","Multilingual local","Jev published"],jev)}
<p>All 72 banking labels were truncated at the default option budgets. Expanding the budgets to preserve their full descriptions raised Laya to 16% and multilingual to 14%, still far below the published Jev 87%. This diagnostic uses the same examples and a different budget; it is reported separately. News intervals are wide at N=100: 97% versus 91% does not establish a statistically reliable win without the paired Jev predictions. Local GPU latency and a historical hosted API latency are not a normalized speed contest.</p>
<h2 id="industry">Our industry feasibility tests</h2><p>We authored 100 distinct scenario families, each with an English and Bokmål version. Thirty-two families were reserved for prompt development and 68 for holdout testing. We chose between two prompts using development scores only. Translations stay in the same split. The 28 document examples and 40 request examples per language are balanced by category, not by real-world frequency.</p>
{table(["Workflow","Language","Correct / N","Accuracy","95% Wilson interval","Simple rules accuracy","Option-order flips"],industry)}
<p class="note">Authored examples, not real customer tickets; no independent subject-matter review. The baseline is a fixed keyword matcher that abstains on ties, not an optimized commercial system. Abstentions count as incorrect in its accuracy. This experiment does not prove superiority to a trained classifier. Selecting the use case after these exploratory tests requires a fresh external validation set.</p>
<p>Multilingual document predictions changed on 7/28 Norwegian excerpts when the category order was reversed. Keep the schema fixed and treat the Norwegian workflow as experimental. English document predictions changed on 1/28. None of these results supports unattended filing.</p>
<h3>Failures that explain the limit</h3>
{table(["Document meaning","Correct category","Laya English chose"],[('Request for a supplier quote referencing an attached drawing','Purchasing','Design'),('Work order assigning investigation of a noisy bearing','Maintenance work record','Inspection'),('Repair job referring to the approved design','Maintenance work record','Design')])}
<p>The model sometimes follows a mentioned subject instead of the document's purpose. A reviewer needs the source excerpt alongside the suggestion. High confidence cannot replace that review.</p>
<h2 id="value">A concrete product: document intake assistance</h2>
<p>Put a suggested document category beside an incoming excerpt: design/specification, inspection/test record, purchasing document, or maintenance work record. Copy explicit asset tags with deterministic rules. Let a records coordinator accept or correct the category, then hand the approved metadata to the document system.</p>
<div class="flow"><div>Extract text</div><span>→</span><div>Suggest category</div><span>→</span><div>Reviewer confirms</div><span>→</span><div>Save approved metadata</div></div>
{table(["Sector","Software workflow","Potential value","Current decision"],[
('Oil and gas','Incoming vendor and maintenance records in an engineering document system','Reduce manual metadata entry and sorting; keep asset references searchable','Pilot reviewed English classification; do not infer safety or fitness for service'),
('Manufacturing','Quality and equipment document intake','Pre-sort inspection records, specifications and supplier documents','Pilot reviewed suggestions; validate on real plant records'),
('Engineering','Project document control and technical archives','Help coordinators choose a filing category without a generative answer','Strongest industry candidate in this test'),
('Helpdesk','Initial request ownership and queue routing','Potential reduction in manual triage','Defer autonomous routing: 72.5% English and 45.0% Norwegian on our authored holdout'),
('Procurement inbox','Spam or suspicious-message review queue','Reduce review load for clearly irrelevant incoming mail','Public-data candidate; requires current, domain-specific mail validation and a false-positive study')])}
<p>This administrative placement fits the review-and-approval stages of <a href="https://www.ibm.com/think/topics/work-order-management">industrial work-order management</a>. We measured classification, not equipment outcomes, avoided downtime, or cost savings.</p>
<p><a class="button" href="http://127.0.0.1:8766/">Open the local document intake demo</a></p><p class="fine">The working demo runs both checkpoints locally, checks input length, shows the source and suggestions, and exports a reviewed JSON draft. It has no document-system connector, PDF parser or automatic maintenance action.</p>
<h3>Value depends on review time</h3><p>The calculator is a scenario, not a measured return. It assumes every wrong suggestion is detected and incurs extra correction time. It does not price residual misfiling, retraining or downstream incidents.</p>
<div class="calculator"><label>Documents per month<input id="volume" type="number" min="0" value="10000"></label><label>Manual classification, seconds<input id="manual" type="number" min="0" value="20"></label><label>Review per suggestion, seconds<input id="review" type="number" min="0" value="8"></label><label>Extra correction time, seconds<input id="correction" type="number" min="0" value="30"></label><label>Classification accuracy, %<input id="accuracy" type="number" min="0" max="100" value="89.3"></label><label>Loaded labour cost, NOK/hour<input id="hourly" type="number" min="0" value="600"></label><label>Monthly compute + amortized integration, NOK<input id="cost" type="number" min="0" value="3000"></label><div class="roi"><b id="net"></b><span id="hours"></span><span id="breakeven"></span></div></div>
<p class="fine">Net value = volume × [manual seconds − review seconds − (1 − accuracy) × correction seconds] ÷ 3,600 × hourly cost − monthly fixed cost. Replace all assumptions with observed workflow measurements before making an investment case.</p>
<h3>What would justify deployment?</h3><p>Run a shadow pilot on at least 1,000 new, de-identified real documents, sampled from the actual language and document mix. Have two domain reviewers resolve disputed labels. Split by document family, vendor and time so revisions do not cross splits. Compare the current process, rules, a trained text classifier and Laya. Measure classification accuracy, per-category errors, rejected cases, review seconds, correction seconds and final misfiling rates.</p><p>A proposed go/no-go condition for assistance is at least 20% lower median handling time with no deterioration in final reviewed accuracy. Pre-register the target and sample size for the site's acceptable error rate. Automation would require a separate validation of the accepted subset and its error bounds. The present evidence supports assistance only.</p>
<h2 id="method">Evidence and reproducibility</h2><p>Python 3.11 venv; Laya 0.3.20; Torch 2.8.0 + CUDA 12.8; RTX 5070 Ti; fixed model revisions; offline inference. Both models were tested with their shipped settings. We retained every decision and audited state and option truncation. The public email replication intentionally preserves the publisher's truncation rules. No industry test excerpt was truncated.</p>
<p>Claim replication: 3,000 examples per model across eight tasks, plus 4,000 extension examples per model. Matched Jev pilot: 300 examples per model, plus 100 banking budget diagnostics each. Industry: 64 development and 136 holdout language-specific records; results remain separated by language, and reversed-order diagnostics are not additional independent examples.</p>
<p>All claim comparisons and all industry task/model/language results are retained, including negative findings. Accuracy intervals reflect sampling uncertainty under the stated sample assumptions, not uncertainty about synthetic-to-real transfer. No public test set can establish absence of training overlap.</p>
<p>Detailed machine-readable results: <code>results/claims/summary.json</code>, <code>results/jev/summary.json</code>, <code>results/industry/summary.json</code>. Sources and hashes are stored with the results. Original English/Norwegian benchmark: <code>results/full/report.html</code>. The LinkedIn figures are generated from these same JSON files.</p>
<p>Sources: <a href="https://huggingface.co/convaiinnovations/laya">Laya model card</a>; <a href="https://huggingface.co/convaiinnovations/laya-multilingual">multilingual model card</a>; <a href="https://github.com/NandhaKishorM/laya/blob/{c['manifest']['publisher_commit']}/BENCHMARKS.md">pinned publisher benchmark report</a>; <a href="https://github.com/AbdelStark/jev-benchmarks/blob/{j['provenance']['reference_repo_commit']}/results/reports/btzsc-pilot-v1.json">pinned independent Jev results</a>.</p>
<footer>Local measurement, independent interpretation. No vendor sponsorship claimed. Dataset text stays in the local cache; shared figures contain aggregate results only.</footer>'''
    css='''*{box-sizing:border-box}html{scroll-behavior:smooth}body{font:16px/1.65 "Segoe UI",Arial,sans-serif;background:#f4f7f8;color:#152c38;margin:0}main{max-width:1200px;margin:0 auto;background:white;padding:55px 55px 25px}.eyebrow{font-size:12px;text-transform:uppercase;letter-spacing:1.8px;color:#617681}h1{font-size:46px;letter-spacing:-1.5px;line-height:1.12;max-width:900px}h2{font-size:29px;margin-top:58px;border-top:1px solid #dce5e9;padding-top:28px;line-height:1.2}h3{font-size:21px;margin-top:30px}.lead{font-size:22px;max-width:960px;color:#49616e}.stats{display:grid;grid-template-columns:repeat(3,1fr);gap:28px;margin:40px 0}.stats b{display:block;font-size:45px;color:#175b76}.stats span{display:block;color:#5f747e;font-size:14px}nav{display:flex;gap:22px;flex-wrap:wrap;border-block:1px solid #dce5e9;padding:15px 0}a{color:#175b76;text-underline-offset:3px}.table{overflow:auto;margin:20px 0}table{border-collapse:collapse;width:100%;font-size:14px}th{text-align:left;color:#536c79;font-weight:600;background:#f3f6f7}th,td{padding:12px 14px;border-bottom:1px solid #dee6e9;vertical-align:top}td:first-child{font-weight:600}.note{background:#f3f6f7;border-left:3px solid #819ba8;padding:18px;font-size:14px}.fine,footer{color:#617681;font-size:13px}.flow{display:flex;align-items:center;gap:14px;margin:30px 0}.flow div{padding:17px;background:#eaf1f4;flex:1;text-align:center;font-size:14px}.button{display:inline-block;padding:11px 20px;background:#175b76;color:white;text-decoration:none;border-radius:4px}.calculator{display:grid;grid-template-columns:repeat(4,1fr);gap:18px;padding:25px;background:#f3f6f7}.calculator label{font-size:13px;display:flex;flex-direction:column;justify-content:space-between;gap:6px}input{width:100%;padding:10px;font:inherit;border:1px solid #bdcdd5;background:white}.roi b{display:block;font-size:24px}.roi span{display:block;font-size:12px;color:#526d79}code{font-size:13px}footer{margin-top:45px;border-top:1px solid #dce5e9;padding-top:20px}@media(max-width:800px){main{padding:28px 20px}h1{font-size:35px}.stats{gap:14px}.stats b{font-size:32px}.calculator{grid-template-columns:1fr 1fr}.flow{flex-wrap:wrap}}'''
    js='''const ids=['volume','manual','review','correction','accuracy','hourly','cost'];function calc(){const v=Object.fromEntries(ids.map(id=>[id,Math.max(0,Number(document.getElementById(id).value)||0)]));v.accuracy=Math.min(100,v.accuracy)/100;const seconds=v.manual-v.review-(1-v.accuracy)*v.correction;const hours=v.volume*seconds/3600;const net=hours*v.hourly-v.cost;document.getElementById('net').textContent=new Intl.NumberFormat('en-GB',{maximumFractionDigits:0}).format(net)+' NOK / month';document.getElementById('hours').textContent=hours.toFixed(1)+' net staff hours saved';const contribution=seconds*v.hourly/3600;document.getElementById('breakeven').textContent=contribution>0?'Break-even: '+Math.ceil(v.cost/contribution).toLocaleString()+' documents/month':'No break-even at these assumptions';}ids.forEach(id=>document.getElementById(id).addEventListener('input',calc));calc();'''
    (out/"report.html").write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Laya: business assessment and claim audit</title><style>'+css+'</style><main>'+body+'</main><script>'+js+'</script></html>',encoding="utf-8")
    def mdtable(headers,rows):
        return '| '+' | '.join(headers)+' |\n| '+' | '.join('---' for _ in headers)+' |\n'+''.join('| '+' | '.join(str(v) for v in row)+' |\n' for row in rows)
    md=f'''# Laya: evidence and business decision

Measured 26 September 2026. Recommend a reviewed English engineering-document classification pilot. Defer autonomous routing and operational decisions; Norwegian remains experimental.

## Claim audit

{reproduced}/16 figures reproduced within a predeclared 2 percentage-point tolerance. This checks practical repeatability, not statistical equivalence. SDK and hardware differ; original dataset revisions were not pinned.

{mdtable(['Model','Task','N','Published','Local','Difference','Outcome'],claims)}

Strong news/email tasks were listed by the publisher as training tasks. Test-example leakage was not established or independently ruled out. New-domain transfer is unverified. Raw calibration and fixed answer labels do not guarantee correctness. Self-hosting avoids API fees, not infrastructure and operating costs. The 76.6% typed-decisions claim belongs to a separate fine-tuned model.

## Same-fixture Jev comparison

{mdtable(['Task','N','Laya','Multilingual','Jev published'],jev)}

Exact independent manifest SHA-256 matched: `{j['provenance']['manifest_sha256']}`. Jev 1.13.0 was NOT called here. Its numbers are the independent author's historical run. No paired significance claim or normalized latency comparison. Increasing banking option budgets to remove truncation produced 16% Laya / 14% multilingual, versus 2% each at defaults.

## Authored industry feasibility tests

{mdtable(['Workflow','Language','Correct/N','Accuracy','95% interval','Rules','Order flips'],industry)}

100 distinct bilingual scenario families; 32 development and 68 test families. Two prompt variants chosen on development only. Synthetic cases lack SME review and real-world frequency weighting. Simple rules are an illustrative comparator, not a tuned incumbent. The strongest use case was selected after exploratory testing and needs fresh external validation.

## Product and conditional value

Document intake desk: suggest design, inspection, purchasing or maintenance categories, copy explicit asset tags, require reviewer confirmation, export approved metadata. Oil and gas: vendor and asset records. Manufacturing: quality and equipment records. Engineering: project document control. The working local demo exports a reviewed JSON draft; it has no PDF parser or business-system connector.

Illustrative economics: 10,000 documents/month, 20 seconds manual classification, 8 seconds review, 30 extra seconds to correct a wrong suggestion, 600 NOK/hour, 3,000 NOK/month fixed cost. At 25/28 accuracy this suggests about 24.4 hours and 11,643 NOK/month net capacity value. These time and cost assumptions are not measurements; final errors and downstream losses are not priced. At the English 95% interval endpoints, estimated net capacity value ranges from about 3,402 to 15,144 NOK/month under the same assumptions. Capacity value is not necessarily cash saved.

Validate on at least 1,000 new real documents with two domain reviewers, split by family/vendor/time. Compare existing process, rules, trained classifier and Laya. Measure final reviewed errors and handling time. Proposed assisted-workflow acceptance: at least 20% lower median handling time with no loss of final accuracy. No current evidence supports unattended filing, safety decisions, or fault diagnosis.

## Sources

- [Laya model card](https://huggingface.co/convaiinnovations/laya)
- [Publisher benchmark report, pinned](https://github.com/NandhaKishorM/laya/blob/{c['manifest']['publisher_commit']}/BENCHMARKS.md)
- [Independent Jev pilot, pinned](https://github.com/AbdelStark/jev-benchmarks/blob/{j['provenance']['reference_repo_commit']}/results/reports/btzsc-pilot-v1.json)
- [IBM work-order workflow](https://www.ibm.com/think/topics/work-order-management)

Full tables, counterexamples, assumptions and interactive calculator: `report.html`. Machine-readable provenance and predictions: `results/claims`, `results/jev`, `results/industry`. No benchmark text is included in the LinkedIn graphics.
'''
    (out/"business_assessment.md").write_text(md,encoding="utf-8")
    write_json(out/"artifact_sources.json",{name:digest(ROOT/f"results/{name}/summary.json") for name in ['claims','jev','industry','full']})
    print(out/"report.html")

if __name__=="__main__":
    build()

"""LinkedIn carousel graphics built from saved evidence; no model calls, no benchmark text."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from .common import ROOT, read_json

OUT = ROOT / "results/linkedin"
W, H, DPI = 10.8, 13.5, 200          # 1080x1350 at 2x -> 2160x2700 px
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
LAYA, REF, NEUTRAL = "#2a78d6", "#eb6834", "#b9b7ae"
SOURCE = "Independent local replication. Laya SDK 0.3.20, RTX 5070 Ti, 26-27 Sep 2026."
plt.rcParams.update({"font.family": "Segoe UI", "text.color": INK, "axes.edgecolor": AXIS,
                     "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": INK2})


def page(n, total, title, subtitle):
    fig = plt.figure(figsize=(W, H), dpi=DPI, facecolor=SURFACE)
    fig.text(.08, .935, title, fontsize=30, weight="semibold", va="top", wrap=True)
    fig.text(.08, .855, subtitle, fontsize=15, color=INK2, va="top", linespacing=1.45)
    fig.text(.08, .04, SOURCE, fontsize=10.5, color=MUTED)
    fig.text(.92, .04, f"{n} / {total}", fontsize=10.5, color=MUTED, ha="right")
    fig.add_artist(plt.Line2D([.08, .92], [.065, .065], color=GRID, lw=1))
    return fig


def hbars(ax, labels, values, colors, notes=None, xmax=100):
    y = list(range(len(labels)))[::-1]
    ax.barh(y, values, height=.62, color=colors, edgecolor=SURFACE, linewidth=2)
    for yi, v, note in zip(y, values, notes or [None] * len(values)):
        ax.text(v + 1.2, yi, f"{v:.0f}%", va="center", fontsize=15, weight="semibold", color=INK, zorder=6, bbox=dict(boxstyle="square,pad=0.15", fc=SURFACE, ec="none"))
        if note:
            ax.text(v + 9.5, yi, note, va="center", fontsize=11.5, color=MUTED)
    ax.set_yticks(y, labels, fontsize=14)
    style(ax, xmax)


def style(ax, xmax=100):
    ax.set_xlim(0, xmax)
    ax.set_facecolor(SURFACE)
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", labelsize=11)
    ax.xaxis.grid(True, color=GRID, lw=.8)
    ax.set_axisbelow(True)
    ax.set_xticks(range(0, xmax + 1, 25), [f"{v}%" for v in range(0, xmax + 1, 25)])


def panel_title(fig, x, y, text, sub=None):
    fig.text(x, y, text, fontsize=17, weight="semibold")
    if sub:
        fig.text(x, y - .022, sub, fontsize=12, color=MUTED)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=DPI, facecolor=SURFACE)
    plt.close(fig)


def build():
    claims = read_json(ROOT / "results/claims/summary.json")["results"]
    native = read_json(ROOT / "results/jev_native/summary.json")["results"]
    jev = read_json(ROOT / "results/jev/summary.json")["results"]
    pub = read_json(ROOT / "results/jev/provenance.json")["published"]["results"]
    full = read_json(ROOT / "results/full/summary.json")
    total = 6
    a = lambda r: 100 * r["accuracy"]

    # 1. Verdict
    fig = page(1, total, "I tested Laya, the fast\nzero-shot classifier.", "")
    lines = [("The published numbers are real.", "15 of 16 publisher figures reproduced within 2 points on my machine."),
             ("They are also narrow.", "Strongest results are on tasks the publisher lists as training data."),
             ("Setup decides the score.", "Same model, same 100 banking messages: 2% to 77% depending on how\nthe labels are written and how many are shown at once."),
             ("A trained baseline still wins.", "With labelled data, a plain TF-IDF classifier beat it by 23 points\nin English and 38 points in Norwegian."),
             ("Where it fits:", "Fast, local, few-label decisions with a human reviewing the output.")]
    y = .77
    for head, body in lines:
        fig.text(.08, y, head, fontsize=19, weight="semibold", color=LAYA if head == "Where it fits:" else INK)
        fig.text(.08, y - .02, body, fontsize=14, color=INK2, va="top", linespacing=1.4)
        y -= .12
    save(fig, "01_verdict.png")

    # 2. Publisher claims reproduce
    fig = page(2, total, "The publisher's numbers\nreproduce.", "English Laya, publisher's own tasks and prompts, 300-400 examples each.\nGrey ring: published. Blue dot: measured here.")
    ax = fig.add_axes([.34, .13, .56, .66])
    tasks = [("spam.replication", "Email spam"), ("phishing.replication", "Phishing"), ("news.replication", "News topics"),
             ("intent", "Intent, 20 options"), ("jailbreaking", "Jailbreak detection"), ("scenario", "Scenario, 18 options"),
             ("emotion.replication", "Emotion"), ("toxicity", "Toxicity")]
    for i, (key, label) in enumerate(tasks[::-1]):
        r = claims[f"laya/{key}"]
        p, m = 100 * r["published_accuracy"], 100 * r["accuracy"]
        ax.plot([p, m], [i, i], color=GRID, lw=2, zorder=1)
        ax.scatter([p], [i], s=420, facecolor="none", edgecolor=MUTED, linewidth=2, zorder=4)
        ax.scatter([m], [i], s=150, color=LAYA, zorder=3, edgecolor=SURFACE, linewidth=2)
        ax.text(max(p, m) + 2.5, i, f"{m:.1f}%", va="center", fontsize=14, weight="semibold")
    ax.set_yticks(range(len(tasks)), [t[1] for t in tasks[::-1]], fontsize=14)
    style(ax)
    ax.set_xlim(40, 108)
    ax.set_xticks([50, 75, 100], ["50%", "75%", "100%"])
    ax.set_ylim(-.6, len(tasks) - .4)
    fig.text(.08, .095, "News and email tasks are listed by the publisher as training tasks. Toxicity at 53% is close to a coin flip.",
             fontsize=11.5, color=MUTED)
    save(fig, "02_claims_reproduce.png")

    # 3. Why results differ: banking setup
    fig = page(3, total, "Why my first results\nlooked broken.", "Same model, same 100 banking messages, 72 possible intents.\nOnly the way the question is posed changes.")
    ax = fig.add_axes([.42, .40, .48, .38])
    rows = [("Generic labels, all 72 at once", a(jev["laya/banking77/default"])),
            ("Plain label names, all 72 at once", a(native["laya/banking77/native"])),
            ("Plain names, top 20 pre-filtered", a(native["laya/banking77/native_shortlist20"])),
            ("Plain names, top 10 pre-filtered", a(native["laya/banking77/native_shortlist10"]))]
    hbars(ax, [r[0] for r in rows], [r[1] for r in rows], [NEUTRAL, NEUTRAL, LAYA, LAYA])
    emb = a(native["embedding-only/banking77"])
    ax.axvline(emb, color=INK2, lw=1.5, ls=(0, (4, 3)), zorder=5)
    ax.text(emb, 3.62, f"Embedding model alone {emb:.0f}%", ha="center", fontsize=11.5, color=INK2, zorder=6, bbox=dict(boxstyle="square,pad=0.2", fc=SURFACE, ec="none"))
    ax.set_ylim(-.5, 3.9)
    fig.text(.08, .33, "What is going on", fontsize=17, weight="semibold")
    fig.text(.08, .305,
             "Laya fits every answer option into one fixed 192-token budget. With 72 long labels,\n"
             "each option is cut to a few tokens and the model cannot tell them apart. The fix\n"
             "the SDK itself recommends is to pre-filter to about 20 options with an embedding\n"
             "model first.\n\n"
             "The catch: that small embedding model alone scores "
             f"{a(native['embedding-only/banking77']):.0f}% on the same messages.\n"
             "Laya adds 6 points with a top-10 shortlist and loses 4 with top-20.",
             fontsize=14, color=INK2, va="top", linespacing=1.5)
    save(fig, "03_setup_decides.png")

    # 4. Head to head on identical Jev examples
    fig = page(4, total, "Head to head on\nidentical examples.",
               "The 300 examples from an independent Jev vs GLiNER benchmark,\nrebuilt byte for byte (manifest hash verified). 100 per task.")
    comps = [("agnews", "News topics, 4 labels", a(native["laya/agnews/native"]), "in Laya's training data"),
             ("emotiondair", "Emotion, 6 labels", a(native["laya/emotiondair/native"]), None),
             ("banking77", "Banking intent, 72 labels", a(native["laya/banking77/native_shortlist20"]), "with top-20 pre-filter")]
    top = .72
    for key, title, laya, note in comps:
        panel_title(fig, .08, top - .005, title, note)
        ax = fig.add_axes([.34, top - .17, .56, .115])
        names = ["Laya", "Jev (hosted)", "GLiNER 2.5", "Embedding only"]
        vals = [laya, 100 * pub["jev"][key]["accuracy"], 100 * pub["gliner"][key]["accuracy"], a(native[f"embedding-only/{key}"])]
        hbars(ax, names, vals, [LAYA, NEUTRAL, NEUTRAL, NEUTRAL])
        if key != "banking77":
            ax.set_xticklabels([])
        top -= .205
    fig.text(.08, .078, "Jev and GLiNER figures are the independent author's published run; not re-run here. Embedding only:\n"
             "bge-small-en-v1.5, nearest label name. n=100 per task: differences under ~10 points are not reliable.",
             fontsize=11.5, color=MUTED, linespacing=1.4)
    save(fig, "04_head_to_head.png")

    # 5. Versus a trained baseline
    fig = page(5, total, "If you have labelled data,\ntrain something simple.",
               "Routing requests to 18 service categories. Full MASSIVE test set,\n2,948 examples per language.")
    for top, lang, key in [(.72, "English", "massive_en"), (.43, "Norwegian Bokmål", "massive_nb")]:
        panel_title(fig, .08, top + .035, lang)
        ax = fig.add_axes([.40, top - .19, .50, .18])
        svm = full["baselines"][key]["tfidf_linear_svm"]
        vals = [100 * full["models"][m]["suites"][key]["accuracy"] for m in ["laya", "laya-multilingual"]] + [100 * svm["accuracy"]]
        hbars(ax, ["Laya", "Laya Multilingual", "TF-IDF + linear SVM"], vals, [LAYA, LAYA, NEUTRAL])
    fig.text(.08, .15, "The SVM is trained on the official MASSIVE training split (11,436 examples).\n"
             "Laya receives no task training. That is the point of zero-shot, and also the limit:\n"
             "once you have labelled examples, it is no longer the best option.",
             fontsize=14, color=INK2, va="top", linespacing=1.5)
    save(fig, "05_trained_baseline.png")

    # 6. Where it fits
    fig = page(6, total, "Useful, with limits.", "")
    cols = [("Worth a pilot", LAYA, ["Few, clearly named options (2-10)", "Spam, phishing, topic-style decisions",
                                     "Fast local inference: ~30 ms per request", "Suggestions a person confirms",
                                     "English document filing: 25 of 28\ncorrect on my synthetic test"]),
            ("Not supported by evidence", INK, ["Many labels without a pre-filter", "Norwegian or other non-English text",
                                                 "Emotion, toxicity, nuance", "Unattended actions or safety decisions",
                                                 "Replacing a trained classifier\nwhen labels exist"])]
    for x, (head, color, items) in zip([.08, .52], cols):
        fig.text(x, .79, head, fontsize=20, weight="semibold", color=color)
        fig.add_artist(plt.Line2D([x, x + .38], [.775, .775], color=color, lw=2))
        y = .74
        for item in items:
            fig.text(x, y, item, fontsize=14.5, color=INK2, va="top", linespacing=1.4)
            y -= .058 if "\n" not in item else .08
    fig.text(.08, .40, "Bottom line", fontsize=17, weight="semibold")
    fig.text(.08, .375, "Laya is not hype in the sense of fake numbers. It is a small, fast, genuinely\n"
             "usable zero-shot classifier. The headline results come from favourable conditions:\n"
             "familiar tasks, few labels, the publisher's own prompts. Test it on your own data,\n"
             "with your own labels, before trusting any benchmark, including this one.",
             fontsize=14, color=INK2, va="top", linespacing=1.5)
    save(fig, "06_where_it_fits.png")


if __name__ == "__main__":
    build()

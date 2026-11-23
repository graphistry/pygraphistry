"""Helpers for the openFDA device-recall demo (gfql_semantic_procurement.ipynb).

The notebook shows the GFQL and semantic-operator calls. This file holds the plumbing: the openFDA download and parse, the six
root-cause families, the hub-design computation, the simulated model (a calibrated mock HTTP session), the stage runner and the chart.
Call `init()` once, then `from procurement_helpers import *`.
"""


import hashlib
import json
import math
import os
import re
import threading
import time
import urllib.request
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl

import graphistry
from graphistry.compute.semantic import (OpenAIChatProvider, OpenAIDecisionsProvider, TokenPrices, explain_sem_decide, sem_decide, semantic_trace,
                                         semantic_verdicts)
from IPython.display import Markdown, display



pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", 70)


DEMO_LIVE = os.environ.get("DEMO_LIVE") == "1"
CACHE_DIR = Path("/tmp/openfda-cache")
FILES = {                                              # name -> (URL, SHA-256 of the file that the committed outputs come from)
    "recall": ("https://download.open.fda.gov/device/recall/device-recall-0001-of-0001.json.zip", "722f262bb47ea506c15da1156a1d49e2a759632f5dcdc0be56b69c7b978a8d18"),
    "enforcement": ("https://download.open.fda.gov/device/enforcement/device-enforcement-0001-of-0001.json.zip", "7aef3d29c13214a0df889108282e3e08034bb9ff74270f7648dc9032ea0dfd2f"),
    "classification": ("https://download.open.fda.gov/device/classification/device-classification-0001-of-0001.json.zip", "c2aace26d6f4a37555eaf685557c3cdfe178527be84bc1e69a5d0228bd050325"),
}
HUB_MIN = 3                                            # a design that this many different firms recalled is a hub


def cached_file(name):
    """Path of a downloaded file. A download goes to a temporary name first."""
    url = FILES[name][0]
    path = CACHE_DIR / url.rsplit("/", 1)[1]
    if not path.exists():
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        part = path.with_suffix(".part")
        with urllib.request.urlopen(url, timeout=300) as reply, open(part, "wb") as handle:
            while chunk := reply.read(1 << 20):
                handle.write(chunk)
        part.replace(path)
    return path


def records(path, keep):
    """The records of an openFDA bulk file, with only the fields in `keep`. The file lists one record per block of lines that
    starts with four spaces and an opening brace, so the notebook never holds the whole file in memory."""
    with zipfile.ZipFile(path) as zf, zf.open(zf.infolist()[0]) as raw:
        block = None
        for line in raw:
            if block is None:
                if line == b"    {\n":
                    block = [line]
                continue
            block.append(line)
            if line in (b"    },\n", b"    }\n"):
                record = json.loads(b"".join(block).rstrip().rstrip(b","))
                yield {field: record.get(field) for field in keep}
                block = None


def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(1 << 22):
            digest.update(chunk)
    return digest.hexdigest()


def load_fda():
    """Return (recalls, FDA classes, device names, source label). Downloads the files into CACHE_DIR when they are not cached."""
    try:
        paths = {name: cached_file(name) for name in FILES}
        for name, path in paths.items():
            digest = sha256_of(path)
            if digest != FILES[name][1]:
                print(f"NOTE: {path.name} is not the file that the committed outputs come from (sha256 {digest[:12]}..., expected {FILES[name][1][:12]}...). The numbers will differ.")
        recall = pd.DataFrame(records(paths["recall"], ["product_res_number", "res_event_number", "product_code", "k_numbers", "firm_fei_number",
                                                        "recalling_firm", "reason_for_recall", "root_cause_description", "recall_status"]))
        enforcement = pd.DataFrame(records(paths["enforcement"], ["recall_number", "classification"]))
        classification = pd.DataFrame(records(paths["classification"], ["product_code", "device_name"]))
        for name, frame in (("recall", recall), ("enforcement", enforcement), ("classification", classification)):
            if frame.empty:
                raise ValueError(f"no record was read from the {name} file")
        return recall, enforcement, classification, "openFDA device recalls (full download)"
    except zipfile.BadZipFile as error:
        raise RuntimeError(f"A file in {CACHE_DIR} is not a valid zip file ({error}). Delete it and run the notebook again.") from error
    except OSError as error:
        raise RuntimeError(f"Cannot load the openFDA files. They are not all cached in {CACHE_DIR} and the download failed "
                           f"({type(error).__name__}: {error}). Connect to the network, or copy the files from {', '.join(u for u, _ in FILES.values())} to that directory.") from error
    except (ValueError, KeyError) as error:
        raise RuntimeError(f"The openFDA files in {CACHE_DIR} do not have the expected format ({type(error).__name__}: {error}). "
                           "openFDA replaces its files every week, so the format can change.") from error


FAMILIES = ["component_or_supplier", "design", "manufacturing_process", "software", "packaging_labeling", "other_or_unknown"]
FAMILY_OF = {}
for family, causes in {
    "component_or_supplier": ["Nonconforming Material/Component", "Component design/selection", "Component change control", "Vendor change control",
                              "Material/Component Contamination", "Mixed-up of materials/components", "Release of Material/Component prior to receiving test results", "Counterfeit"],
    "design": ["Device Design", "Process design"],
    "manufacturing_process": ["Process control", "Process change control", "Employee error", "Equipment maintenance", "Environmental control", "Storage",
                              "Reprocessing Controls", "Manufacturing material removal", "Finished device change control"],
    "software": ["Software design", "Software Design Change", "Software design (manufacturing process)", "Software change control",
                 "Software Manufacturing/Software Deployment", "Software in the Use Environment"],
    "packaging_labeling": ["Packaging", "Packaging process control", "Packaging change control", "Package design/selection", "Error in labeling", "Labeling design",
                           "Labeling Change Control", "Labeling mix-ups", "Labeling False and Misleading", "Incorrect or no expiration date"],
}.items():
    FAMILY_OF.update({cause: family for cause in causes})
CLASS_LEVELS = ["class_III", "class_II", "class_I"]
CLASS_OF = {"Class III": "class_III", "Class II": "class_II", "Class I": "class_I"}


FIRM_SKIP = {"inc", "llc", "ltd", "limited", "corp", "corporation", "co", "company", "lp", "llp", "gmbh", "ag", "sa", "plc", "usa", "us", "na", "america", "americas",
             "north", "sales", "service", "services", "international", "intl", "the", "of", "and", "medical", "surgical", "systems", "healthcare", "health",
             "technologies", "technology", "biomedical", "devices", "device", "holdings", "group"}


def firm_key(name):
    """A crude company key: the first two significant words of the firm name. A missing name gives an empty key."""
    if not isinstance(name, str) or not name.strip():
        return ""
    words = [w for w in re.findall(r"[a-z0-9]+", str(name).lower()) if w not in FIRM_SKIP]
    return " ".join(words[:2]) or str(name).lower()


QUESTIONS = [
    {"id": "supplier_related", "kind": "predicate", "text": "Does the reason for this device recall point to a component, a material or a supplier?"},
    {"id": "root_cause_family", "kind": "choice", "text": "Which kind of root cause does this device recall have?", "options": FAMILIES},
    {"id": "fda_class", "kind": "score", "text": "How serious does the FDA rank this device recall?", "levels": CLASS_LEVELS},
]
EVIDENCE = ["reason_for_recall", "device_name"]


SIM = {
    "confidence_beta": {"easy": (24, 1.5), "hard": (10, 2)},
    "latency_s": {"chat_request": 0.9, "decisions_base": 1.2, "decisions_per_question": 0.15},
    "answer_tokens": {"chat": 1, "decisions_per_question": 12, "takeaway": 45},
}
PRICES = TokenPrices(input=2.50, cached_input=1.25, output=10.0)                # USD per million tokens; illustrative
PARALLEL = 8                                                                    # parallel requests in stages 4 and 5


json_dumps, json_loads = json.dumps, json.loads   # the post() argument below is also named json
QUESTION_BY_TEXT = {q["text"]: q for q in QUESTIONS}
CHOICES = {q["id"]: q.get("options") or q.get("levels") or [True, False] for q in QUESTIONS}
FIRST = QUESTIONS[0]["id"]
KEY = EVIDENCE
LABELS = ["supplier_related", "root_cause_family", "fda_class"]


class MockResponse:
    def __init__(self, body):
        self.status_code, self._body, self.headers = 200, body, {}

    def json(self):
        return self._body


def most_common_labels(frame):
    """For each distinct prompt, the most common FDA label of each question among the recalls that have that prompt."""
    modes = {}
    for label in LABELS:
        modes[label] = frame.groupby(KEY)[label].agg(lambda values: values.value_counts().idxmax())
    return pd.DataFrame(modes)


class SimulatedSession:
    """A `requests`-style session. It answers OpenAI-format requests from a seeded, calibrated model that corrupts the most common FDA
    label of each prompt. It sends nothing over the network. It records request counts, tokens and modeled latency."""

    def __init__(self, labels, sim=SIM, takeaway=None):
        self.sim, self.takeaway, self.lock = sim, takeaway, threading.Lock()
        self.reset()
        self.labels = {key: tuple(row) for key, row in zip(labels.index, labels[LABELS].itertuples(index=False))}

    def reset(self):
        self.requests = self.input_tokens = self.output_tokens = 0
        self.latency_s = 0.0

    def draw(self, record, qid):
        """(answer index, probability of that answer). Fixed for each (prompt, question)."""
        key = tuple(record[c] for c in KEY)
        supplier, family, level = self.labels[key]
        options = CHOICES[qid]
        true_value = {"supplier_related": supplier, "root_cause_family": family, "fda_class": level}[qid]
        truth_index = (0 if true_value else 1) if qid == FIRST else options.index(true_value)
        rng = np.random.default_rng(int.from_bytes(hashlib.sha256(f"{key}|{qid}".encode()).digest()[:8], "big"))
        a, b = self.sim["confidence_beta"]["easy" if level == "class_II" else "hard"]
        p = float(np.clip(rng.beta(a, b), 1 / len(options) + 0.01, 0.995))
        wrong = [i for i in range(len(options)) if i != truth_index]
        return (truth_index if rng.random() < p else int(rng.choice(wrong))), p

    def distribution(self, qid, index, p):
        k = len(CHOICES[qid])
        return [p if i == index else (1 - p) / (k - 1) for i in range(k)]

    def decisions_answer(self, name, qid, record):
        index, p = self.draw(record, qid)
        dist, options = self.distribution(qid, index, p), CHOICES[qid]
        if qid == FIRST:
            return {"type": "predicate", "name": name, "probability": dist[0]}
        if qid == "root_cause_family":
            return {"type": "choice", "name": name, "choice": options[index],
                    "probabilities": [{"value": o, "probability": pr} for o, pr in zip(options, dist)]}
        return {"type": "score", "name": name,
                "probabilities": [{"value": i, "label": o, "probability": pr} for i, (o, pr) in enumerate(zip(options, dist))]}

    def post(self, url, *, json, headers, timeout):
        body, sim = json, self.sim
        tokens = math.ceil(len(json_dumps(body)) / 4)
        if "questions" in body:                                           # OpenAI decisions format
            shared = None if body["input"].startswith("Each question below") else json_loads(body["input"])
            answers = []
            for question in body["questions"]:
                if shared is None:                                        # a packed request: each question holds its record
                    record = json_loads(question["instructions"].split("\n\nRecord:\n", 1)[1])
                else:
                    record = shared
                answers.append(self.decisions_answer(question["name"], question["name"].split(".")[-1], record))
            out = sim["answer_tokens"]["decisions_per_question"] * len(answers)
            wait = sim["latency_s"]["decisions_base"] + sim["latency_s"]["decisions_per_question"] * len(answers)
            reply = {"model": "simulated-decisions", "answers": answers,
                     "usage": {"input_tokens": tokens, "input_tokens_details": {"cached_tokens": 0}, "output_tokens": out}}
        else:                                                             # OpenAI chat format
            text = body["messages"][-1]["content"]
            if "<evidence-" in text:                                      # one question about one recall
                record = json_loads(re.search(r"<evidence-[0-9a-f]+>\n(.*?)\n</evidence-", text, re.S).group(1))
                question = QUESTION_BY_TEXT[re.search(r"Question: (.*)\n", text).group(1)]
                index, _ = self.draw(record, question["id"])
                content, out, wait = chr(65 + index), sim["answer_tokens"]["chat"], sim["latency_s"]["chat_request"]
            else:                                                         # a finding for one recall
                content, out, wait = self.takeaway(text), sim["answer_tokens"]["takeaway"], sim["latency_s"]["chat_request"]
            reply = {"model": "simulated-chat", "choices": [{"message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
                     "usage": {"prompt_tokens": tokens, "completion_tokens": out, "prompt_tokens_details": {"cached_tokens": 0}}}
        with self.lock:
            self.requests += 1
            self.input_tokens += tokens
            self.output_tokens += out
            self.latency_s += wait
        return MockResponse(reply)


API_KEY, CHAT_MODEL, DECISIONS_MODEL = "not-a-real-key", "simulated-chat", "simulated-decisions"
BUDGET = {"max_cost_usd": 0.50}                    # cap on the estimated input cost of the live decision stages; the default run sends nothing
CHAT_URL = "https://api.openai.com/v1"
LABEL = "SIMULATED model on real openFDA device recalls" if not DEMO_LIVE else "Measured with a live model (every stage asks about a small sample and scales up)"
PLANS = []                                         # (stage, plan, stats): the last table of the notebook compares them
results = {}                                       # one row of numbers for each stage
COLS = ["rows", "asked", "requests", "cost_usd", "seconds", "flagged", "true_hits", "in_queue", "precision", "recall", "errors"]


class CountingSession:
    """Wraps a live `requests` session and counts the requests that the notebook sends."""

    def __init__(self, inner):
        self.inner, self.requests, self.lock = inner, 0, threading.Lock()

    def post(self, *args, **kwargs):
        with self.lock:
            self.requests += 1
        return self.inner.post(*args, **kwargs)


if DEMO_LIVE:
    CHAT_URL = os.environ.get("OPENAI_BASE_URL", CHAT_URL)
    API_KEY, CHAT_MODEL, DECISIONS_MODEL = os.environ["OPENAI_API_KEY"], os.environ["OPENAI_CHAT_MODEL"], os.environ["OPENAI_DECISIONS_MODEL"]
    PRICES = TokenPrices(input=float(os.environ["PRICE_IN"]), cached_input=float(os.environ["PRICE_CACHED"]), output=float(os.environ["PRICE_OUT"]))
OPTIONS = {"budget": BUDGET} if DEMO_LIVE else {}


def show(frame, digits=3):
    """Show a result table under a caption that says where its numbers come from."""
    display(Markdown(f"*{LABEL}*"))
    display(frame.round(digits))


def takeaway_text(prompt):
    """The simulated model writes a fixed-form finding from the evidence. A live model writes its own."""
    record = json_loads(re.search(r"\{.*\}", prompt, re.S).group(0))
    action = "review the supply chain of this design" if record["verdict"] == "yes" else "an analyst should check this recall"
    return (f"{record['device_name'][:40]}: {record['reason_for_recall'][:70].rstrip()}; design {record['hub_design']} was recalled by "
            f"{record['hub_firms']} firms, so {action}.")


def chat_provider(session):
    return OpenAIChatProvider(API_KEY, CHAT_MODEL, base_url=CHAT_URL, prices=PRICES, session=session, strategy="text", concurrency_limit=PARALLEL)


def decisions_provider(session, rows_per_request=1):
    return OpenAIDecisionsProvider(API_KEY, DECISIONS_MODEL, prices=PRICES, session=session, rows_per_request=rows_per_request,
                                   concurrency_limit=PARALLEL, base_url=CHAT_URL)


def decide(frame, provider, parallel, stage, dedupe=False, **options):
    """Plan, run under a trace, and read the verdicts. Returns the result table, the run statistics and the plan."""
    args = dict(questions=QUESTIONS, provider=provider, task_id="recall-triage", evidence_cols=EVIDENCE, id_col="recall_id",
                max_concurrency=parallel, dedupe=dedupe, **options)
    started = time.perf_counter()
    plan = explain_sem_decide(frame, **args)                   # no request is sent
    with semantic_trace() as runs:
        out = sem_decide(frame, **args)
    stats = runs[0]
    out["verdict"] = semantic_verdicts(out, "fda_class", positive="class_I")
    errors = sum(counts.get("error", 0) for counts in stats["status_counts"].values())
    if errors:
        print(f"WARNING: {errors} answers are errors ({stats['error_codes']}). Check the key, the model names and the rate limits. An error row is never counted as no.")
        assert DEMO_LIVE, "the simulated model must not return errors"
        if errors == len(out) * len(QUESTIONS):
            raise RuntimeError("every answer is an error, so the run stops here. Fix the key, the model names or the host, then run again.")
    skipped = int((out.verdict == "skipped").sum())
    if skipped:
        print(f"WARNING: a limit skipped {skipped} rows; the numbers cover the rows that ran")
    if stage is not None:
        PLANS.append((stage, plan, stats))
    return out, stats, plan, time.perf_counter() - started


def asked(out):
    """The rows that the model answered. A row that a limit stopped is `skipped`."""
    return out[out.verdict != "skipped"]


def quality(out):
    """Precision and recall of the escalate list, and the share of rows in the review queue. A stage that asks about
    a sample of the rows is scored on that sample."""
    out = asked(out)
    flagged = out.loc[out.verdict == "yes", "recall_id"]
    hits = int(truth.positive.reindex(flagged).sum())
    positives = max(int(truth.positive.reindex(out.recall_id).sum()), 1)       # positives among the rows this stage asked about
    precision, recall_ = hits / max(len(flagged), 1), hits / positives
    queued = int(truth.positive.reindex(out.recall_id[out.verdict == "uncertain"]).sum())          # positives that wait in the review queue
    return {"asked": len(out), "flagged": len(flagged), "true_hits": hits, "in_queue": queued, "precision": precision, "recall": recall_,
            "f1": 2 * precision * recall_ / max(precision + recall_, 1e-9), "uncertain_pct": 100 * (out.verdict == "uncertain").mean(),
            "errors": int((out.fda_class__status == "error").sum())}


def stage_row(session, stats, local_s, parallel, scale=1.0, gfql_ms=0.0, extra_requests=0):
    """Requests, billed cost and time of one stage, from the trace. The default run models the time of the provider calls."""
    requests_sent = stats["provider_calls"] + stats["probe_calls"] + extra_requests
    if isinstance(session, SimulatedSession):
        assert session.requests == requests_sent, "the trace must count every request that the simulated session saw"
    cost = stats["billed_cost_usd"] * scale
    seconds = (session.latency_s / parallel if isinstance(session, SimulatedSession) else local_s) * scale + gfql_ms / 1000
    return {"requests": round(requests_sent * scale), "cost_usd": cost, "seconds": seconds}


def sample_of(frame, live_rows):
    """The rows that a stage asks about, and the factor that scales the stage up to all rows. The default run asks about all rows. Live mode asks about a small sample."""
    cap = live_rows if DEMO_LIVE else None
    if cap is None or len(frame) <= cap:
        return frame.reset_index(drop=True), 1.0
    return frame.sample(n=cap, random_state=7).reset_index(drop=True), len(frame) / cap


def run_unbatched(frame, stage):
    """Three chat requests for each row, one at a time, with no `dedupe`."""
    rows, scale = sample_of(frame, 50)
    session = session_for()
    out, stats, plan, local_s = decide(rows, chat_provider(session), 1, stage, **OPTIONS)
    if asked(out).empty:
        raise RuntimeError("a limit skipped every row of this stage, so there is nothing to scale from. Raise the budget or the limits.")
    return session, out, stats, plan, local_s, scale * len(rows) / len(asked(out))




def accuracy(out, qid):
    ok = out[out[f"{qid}__status"] == "ok"]
    return float((ok[qid].astype(object) == truth[qid].reindex(ok.recall_id).to_numpy()).mean()), len(ok)


def replay(out, min_p):
    p = out.fda_class__p
    uncertain = p < min_p
    flagged = out[~uncertain & (out.fda_class == "class_I")].recall_id
    hits = int(truth.positive.reindex(flagged).sum())
    return {"min_p": min_p, "yes": len(flagged), "precision": hits / max(len(flagged), 1),
            "recall": hits / max(int(truth.positive.reindex(out.recall_id).sum()), 1), "review queue": f"{100 * uncertain.mean():.0f}%"}


CHART_PATH = Path("gfql_semantic_procurement_chart.png")


def fmt_time(seconds):
    return f"{seconds:.0f} s" if seconds < 90 else f"{seconds / 60:.0f} min" if seconds < 5400 else f"{seconds / 3600:.1f} h"


def draw_chart(results, source, simulated, flat_rows, hub_rows, positives, path=CHART_PATH):
    names = list(results)
    labels = ["1 Naive\nall recalls,\none question\nper call, serial", "2 + flat filter\n(recall table,\nno graph)", "3 + hub filter\n(shared designs;\nGFQL or pandas)",
              "4 + 3 questions\nper call, 8 at\nonce, + dedupe", "5 + min_p\n+ sem_map"]
    colors = ["#8a94a0", "#6f8fb0", "#2f80c0", "#1d9a8a", "#e07a1f"]
    rows = [results[name] for name in names]
    tag = "simulated" if simulated else "live"
    fig, axes = plt.subplots(1, 3, figsize=(20, 6.6))
    x = np.arange(len(rows))
    for ax, key, title, fmt in ((axes[0], "seconds", f"Speed: time to answer ({tag}, log scale)", fmt_time),
                                (axes[1], "cost_usd", f"Cost: billed spend in USD from the trace ({tag}, log scale)", lambda v: f"${v:,.2f}")):
        values = [row[key] for row in rows]
        bars = ax.bar(x, values, color=colors, width=0.62)
        ax.set_yscale("log")
        ax.set_ylim(min(values) / 4, max(values) * 6)
        for bar, value, i in zip(bars, values, x):
            ratio = values[0] / value
            note = fmt(value) + ("" if i == 0 else f"\n{ratio:.1f}x {'faster' if key == 'seconds' else 'cheaper'}" if ratio < 10 else f"\n{ratio:,.0f}x {'faster' if key == 'seconds' else 'cheaper'}")
            ax.text(bar.get_x() + bar.get_width() / 2, value * 1.25, note, ha="center", va="bottom", fontsize=9.5, fontweight="bold")
        ax.set_title(title, fontsize=12, loc="left", fontweight="bold")
        ax.set_yticks([])
        ax.minorticks_off()
    quality_ax = axes[2]
    precision = [0 if np.isnan(row["precision"]) else 100 * row["precision"] for row in rows]
    bars = quality_ax.bar(x, precision, color=colors, width=0.62)
    quality_ax.scatter(x, [np.nan if np.isnan(row["recall"]) else 100 * row["recall"] for row in rows], marker="D", s=70, color="white", edgecolor="#222", zorder=3, label="recall")
    for bar, row in zip(bars, rows):
        text = "not measured" if np.isnan(row["precision"]) else f"{100 * row['precision']:.0f}% precise\n{row['uncertain_pct']:.0f}% to review" + (f"\n+{row['in_queue']:,} positives\nqueued" if row["in_queue"] else "")
        top = 0 if np.isnan(row["precision"]) else 100 * max(row["precision"], row["recall"])     # clear of the bar and the recall marker
        quality_ax.text(bar.get_x() + bar.get_width() / 2, top + 3.5, text, ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    quality_ax.set_ylim(0, 140)
    quality_ax.set_yticks([0, 25, 50, 75, 100])
    quality_ax.set_yticklabels([f"{t}%" for t in (0, 25, 50, 75, 100)])
    quality_ax.set_title(f"Quality: precision of the class_I list ({tag})", fontsize=12, loc="left", fontweight="bold")
    quality_ax.legend(loc="upper left", frameon=False, fontsize=9)
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8.5)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    kind = ("SIMULATED model on real openFDA device recalls. GFQL time, row counts and every plan are measured. Billed cost, model time and answers come from stated assumptions. Not a live benchmark."
            if simulated else "Measured with a live model. Every stage asks about a small sample and scales up.")
    fig.suptitle("GFQL semantic operators on FDA device recalls: scan everything, filter flat, or follow the shared designs", fontsize=15, fontweight="bold", x=0.01, ha="left")
    fig.text(0.01, 0.915, f"{source}  |  {results[names[0]]['rows']:,} recalls with an FDA class, {flat_rows:,} pass the flat filter, {hub_rows:,} sit on a design that {HUB_MIN}+ firms recalled, "
             f"{positives:,} of them are Class I", fontsize=10.5, color="#444")
    fig.text(0.01, 0.01, kind, fontsize=9.5, color="#b03a2e" if simulated else "#1d6b3a", fontweight="bold")
    fig.text(0.99, 0.01, "Each ratio compares a stage with stage 1.", fontsize=9, color="#444", ha="right")
    fig.text(0.01, 0.045, "The positive class (Class I, on a shared design) uses the same hub evidence as the GFQL filter, so GFQL keeps all positives by design, and it is small. "
             "Labels are the FDA's. Class I is about as common on hubs as elsewhere. A pandas hub filter reads the same recalls as GFQL.", fontsize=8.5, color="#444")
    fig.tight_layout(rect=(0, 0.075, 1, 0.9))
    fig.savefig(path, dpi=180)
    plt.show()



truth = LABELS_BY_PROMPT = None                   # set by init(): the FDA labels of each recall, and the most common label of each prompt


def session_for():
    """A fresh session: the simulator by default, a counting wrapper around `requests` in live mode."""
    if DEMO_LIVE:
        import requests
        return CountingSession(requests.Session())
    return SimulatedSession(LABELS_BY_PROMPT, takeaway=takeaway_text)


def build_graph(recalls, designs, hub_designs):
    """Nodes `recall`, `firm` and `design`; edges `recalled` (firm to recall) and `cleared_as` (recall to design). A design has a `hub` flag."""
    firms = sorted(f for f in recalls.firm.unique() if f)
    design_ids = sorted(designs.design.unique())
    nodes = pl.DataFrame({
        "id": list(recalls.recall_id) + [f"firm:{f}" for f in firms] + design_ids,
        "kind": ["recall"] * len(recalls) + ["firm"] * len(firms) + ["design"] * len(design_ids),
        "hub": [False] * (len(recalls) + len(firms)) + [d in hub_designs for d in design_ids]})
    recalled = recalls[recalls.firm != ""].assign(src="firm:" + recalls.firm, edge_type="recalled")[["src", "recall_id", "edge_type"]].rename(columns={"recall_id": "dst"})
    cleared = designs.rename(columns={"recall_id": "src", "design": "dst"}).assign(edge_type="cleared_as")[["src", "dst", "edge_type"]].drop_duplicates()
    return nodes, pl.from_pandas(pd.concat([recalled, cleared], ignore_index=True))


def init():
    """Load the openFDA files, derive the labels, the hubs and the graph, and publish the results as module globals."""
    started = time.perf_counter()
    recall, enforcement, classification, source = load_fda()
    print(source, "|", f"{len(recall):,} recall records | {len(enforcement):,} enforcement records | {len(classification):,} product codes | read in {time.perf_counter() - started:.0f} s")

    recalls = (recall.merge(enforcement[enforcement.classification.isin(CLASS_OF)].drop_duplicates("recall_number"), left_on="product_res_number", right_on="recall_number")
                     .merge(classification.drop_duplicates("product_code"), on="product_code", how="left"))
    recalls["device_name"] = recalls.device_name.fillna("unknown device")
    recalls["reason_for_recall"] = recalls.reason_for_recall.fillna("").str.strip().str.slice(0, 600)       # the model reads the first 600 characters
    recalls["root_cause_family"] = recalls.root_cause_description.map(FAMILY_OF).fillna("other_or_unknown")
    recalls["fda_class"] = recalls.classification.map(CLASS_OF)
    recalls["supplier_related"] = recalls.root_cause_family == "component_or_supplier"
    recalls = recalls.rename(columns={"product_res_number": "recall_id"}).reset_index(drop=True)

    recalls["firm"] = recalls.recalling_firm.map(firm_key)
    designs = recalls[["recall_id", "firm", "k_numbers"]].explode("k_numbers")
    designs = designs[designs.k_numbers.notna() & (designs.k_numbers != "") & (designs.firm != "")].rename(columns={"k_numbers": "design"})
    firms_per_design = designs.groupby("design").firm.nunique()
    raw_hubs = int((recalls[["recall_id", "recalling_firm", "k_numbers"]].explode("k_numbers").query("k_numbers == k_numbers and k_numbers != ''").groupby("k_numbers").recalling_firm.nunique() >= HUB_MIN).sum())
    hub_designs = set(firms_per_design[firms_per_design >= HUB_MIN].index)
    on_hub = set(designs[designs.design.isin(hub_designs)].recall_id)
    recalls["on_hub"] = recalls.recall_id.isin(on_hub)
    recalls["hub_design"] = recalls.recall_id.map(designs[designs.design.isin(hub_designs)].assign(n_firms=lambda f: f.design.map(firms_per_design))
                                                  .sort_values("n_firms", ascending=False).drop_duplicates("recall_id").set_index("recall_id").design).fillna("")
    recalls["hub_firms"] = recalls.hub_design.map(firms_per_design).fillna(0).astype(int)
    has_design = recalls.recall_id.isin(set(designs.recall_id))
    groups = np.where(recalls.on_hub, "on a hub", np.where(has_design, "not on a hub, has a 510(k)", "no 510(k) named"))
    rate = recalls.groupby(groups).fda_class.value_counts(normalize=True).unstack().reindex(columns=CLASS_LEVELS)
    rate["recalls"] = pd.Series(groups).value_counts()

    recalls["positive"] = (recalls.fda_class == "class_I") & recalls.on_hub
    truth = recalls.set_index("recall_id")[["supplier_related", "root_cause_family", "fda_class", "on_hub", "positive"]]
    assert truth.index.is_unique, "a recall id must be unique"
    assert not set(EVIDENCE) & {"recall_id", "recalling_firm", "k_numbers", "hub_design", "root_cause_description", "root_cause_family", "classification", "fda_class", "supplier_related", "on_hub", "positive"}, "a prompt column must not carry a label"
    key_count = recalls.groupby(EVIDENCE).ngroups
    conflict = int(recalls.groupby(EVIDENCE).fda_class.nunique().gt(1).sum())
    conflict_family = int(recalls.groupby(EVIDENCE).root_cause_family.nunique().gt(1).sum())

    nodes_pl, edges_pl = build_graph(recalls, designs, hub_designs)
    globals().update(
        recall=recall, enforcement=enforcement, classification=classification, SOURCE=source, recalls=recalls, designs=designs, firms_per_design=firms_per_design,
        raw_hubs=raw_hubs, hub_designs=hub_designs, on_hub=on_hub, rate=rate, truth=truth, key_count=key_count, conflict=conflict, conflict_family=conflict_family,
        nodes_pl=nodes_pl, edges_pl=edges_pl, g=graphistry.nodes(nodes_pl, "id").edges(edges_pl, "src", "dst"), LABELS_BY_PROMPT=most_common_labels(recalls))
    print(f"{len(recalls):,} recalls with an FDA class | {len(designs.design.unique()):,} designs | {len(hub_designs):,} hubs | {nodes_pl.height:,} nodes | {edges_pl.height:,} edges")

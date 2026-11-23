"""Helpers for the Panama Papers demo (gfql_semantic_fraud_aml.ipynb).

The notebook shows the GFQL and semantic-operator calls. This file holds the plumbing: the ICIJ download and parse, the company profiles,
the rulebook that labels them, the hub-officer computation, the simulated model (a calibrated mock HTTP session), the stage runner and the chart.
Call `init()` once, then `from fraud_aml_helpers import *`.
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
from graphistry.compute.semantic import OpenAIChatProvider, OpenAIDecisionsProvider, TokenPrices, sem_decide
from IPython.display import Markdown, display

pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", 70)


DEMO_LIVE = os.environ.get("DEMO_LIVE") == "1"
ICIJ_URL = "https://offshoreleaks-data.icij.org/offshoreleaks/csv/full-oldb.LATEST.zip"
ICIJ_SHA256 = "9e5faeed998b5715157bbdb2a600681d1b5bda259ba004c998b3b2836d3c49aa"    # the archive that the committed outputs come from (generated 2026-09-09)
CACHE_DIR = Path("/tmp/icij-offshore-cache")
ARCHIVE = CACHE_DIR / "full-oldb.LATEST.zip"
SOURCE_NAME = "Panama Papers"
HUB_MIN = 10                                          # an officer of this many companies or more is a hub


def cached_archive():
    """Path of the downloaded archive. A download goes to a temporary name first."""
    if not ARCHIVE.exists():
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        part = ARCHIVE.with_suffix(".part")
        with urllib.request.urlopen(ICIJ_URL, timeout=300) as reply, open(part, "wb") as handle:
            while chunk := reply.read(1 << 20):
                handle.write(chunk)
        part.replace(ARCHIVE)
    return ARCHIVE


def read_member(archive, name, columns):
    """The Panama Papers rows of one CSV file in the archive."""
    with zipfile.ZipFile(archive) as zf, zf.open(name) as handle:
        frame = pd.read_csv(handle, dtype=str, keep_default_na=False, usecols=columns + ["sourceID"])
    return frame[frame.sourceID == SOURCE_NAME].drop(columns="sourceID").reset_index(drop=True)


def load_icij():
    """Return (companies, officers, agents, edges, source label). Downloads the archive into CACHE_DIR when it is not cached."""
    try:
        archive = cached_archive()
        digest = hashlib.sha256()
        with open(archive, "rb") as handle:
            while chunk := handle.read(1 << 22):
                digest.update(chunk)
        if digest.hexdigest() != ICIJ_SHA256:
            print(f"NOTE: the archive in {CACHE_DIR} is not the one that the committed outputs come from (sha256 {digest.hexdigest()[:12]}..., expected {ICIJ_SHA256[:12]}...). The numbers will differ.")
        companies = read_member(archive, "nodes-entities.csv", ["node_id", "name", "jurisdiction_description", "status", "incorporation_date"])
        officers = read_member(archive, "nodes-officers.csv", ["node_id", "name"])
        agents = read_member(archive, "nodes-intermediaries.csv", ["node_id", "name"])
        edges = read_member(archive, "relationships.csv", ["node_id_start", "node_id_end", "rel_type"])
        return companies, officers, agents, edges[edges.rel_type.isin(["officer_of", "intermediary_of"])].reset_index(drop=True), "ICIJ Offshore Leaks, Panama Papers (full download)"
    except zipfile.BadZipFile as error:
        raise RuntimeError(f"The file {ARCHIVE} is not a valid zip file ({error}). Delete it and run the notebook again, or copy the "
                           f"archive from {ICIJ_URL} to that path.") from error
    except OSError as error:
        raise RuntimeError(f"Cannot load the ICIJ archive. It is not cached in {CACHE_DIR} and the download from {ICIJ_URL} failed "
                           f"({type(error).__name__}: {error}). Connect to the network, or copy the archive to {ARCHIVE}.") from error
    except (ValueError, KeyError) as error:
        raise RuntimeError(f"The archive in {ARCHIVE} does not have the expected files or columns ({type(error).__name__}: {error}). "
                           "ICIJ replaces the LATEST archive, so its format can change. Use the archive with the SHA-256 above.") from error


BEARER = re.compile(r"bearer|portador", re.I)
PLACEHOLDER = re.compile(r"^\s*(the\s+)?(bearer|el\s+portador)\s*$", re.I)       # only the placeholder itself: `THE BEARER (MR. NAME)` can hold a person
FIRM = re.compile(r"\b(ltd|limited|s\.?a\.?|inc|corp|corporation|company|co|llc|llp|ag|gmbh|n\.?v\.?|b\.?v\.?|trust|trustees|foundation|nominees|secretaries|"
                  r"subscribers|services|holdings|investments|management|shareholders|bearer|portador|stiftung|anstalt)\b\.?", re.I)


SHOWN = set()                                          # names that the tables may show; init() fills it



def public_names(text):
    """The names of a `; `-separated list of officers or agents that the tables may show. Any other name shows as `<person>`."""
    return "; ".join(part if part in SHOWN or part in ("none listed", "unknown", "") else "<person>" for part in str(text).split("; "))


OFFICER_CHARS = 80                                     # the model reads the first 80 characters of an officer or agent name


def build_profiles(companies, officers, agents, edges):
    holds = edges[edges.rel_type == "officer_of"].merge(officers.rename(columns={"node_id": "node_id_start", "name": "officer"}), on="node_id_start")
    serves = edges[edges.rel_type == "intermediary_of"].merge(agents.rename(columns={"node_id": "node_id_start", "name": "agent"}), on="node_id_start")
    fanout = holds.groupby("node_id_start").size()
    holds["fanout"] = holds.node_id_start.map(fanout)
    holds["hub"] = (holds.fanout >= HUB_MIN) & ~holds.officer.str.contains(BEARER)
    ordered = holds.sort_values(["node_id_end", "officer"]).drop_duplicates(["node_id_end", "officer"])
    ordered["rank"] = ordered.groupby("node_id_end").cumcount()
    first_two = ordered[ordered["rank"] < 2].groupby("node_id_end").officer.agg("; ".join).str.slice(0, 2 * OFFICER_CHARS)
    biggest = holds[holds.hub].sort_values(["node_id_end", "fanout"], ascending=[True, False]).drop_duplicates("node_id_end").set_index("node_id_end")
    profiles = companies.rename(columns={"node_id": "entity_id", "jurisdiction_description": "jurisdiction"}).copy()
    profiles["incorporated"] = profiles.incorporation_date.str[-4:].replace("", "unknown")
    profiles["intermediary"] = profiles.entity_id.map(serves.drop_duplicates("node_id_end").set_index("node_id_end").agent).fillna("unknown").str.slice(0, OFFICER_CHARS)
    profiles["officers"] = profiles.entity_id.map(first_two).fillna("none listed")
    profiles["hub_officer"] = profiles.entity_id.map(biggest.officer).fillna("")
    profiles["hub_size"] = profiles.entity_id.map(biggest.fanout).fillna(0).astype(int)
    profiles["in_cluster"] = profiles.hub_size > 0
    return profiles.drop(columns="incorporation_date"), holds, serves


def build_graph(profiles, holds, officers, agents, edges):
    """Nodes `entity` (company), `officer` (with a `hub` flag) and `agent`; edges `officer_of` and `intermediary_of`."""
    flags = holds.drop_duplicates("node_id_start").set_index("node_id_start").hub
    nodes = pl.DataFrame({
        "id": list(profiles.entity_id) + list(officers.node_id) + list(agents.node_id),
        "kind": ["entity"] * len(profiles) + ["officer"] * len(officers) + ["agent"] * len(agents),
        "hub": [False] * len(profiles) + [bool(flags.get(i, False)) for i in officers.node_id] + [False] * len(agents),
        "label": list(profiles.name) + list(officers.name) + list(agents.name)})
    pairs = edges.rename(columns={"node_id_start": "src", "node_id_end": "dst", "rel_type": "edge_type"})
    return nodes, pl.from_pandas(pairs[["src", "dst", "edge_type"]])


EVIDENCE = ["jurisdiction", "status", "incorporated", "intermediary", "officers"]

ROLES = ["holding_shell", "trust_or_foundation", "operating_company", "nominee_vehicle"]
LEVELS = ["low", "review", "escalate"]
QUESTIONS = [
    {"id": "nominee_or_shell_risk", "kind": "predicate", "text": "Does this offshore company profile look like a nominee or shell vehicle?"},
    {"id": "entity_role", "kind": "choice", "text": "Which role does this offshore company play?", "options": ROLES},
    {"id": "aml_priority", "kind": "score", "text": "How fast must a compliance officer review this offshore company?", "levels": LEVELS},
]
NOMINEE = re.compile(r"nominee|subscribers|secretaries|bearer|portador", re.I)
TRUST = re.compile(r"trust|foundation|stiftung|anstalt|fideicomiso", re.I)
SHELF = ["Shelf company", "Dissolved shelf company", "Trash company"]
HOLDING = SHELF + ["Defaulted", "Dissolved", "Inactivated", "Changed agent", "Resigned as agent", "Bad debt account"]
SECRECY = ["Niue", "Samoa", "Seychelles", "British Anguilla"]


def rulebook(frame):
    risk = frame.officers.str.contains(NOMINEE) | frame.status.isin(SHELF)
    role = np.select([frame.officers.str.contains(TRUST), frame.officers.str.contains(NOMINEE), frame.status.isin(HOLDING)],
                     ["trust_or_foundation", "nominee_vehicle", "holding_shell"], "operating_company")
    level = np.select([~risk, frame.jurisdiction.isin(SECRECY)], ["low", "escalate"], "review")
    return risk, role, level


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


class MockResponse:
    def __init__(self, body):
        self.status_code, self._body, self.headers = 200, body, {}

    def json(self):
        return self._body


class SimulatedSession:
    """A `requests`-style session. It answers OpenAI-format requests from a seeded, calibrated model that corrupts the
    held-out labels. It sends nothing over the network. It records request counts, tokens and modeled latency."""

    def __init__(self, frame, sim=SIM, takeaway=None):
        self.sim, self.takeaway, self.lock = sim, takeaway, threading.Lock()
        self.reset()
        self.labels = {tuple(r[:5]): r[5:] for r in frame[KEY + [q["id"] for q in QUESTIONS]].drop_duplicates(KEY).itertuples(index=False)}

    def reset(self):
        self.requests = self.input_tokens = self.output_tokens = 0
        self.latency_s = 0.0

    def draw(self, record, qid):
        """(answer index, probability of that answer). Fixed for each (profile, question)."""
        key = tuple(record[c] for c in KEY)
        risk, role, level = self.labels[key]
        options = CHOICES[qid]
        true_value = {"nominee_or_shell_risk": risk, "entity_role": role, "aml_priority": level}[qid]
        truth_index = (0 if true_value else 1) if qid == FIRST else options.index(true_value)
        rng = np.random.default_rng(int.from_bytes(hashlib.sha256(f"{key}|{qid}".encode()).digest()[:8], "big"))
        a, b = self.sim["confidence_beta"]["easy" if level == "low" else "hard"]
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
        if qid == "entity_role":
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
            if "<evidence-" in text:                                      # one question about one profile
                record = json_loads(re.search(r"<evidence-[0-9a-f]+>\n(.*?)\n</evidence-", text, re.S).group(1))
                question = QUESTION_BY_TEXT[re.search(r"Question: (.*)\n", text).group(1)]
                index, _ = self.draw(record, question["id"])
                content, out, wait = chr(65 + index), sim["answer_tokens"]["chat"], sim["latency_s"]["chat_request"]
            else:                                                         # a finding for one company
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
LABEL = "SIMULATED model on real ICIJ Panama Papers data" if not DEMO_LIVE else "Measured with a live model (every stage asks about a small sample and scales up)"


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
    """Show a result table under a caption that says where its numbers come from. Names of people are hidden."""
    shown = frame.copy()
    for column in ("officers", "hub_officer", "intermediary"):
        if column in shown:
            shown[column] = shown[column].map(public_names)
    display(Markdown(f"*{LABEL}*"))
    display(shown.round(digits))


def takeaway_text(prompt):
    """The simulated model writes a fixed-form finding from the evidence. A live model writes its own."""
    record = json_loads(re.search(r"\{.*\}", prompt, re.S).group(0))
    action = "escalate it to compliance" if record["verdict"] == "flagged" else "a compliance officer should check this profile"
    return (f"A {record['jurisdiction']} company ({record['status']}), set up by {public_names(record['intermediary'])[:40]} with officers {public_names(record['officers'])[:60]}; "
            f"it sits in a cluster of {record['hub_size']} companies of {public_names(record['hub_officer'])[:40]}, so {action}.")


def chat_provider(session):
    return OpenAIChatProvider(API_KEY, CHAT_MODEL, base_url=CHAT_URL, prices=PRICES, session=session, strategy="text", concurrency_limit=PARALLEL)


def decisions_provider(session, rows_per_request=1):
    return OpenAIDecisionsProvider(API_KEY, DECISIONS_MODEL, prices=PRICES, session=session, rows_per_request=rows_per_request,
                                   concurrency_limit=PARALLEL, base_url=CHAT_URL)


def decide(frame, provider, parallel, dedupe=False, **options):
    started = time.perf_counter()
    out = sem_decide(frame, questions=QUESTIONS, provider=provider, task_id="offshore-triage", evidence_cols=EVIDENCE,
                     id_col="entity_id", max_concurrency=parallel, dedupe=dedupe, **options)
    status, escalate = out.aml_priority__status, out.aml_priority.eq("escalate").fillna(False).astype(bool)
    out["verdict"] = np.select([status == "uncertain", status == "error", status == "skipped", escalate], ["uncertain", "error", "skipped", "flagged"], "cleared")
    errors, skipped = int(sum((out[f"{q['id']}__status"] == "error").sum() for q in QUESTIONS)), int((status == "skipped").sum())
    if errors:
        print(f"WARNING: {errors} answers are errors. Check the key, the model names and the rate limits. An error row is never counted as cleared.")
        assert DEMO_LIVE, "the simulated model must not return errors"
        if errors == len(out) * len(QUESTIONS):
            raise RuntimeError("every answer is an error, so the run stops here. Fix the key, the model names or the host, then run again.")
    if skipped:
        print(f"WARNING: a limit skipped {skipped} rows; the numbers cover the rows that ran")
    metas = [out[f"{q['id']}__meta"] for q in QUESTIONS]
    out["cost_usd"] = [sum(json_loads(column.iloc[i])["cost_usd"] or 0.0 for column in metas) for i in range(len(out))]
    out["deduped"] = [bool((json_loads(metas[0].iloc[i])["extra"] or {}).get("deduped_from")) for i in range(len(out))]
    return out, time.perf_counter() - started


def asked(out):
    """The rows that the model answered. A row that a limit stopped is `skipped`."""
    return out[out.aml_priority__status != "skipped"]


def quality(out):
    """Precision and recall of the escalate list, and the share of rows in the review queue. A stage that asks about
    a sample of the rows is scored on that sample."""
    out = asked(out)
    flagged = out.loc[out.verdict == "flagged", "entity_id"]
    hits = int(truth.positive.reindex(flagged).sum())
    positives = max(int(truth.positive.reindex(out.entity_id).sum()), 1)       # positives among the rows this stage asked about
    precision, recall = hits / max(len(flagged), 1), hits / positives
    queued = int(truth.positive.reindex(out.entity_id[out.verdict == "uncertain"]).sum())          # positives that wait in the compliance queue
    return {"asked": len(out), "flagged": len(flagged), "true_hits": hits, "in_queue": queued, "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / max(precision + recall, 1e-9), "uncertain_pct": 100 * (out.verdict == "uncertain").mean(),
            "errors": int(sum((out[f"{q['id']}__status"] == "error").sum() for q in QUESTIONS))}


def stage_row(session, out, local_s, parallel, scale=1.0, gfql_ms=0.0):
    """Requests, billed cost and time of one stage. The default run models the time of the provider calls."""
    cost = float(asked(out).cost_usd.sum()) * scale
    if isinstance(session, SimulatedSession):
        return {"requests": round(session.requests * scale), "cost_usd": cost, "seconds": session.latency_s / parallel * scale + gfql_ms / 1000}
    return {"requests": round(session.requests * scale), "cost_usd": cost, "seconds": local_s * scale + gfql_ms / 1000}


def sample_of(frame, live_rows):
    """The rows that a stage asks about, and the factor that scales the stage up to all rows. The default run asks about all rows. Live mode asks about a small sample."""
    cap = live_rows if DEMO_LIVE else None
    if cap is None or len(frame) <= cap:
        return frame.reset_index(drop=True), 1.0
    return frame.sample(n=cap, random_state=7).reset_index(drop=True), len(frame) / cap


def run_unbatched(frame):
    """Three chat requests for each row, one at a time, with no `dedupe`."""
    rows, scale = sample_of(frame, 50)
    session = session_for()
    out, local_s = decide(rows, chat_provider(session), parallel=1, **OPTIONS)
    if asked(out).empty:
        raise RuntimeError("a limit skipped every row of this stage, so there is nothing to scale from. Raise the budget or the limits.")
    return session, out, local_s, scale * len(rows) / len(asked(out))


COLS = ["rows", "asked", "requests", "cost_usd", "seconds", "flagged", "true_hits", "in_queue", "precision", "recall", "errors"]
results = {}


def accuracy(out, qid):
    ok = out[out[f"{qid}__status"] == "ok"]
    return float((ok[qid].astype(object) == truth[qid].reindex(ok.entity_id).to_numpy()).mean()), len(ok)


def replay(out, min_p):
    p = out.aml_priority__p
    uncertain = p < min_p
    flagged = out[~uncertain & (out.aml_priority == "escalate")].entity_id
    hits = int(truth.positive.reindex(flagged).sum())
    return {"min_p": min_p, "escalate": len(flagged), "precision": hits / max(len(flagged), 1),
            "recall": hits / max(int(truth.positive.reindex(out.entity_id).sum()), 1), "compliance queue": f"{100 * uncertain.mean():.0f}%"}


CHART_PATH = Path("gfql_semantic_fraud_aml_chart.png")


def fmt_time(seconds):
    return f"{seconds:.0f} s" if seconds < 90 else f"{seconds / 60:.0f} min" if seconds < 5400 else f"{seconds / 3600:.1f} h"


def draw_chart(results, source, simulated, flat_rows, cluster_rows, positives, path=CHART_PATH):
    names = list(results)
    labels = ["1 Naive\nall companies,\none question\nper call, serial", "2 + flat filter\n(company table,\nno graph)", "3 + GFQL\ncluster filter\n(nominee hubs)",
              "4 + 3 questions\nper call, 8 at\nonce, + dedupe", "5 + min_p\n+ sem_map"]
    colors = ["#8a94a0", "#6f8fb0", "#2f80c0", "#1d9a8a", "#e07a1f"]
    rows = [results[name] for name in names]
    tag = "simulated" if simulated else "live"
    fig, axes = plt.subplots(1, 3, figsize=(20, 6.6))
    x = np.arange(len(rows))
    for ax, key, title, fmt in ((axes[0], "seconds", f"Speed: time to answer ({tag}, log scale)", fmt_time),
                                (axes[1], "cost_usd", f"Cost: model spend in USD ({tag}, log scale)", lambda v: f"${v:,.2f}")):
        values = [row[key] for row in rows]
        bars = ax.bar(x, values, color=colors, width=0.62)
        ax.set_yscale("log")
        ax.set_ylim(min(values) / 4, max(values) * 6)
        for bar, value, i in zip(bars, values, x):
            note = fmt(value) + ("" if i == 0 else f"\n{values[0] / value:,.0f}x {'faster' if key == 'seconds' else 'cheaper'}")
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
    quality_ax.set_title(f"Quality: precision of the escalate list ({tag})", fontsize=12, loc="left", fontweight="bold")
    quality_ax.legend(loc="upper left", frameon=False, fontsize=9)
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8.5)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    kind = ("SIMULATED model on real ICIJ Panama Papers data. GFQL time and row counts are measured. Cost, model time and answers come from stated assumptions. Not a live benchmark."
            if simulated else "Measured with a live model. Every stage asks about a small sample and scales up.")
    fig.suptitle("GFQL semantic operators on the Panama Papers: scan everything, filter flat, or follow the nominee hubs", fontsize=15, fontweight="bold", x=0.01, ha="left")
    fig.text(0.01, 0.915, f"{source}  |  {results[names[0]]['rows']:,} companies, {flat_rows:,} pass the flat filter, {cluster_rows:,} are in a nominee cluster, "
             f"{positives:,} escalate there", fontsize=10.5, color="#444")
    fig.text(0.01, 0.01, kind, fontsize=9.5, color="#b03a2e" if simulated else "#1d6b3a", fontweight="bold")
    fig.text(0.99, 0.01, "Each ratio compares a stage with stage 1.", fontsize=9, color="#444", ha="right")
    fig.text(0.01, 0.045, "The positive class (escalate by the rules, in a nominee cluster) uses the same cluster evidence as the GFQL filter, so GFQL keeps all positives by design. "
             "Rules, not compliance officers, label the companies. Prompts have no company name.", fontsize=8.5, color="#444")
    fig.tight_layout(rect=(0, 0.075, 1, 0.9))
    fig.savefig(path, dpi=180)
    plt.show()


truth = None                                       # set by init(): the rulebook labels of each company
profiles = None                                    # set by init(): one profile row for each company


def session_for():
    """A fresh session: the simulator by default, a counting wrapper around `requests` in live mode."""
    if DEMO_LIVE:
        import requests
        return CountingSession(requests.Session())
    return SimulatedSession(profiles, takeaway=takeaway_text)


def init():
    """Load the archive, build the profiles, labels and graph, and publish them as module globals."""
    started = time.perf_counter()
    companies, officers, agents, edges, source = load_icij()
    print(source, "|", f"{len(companies):,} companies | {len(officers):,} officers | {len(agents):,} agents | {len(edges):,} edges | read in {time.perf_counter() - started:.0f} s")
    profiles, holds, serves = build_profiles(companies, officers, agents, edges)
    hub_ids = set(holds[holds.hub].node_id_start)
    per_officer = holds.groupby("node_id_start").agg(name=("officer", "first"), companies=("officer", "size"))
    per_agent = serves.groupby("node_id_start").agg(name=("agent", "first"), companies=("agent", "size"))
    SHOWN.update(name for name in list(per_officer[per_officer.companies >= HUB_MIN].name) + list(per_agent[per_agent.companies >= 100].name) if FIRM.search(name))
    SHOWN.update(name for name in per_officer.name if PLACEHOLDER.match(name))
    profiles["nominee_or_shell_risk"], profiles["entity_role"], profiles["aml_priority"] = rulebook(profiles)
    profiles["positive"] = (profiles.aml_priority == "escalate") & profiles.in_cluster
    truth = profiles.set_index("entity_id")[["nominee_or_shell_risk", "entity_role", "aml_priority", "in_cluster", "positive"]]
    assert truth.index.is_unique, "a company id must be unique"
    assert not set(EVIDENCE) & {"entity_id", "name", "hub_officer", "hub_size", "in_cluster", "nominee_or_shell_risk", "entity_role", "aml_priority", "positive"}, "a prompt column must not carry a name or a label"
    nodes_pl, edges_pl = build_graph(profiles, holds, officers, agents, edges)
    globals().update(companies=companies, officers=officers, agents=agents, edges=edges, SOURCE=source, profiles=profiles, holds=holds, serves=serves, hub_ids=hub_ids,
                     truth=truth, nodes_pl=nodes_pl, edges_pl=edges_pl, g=graphistry.nodes(nodes_pl, "id").edges(edges_pl, "src", "dst"))
    print(f"{len(profiles):,} companies | {len(hub_ids):,} hub officers | {nodes_pl.height:,} nodes | {edges_pl.height:,} edges")

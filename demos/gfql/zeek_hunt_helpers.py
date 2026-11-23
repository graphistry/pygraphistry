"""Helpers for the Zeek threat-hunt demo (gfql_semantic_zeek_hunt.ipynb).

The notebook shows the GFQL and semantic-operator calls. This file holds the plumbing: the Zeek log download and parse, the pivot analysis, the
rulebook that labels each request, the simulated model (a calibrated mock HTTP session), the stage runner and the chart.
Call `init()` once, then `from zeek_hunt_helpers import *`.
"""


import gzip
import hashlib
import json
import math
import os
import re
import threading
import time
import urllib.request
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
ZEEK_URL = "https://raw.githubusercontent.com/graphistry/zed-sample-data/8da0758e1ee1a8f94895d640a7646fd2c27ca803/zeek-json"
CACHE_DIR = Path("/tmp/zeek-wrccdc-cache")
URI_CHARS = 200                                       # the model reads the first 200 characters of a path


def cached(name):
    """Path of a downloaded log. A download goes to a temporary name first."""
    path = CACHE_DIR / f"{name}.json.gz"
    if not path.exists():
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        part = path.with_suffix(".part")
        with urllib.request.urlopen(f"{ZEEK_URL}/{name}.json.gz", timeout=120) as reply:
            part.write_bytes(reply.read())
        part.replace(path)
    return path


def lines(path, contains=None):
    """JSON records of a Zeek log. `contains` skips the lines without that text before parsing."""
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if contains is None or contains in line:
                yield json.loads(line)


def millis(ts):
    return ((pd.to_datetime(ts, utc=True) - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(milliseconds=1)).astype("int64")


def read_http(path):
    rows = [(f"{d['uid']}.{d.get('trans_depth', 1)}", d["ts"], d["id.orig_h"], d["id.resp_h"], d["id.resp_p"], d.get("method") or "", d.get("host") or "", (d.get("uri") or "")[:URI_CHARS],
             d.get("user_agent") or "", int(d.get("status_code") or 0)) for d in lines(path)]
    frame = pd.DataFrame(rows, columns=["uid", "ts", "src", "dst", "dport", "method", "host", "uri", "user_agent", "status"])
    frame["ts_ms"] = millis(frame.pop("ts"))
    return frame


def read_conn_edges(path):
    """One row for each (source, destination, port) of a connection that a defended host opened, with the time of the first one."""
    rows = [(d["ts"], d["id.orig_h"], d["id.resp_h"], d["id.resp_p"]) for d in lines(path, contains='"id.orig_h":"10.47.')]
    frame = pd.DataFrame(rows, columns=["ts", "src", "dst", "dport"])
    frame["ts_ms"] = millis(frame.pop("ts"))
    return frame.groupby(["src", "dst", "dport"], as_index=False).agg(first_ms=("ts_ms", "min"), connections=("ts_ms", "size"))


def load_zeek():
    """Return (http sessions, connection pairs of defended hosts, source label). Downloads the logs into CACHE_DIR when they are not cached."""
    try:
        return read_http(cached("http")), read_conn_edges(cached("conn")), "WRCCDC Zeek logs (full download)"
    except (OSError, EOFError, ValueError) as error:
        if isinstance(error, (gzip.BadGzipFile, EOFError)):
            for name in ("http", "conn"):
                (CACHE_DIR / f"{name}.json.gz").unlink(missing_ok=True)       # drop a broken cache so the next run downloads again
        raise RuntimeError(f"Cannot load the Zeek logs. They are not cached in {CACHE_DIR} and the download from {ZEEK_URL} failed "
                           f"({type(error).__name__}: {error}). Connect to the network, or copy http.json.gz and conn.json.gz to {CACHE_DIR}.") from error


INFRA_PORTS = [53, 123, 113, 137, 138, 514, 67, 68, 0, 2]


CLASSES = ["webshell_or_rce", "sqli_or_lfi", "recon_scan", "benign"]
LEVELS = ["ignore", "investigate", "contain_now"]
QUESTIONS = [
    {"id": "is_exploitation", "kind": "predicate", "text": "Is this web request an attempt to run code or read files on the server?"},
    {"id": "attack_stage", "kind": "choice", "text": "Which kind of attack is this web request?", "options": CLASSES},
    {"id": "triage_priority", "kind": "score", "text": "How fast must a security analyst act on this web request?", "levels": LEVELS},
]
EVIDENCE = ["method", "host", "uri", "user_agent", "status"]

RCE = re.compile(r"\(\)\s*\{|cmd=|exec=|/bin/(ba)?sh|;\s*(id|ls|cat|wget|curl)\b|%3b\s*(id|ls|cat)|\$\{jndi|\bsystem\(|eval\(|struts|\.cgi\?.*(\||;)", re.I)
LFI = re.compile(r"\.\./|%2e%2e|/etc/passwd|win\.ini|boot\.ini|union\s+select|union%20select|'\s*or\s+'?1|<script|%3cscript|nasl|\bselect\b.+\bfrom\b|xp_cmdshell", re.I)
SCANNER = re.compile(r"wpscan|nikto|nessus|sqlmap|masscan|nmap|zgrab|dirbuster|acunetix|openvas|w3af|gobuster", re.I)


def rulebook(method, uri, agent):
    text = f"{uri} {agent}"
    if RCE.search(text):
        return "webshell_or_rce"
    if LFI.search(text):
        return "sqli_or_lfi"
    return "recon_scan" if SCANNER.search(agent) or method in ("PROPFIND", "SEARCH", "TRACE", "PRI") else "benign"


KEY = ["method", "host", "uri", "user_agent", "status"]


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
        self.labels = {tuple(r[:5]): r[5:] for r in frame[KEY + ["attack_stage", "is_exploitation", "triage_priority"]].drop_duplicates(KEY).itertuples(index=False)}

    def reset(self):
        self.requests = self.input_tokens = self.output_tokens = 0
        self.latency_s = 0.0

    def draw(self, record, qid):
        """(answer index, probability of that answer). Fixed for each (request text, question)."""
        key = tuple(record[c] for c in KEY)
        stage, exploit, level = self.labels[key]
        options = CHOICES[qid]
        true_value = {"is_exploitation": exploit, "attack_stage": stage, "triage_priority": level}[qid]
        truth_index = (0 if true_value else 1) if qid == "is_exploitation" else options.index(true_value)
        rng = np.random.default_rng(int.from_bytes(hashlib.sha256(f"{key}|{qid}".encode()).digest()[:8], "big"))
        a, b = self.sim["confidence_beta"]["easy" if stage == "benign" else "hard"]
        p = float(np.clip(rng.beta(a, b), 1 / len(options) + 0.01, 0.995))
        wrong = [i for i in range(len(options)) if i != truth_index]
        return (truth_index if rng.random() < p else int(rng.choice(wrong))), p

    def distribution(self, qid, index, p):
        k = len(CHOICES[qid])
        return [p if i == index else (1 - p) / (k - 1) for i in range(k)]

    def decisions_answer(self, name, qid, record):
        index, p = self.draw(record, qid)
        dist, options = self.distribution(qid, index, p), CHOICES[qid]
        if qid == "is_exploitation":
            return {"type": "predicate", "name": name, "probability": dist[0]}
        if qid == "attack_stage":
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
            if "<evidence-" in text:                                      # one question about one request
                record = json_loads(re.search(r"<evidence-[0-9a-f]+>\n(.*?)\n</evidence-", text, re.S).group(1))
                question = QUESTION_BY_TEXT[re.search(r"Question: (.*)\n", text).group(1)]
                index, _ = self.draw(record, question["id"])
                content, out, wait = chr(65 + index), sim["answer_tokens"]["chat"], sim["latency_s"]["chat_request"]
            else:                                                         # a finding for one request
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
LABEL = "SIMULATED model on real WRCCDC Zeek logs" if not DEMO_LIVE else "Measured with a live model (every stage asks about a small sample and scales up)"


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


def show(frame):
    """Show a result table under a caption that says where its numbers come from."""
    display(Markdown(f"*{LABEL}*"))
    display(frame.round(3))


def takeaway_text(prompt):
    """The simulated model writes a fixed-form finding from the evidence. A live model writes its own."""
    record = json_loads(re.search(r"\{.*\}", prompt, re.S).group(0))
    action = "isolate the server and check them" if record["verdict"] == "flagged" else "an analyst should check this request"
    return (f"{record['method']} {record['uri'][:70]} to {record['victim']} (status {record['status']}); the server reached "
            f"{record['new_hosts']} new hosts after its first outside request, so {action}.")


def chat_provider(session):
    return OpenAIChatProvider(API_KEY, CHAT_MODEL, base_url=CHAT_URL, prices=PRICES, session=session, strategy="text", concurrency_limit=PARALLEL)


def decisions_provider(session, rows_per_request=1):
    return OpenAIDecisionsProvider(API_KEY, DECISIONS_MODEL, prices=PRICES, session=session, rows_per_request=rows_per_request,
                                   concurrency_limit=PARALLEL, base_url=CHAT_URL)


def decide(frame, provider, parallel, dedupe=False, **options):
    started = time.perf_counter()
    out = sem_decide(frame, questions=QUESTIONS, provider=provider, task_id="web-request-triage", evidence_cols=EVIDENCE,
                     id_col="uid", max_concurrency=parallel, dedupe=dedupe, **options)
    status, contain = out.triage_priority__status, out.triage_priority.eq("contain_now").fillna(False).astype(bool)
    out["verdict"] = np.select([status == "uncertain", status == "error", status == "skipped", contain], ["uncertain", "error", "skipped", "flagged"], "cleared")
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
    return out, time.perf_counter() - started


def asked(out):
    """The rows that the model answered. A row that a limit stopped is `skipped`."""
    return out[out.triage_priority__status != "skipped"]


def quality(out):
    """Precision and recall of the contain_now list, and the share of rows in the review queue. A stage that asks about
    a sample of the rows is scored on that sample."""
    out = asked(out)
    flagged = out.loc[out.verdict == "flagged", "uid"]
    hits = int(truth.positive.reindex(flagged).sum())
    positives = max(int(truth.positive.reindex(out.uid).sum()), 1)         # positives among the rows this stage asked about
    precision, recall = hits / max(len(flagged), 1), hits / positives
    queued = int(truth.positive.reindex(out.uid[out.verdict == "uncertain"]).sum())             # positives that wait in the analyst queue
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
    """Three chat requests for each row, one at a time."""
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
    return float((ok[qid].astype(object) == truth[qid].reindex(ok.uid).to_numpy()).mean()), len(ok)


def replay(out, min_p):
    p = out.triage_priority__p
    uncertain = p < min_p
    flagged = out[~uncertain & (out.triage_priority == "contain_now")].uid
    hits = int(truth.positive.reindex(flagged).sum())
    return {"min_p": min_p, "contain_now": len(flagged), "precision": hits / max(len(flagged), 1),
            "recall": hits / max(int(truth.positive.reindex(out.uid).sum()), 1), "analyst queue": f"{100 * uncertain.mean():.0f}%"}


def build_graph(http, conn_pairs):
    """Nodes are hosts with a `zone`; edges are `http` (one per web request) and `conn` (one per connection pair, with an `is_pivot` flag)."""
    hosts = sorted(set(http.src) | set(http.dst) | set(conn_pairs.src) | set(conn_pairs.dst))
    nodes = pl.DataFrame({"id": hosts, "zone": ["blue" if h.startswith("10.47.") else "outside" for h in hosts]})
    web = http[["src", "dst", "uid"]].assign(edge_type="http", is_pivot=False)
    pairs = conn_pairs[["src", "dst"]].assign(uid="", edge_type="conn", is_pivot=conn_pairs.is_pivot)
    return nodes, pl.from_pandas(pd.concat([web, pairs], ignore_index=True))


CHART_PATH = Path("gfql_semantic_zeek_hunt_chart.png")


def fmt_time(seconds):
    return f"{seconds:.0f} s" if seconds < 90 else f"{seconds / 60:.0f} min" if seconds < 5400 else f"{seconds / 3600:.1f} h"


def draw_chart(results, source, simulated, flat_rows, pivot_rows, positives, path=CHART_PATH):
    names = list(results)
    labels = ["1 Naive\nall requests,\none question per\ncall, serial", "2 + flat filter\n(plain table,\nno graph)", "3 + GFQL\npivot filter\n(hosts that pivoted)",
              "4 + 3 questions\nper call,\n8 at once", "5 + min_p\n+ sem_map"]
    colors = ["#8a94a0", "#6f8fb0", "#2f80c0", "#1d9a8a", "#e07a1f"]
    rows = [results[name] for name in names]
    tag = "simulated" if simulated else "live"
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.4))
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
        text = "not measured" if np.isnan(row["precision"]) else f"{100 * row['precision']:.0f}% precise\n{row['uncertain_pct']:.0f}% to review" + (f"\n+{row['in_queue']:,} positives queued" if row["in_queue"] else "")
        top = 0 if np.isnan(row["precision"]) else 100 * max(row["precision"], row["recall"])     # clear of the bar and the recall marker
        quality_ax.text(bar.get_x() + bar.get_width() / 2, top + 3.5, text,
                        ha="center", va="bottom", fontsize=9, fontweight="bold")
    quality_ax.set_ylim(0, 128)
    quality_ax.set_yticks([0, 25, 50, 75, 100])
    quality_ax.set_yticklabels([f"{t}%" for t in (0, 25, 50, 75, 100)])
    quality_ax.set_title(f"Quality: precision of the contain_now list ({tag})", fontsize=12, loc="left", fontweight="bold")
    quality_ax.legend(loc="upper left", frameon=False, fontsize=9)
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    kind = ("SIMULATED model on real WRCCDC Zeek logs. GFQL time and row counts are measured. Cost, model time and answers come from stated assumptions. Not a live benchmark."
            if simulated else "Measured with a live model. Every stage asks about a small sample and scales up.")
    fig.suptitle("GFQL semantic operators on WRCCDC Zeek logs: scan everything, filter flat, or follow the pivot", fontsize=15, fontweight="bold", x=0.01, ha="left")
    fig.text(0.01, 0.915, f"{source}  |  {results[names[0]]['rows']:,} web requests, {flat_rows:,} pass the flat filter, {pivot_rows:,} hit a server that pivoted, "
             f"{positives:,} exploit attempts there", fontsize=10.5, color="#444")
    fig.text(0.01, 0.01, kind, fontsize=9.5, color="#b03a2e" if simulated else "#1d6b3a", fontweight="bold")
    fig.text(0.99, 0.01, "Each ratio compares a stage with stage 1.", fontsize=9, color="#444", ha="right")
    fig.text(0.01, 0.045, "The positive class (exploit attempts on a pivoting server) uses the same pivot evidence as the GFQL filter, so GFQL keeps all positives by design. "
             "Stage 3 reads as many requests as stage 2: it changes which ones, not how many.", fontsize=8.5, color="#444")
    fig.tight_layout(rect=(0, 0.075, 1, 0.9))
    fig.savefig(path, dpi=180)
    plt.show()


http = truth = None                                # set by init(): the web requests, and the rulebook labels of each request


def session_for():
    """A fresh session: the simulator by default, a counting wrapper around `requests` in live mode."""
    if DEMO_LIVE:
        import requests
        return CountingSession(requests.Session())
    return SimulatedSession(http, takeaway=takeaway_text)


def init():
    """Load the logs, find the pivoting servers, label each request and build the graph. Publishes the results as module globals."""
    started = time.perf_counter()
    http, conn_pairs, source = load_zeek()
    print(source, "|", f"{len(http):,} web requests | {len(conn_pairs):,} connection pairs from defended hosts | read in {time.perf_counter() - started:.0f} s")

    http["inbound"] = ~http.src.str.startswith("10.47.") & http.dst.str.startswith("10.47.")
    first_hit = http[http.inbound].groupby("dst").ts_ms.min()                       # time of the first outside request to each defended host
    conn_pairs = conn_pairs[conn_pairs.src.isin(first_hit.index)].copy()            # only victims can pivot
    first_contact = conn_pairs.groupby(["src", "dst"]).first_ms.transform("min")      # first time that the victim contacted this host on any port
    conn_pairs["is_pivot"] = (first_contact >= conn_pairs.src.map(first_hit)) & ~conn_pairs.dport.isin(INFRA_PORTS) & (conn_pairs.dst != conn_pairs.src)
    pivot_targets = conn_pairs[conn_pairs.is_pivot].groupby("src").dst.nunique()   # new hosts that each victim reached
    pivot_victims = set(pivot_targets.index)

    texts = http[KEY].drop_duplicates()
    texts["attack_stage"] = [rulebook(r.method, r.uri, r.user_agent) for r in texts.itertuples(index=False)]
    http = http.merge(texts, on=KEY, how="left")
    http["is_exploitation"] = http.attack_stage.isin(["webshell_or_rce", "sqli_or_lfi"])
    http["triage_priority"] = np.select([http.is_exploitation, http.attack_stage == "recon_scan"], ["contain_now", "investigate"], "ignore")
    http["on_pivot_victim"] = http.inbound & http.dst.isin(pivot_victims)
    http["positive"] = http.is_exploitation & http.on_pivot_victim
    truth = http.set_index("uid")[["attack_stage", "is_exploitation", "triage_priority", "on_pivot_victim", "positive"]]
    assert truth.index.is_unique, "a session id must be unique"
    assert not set(EVIDENCE) & {"src", "dst", "inbound", "attack_stage", "is_exploitation", "triage_priority", "on_pivot_victim", "positive"}, "a prompt column must not carry an address or a label"
    nodes_pl, edges_pl = build_graph(http, conn_pairs)
    globals().update(http=http, conn_pairs=conn_pairs, SOURCE=source, first_hit=first_hit, pivot_targets=pivot_targets, pivot_victims=pivot_victims, truth=truth,
                     nodes_pl=nodes_pl, edges_pl=edges_pl, g=graphistry.nodes(nodes_pl, "id").edges(edges_pl, "src", "dst"))
    print(f"{len(http):,} web requests | {len(first_hit)} victims | {len(pivot_victims)} pivoting victims | {nodes_pl.height:,} nodes | {edges_pl.height:,} edges")

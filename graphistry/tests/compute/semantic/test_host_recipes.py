"""Smoke pins for the provider recipes in the docs: a fake HTTP session, no network, no keys.

Each pin builds the provider exactly as its recipe does and checks the URL, the auth header and the payload that a host sees.
"""
import math
from typing import Dict, List

import pytest

from graphistry.compute.semantic.providers.chat import OpenAIChatProvider
from graphistry.compute.semantic.providers.openai_decisions import OpenAIDecisionsProvider
from graphistry.tests.compute.semantic.http_test_utils import CHOICE, PRED, SCORE, FakeResponse, FakeSession, request


def decisions_reply(body: Dict[str, object]) -> FakeResponse:
    answers: List[Dict[str, object]] = []
    for q in body["questions"]:  # type: ignore[union-attr]
        if q["type"] == "predicate":
            answers.append({"type": "predicate", "name": q["name"], "probability": 0.8})
        elif q["type"] == "choice":
            answers.append({"type": "choice", "name": q["name"], "choice": q["choices"][0]["value"],
                            "probabilities": [{"value": c["value"], "probability": p} for c, p in zip(q["choices"], (0.9, 0.06, 0.04))]})
        else:
            answers.append({"type": "score", "name": q["name"],
                            "probabilities": [{"value": i, "label": lv["label"], "probability": p}
                                              for i, (lv, p) in enumerate(zip(q["levels"], (0.1, 0.2, 0.7)))]})
    usage = {"input_tokens": 100, "input_tokens_details": {"cached_tokens": 0}, "output_tokens": 10}
    return FakeResponse(200, {"model": "vendor/model", "answers": answers, "usage": usage})


def chat_logprob_reply(letter: str, alternatives: Dict[str, float]) -> FakeResponse:
    top = [{"token": token, "logprob": math.log(p)} for token, p in alternatives.items()]
    choice = {"message": {"role": "assistant", "content": letter},
              "logprobs": {"content": [{"token": letter, "logprob": math.log(alternatives[letter]), "top_logprobs": top}]}}
    return FakeResponse(200, {"model": "llama3.1", "choices": [choice],
                              "usage": {"prompt_tokens": 50, "completion_tokens": 1}})


class TestHostRecipes:

    def test_openrouter_native_decisions_recipe(self) -> None:
        session = FakeSession([decisions_reply])
        provider = OpenAIDecisionsProvider("or-key", "vendor/model", base_url="https://openrouter.ai/api/v1",
                                           session=session, retry_backoff_s=0.0)  # type: ignore[arg-type]
        results = provider.decide_batch([request()], deadline_s=None, budget=None)
        post = session.posts[0]
        assert post["url"] == "https://openrouter.ai/api/v1/decisions" and len(session.posts) == 1
        assert post["headers"]["Authorization"] == "Bearer or-key"  # type: ignore[index]
        assert post["json"]["model"] == "vendor/model"  # type: ignore[index]
        assert [q["name"] for q in post["json"]["questions"]] == [PRED.id, CHOICE.id, SCORE.id]  # type: ignore[index]
        assert [r.status for r in results] == ["ok", "ok", "ok"]
        assert {r.probability_kind for r in results} == {"native_decision"}

    def test_ollama_chat_recipe_sends_logprobs_to_the_local_endpoint(self) -> None:
        session = FakeSession([chat_logprob_reply("A", {"A": 0.8, "B": 0.15})])
        provider = OpenAIChatProvider("ollama", "llama3.1", base_url="http://localhost:11434/v1", strategy="logprobs",
                                      session=session, retry_backoff_s=0.0)  # type: ignore[arg-type]
        results = provider.decide_batch([request((PRED,))], deadline_s=None, budget=None)
        post = session.posts[0]
        assert post["url"] == "http://localhost:11434/v1/chat/completions" and len(session.posts) == 1
        assert post["headers"]["Authorization"] == "Bearer ollama"  # type: ignore[index]
        assert post["json"]["model"] == "llama3.1" and post["json"]["logprobs"] is True  # type: ignore[index]
        assert results[0].value is True and results[0].probability_kind == "token_logprob"
        assert results[0].distribution == pytest.approx({"true": 0.8 / 0.95, "false": 0.15 / 0.95})

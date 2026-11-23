"""Fake HTTP sessions for provider tests: no network, no keys."""
import json
from typing import Callable, Dict, List, Optional, Union

from graphistry.compute.semantic.contract import DecisionQuestion, DecisionRequest


class FakeResponse:
    def __init__(self, status: int, body: object = None, headers: Optional[Dict[str, str]] = None,
                 text: Optional[str] = None) -> None:
        self.status_code = status
        self._body = body
        self.headers = headers or {}
        self._text = text

    def json(self) -> object:
        return json.loads(self._text) if self._text is not None else self._body


Reply = Union[FakeResponse, Exception, Callable[[Dict[str, object]], FakeResponse]]


class FakeSession:
    """Replays queued replies (the last one repeats) and records every POST."""

    def __init__(self, replies: List[Reply]) -> None:
        self.replies = list(replies)
        self.posts: List[Dict[str, object]] = []

    def post(self, url: str, *, json: Dict[str, object], headers: Dict[str, str], timeout: float) -> FakeResponse:
        self.posts.append({"url": url, "json": json, "headers": headers, "timeout": timeout})
        reply = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        if isinstance(reply, Exception):
            raise reply
        return reply(json) if callable(reply) else reply


PRED = DecisionQuestion(id="p", kind="predicate", text="Is it damaged?")
CHOICE = DecisionQuestion(id="c", kind="choice", text="Which team?", options=("billing", "tech", "other"))
SCORE = DecisionQuestion(id="s", kind="score", text="How severe?", levels=("low", "mid", "high"))


def request(questions=(PRED, CHOICE, SCORE), request_id: str = "r0",
            evidence: object = "a box arrived crushed") -> DecisionRequest:
    return DecisionRequest(request_id=request_id, evidence=evidence, questions=tuple(questions),  # type: ignore[arg-type]
                           task_id="t", task_version="1")

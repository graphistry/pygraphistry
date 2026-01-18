"""Optional OpenTelemetry setup for benchmarks.

This keeps deps optional: if opentelemetry is missing, it no-ops.
"""

from __future__ import annotations

import os
import sys
from typing import Optional


def setup_tracer() -> bool:
    if os.environ.get("GRAPHISTRY_DF_EXECUTOR_OTEL", "").strip().lower() not in {"1", "true", "yes", "on"}:
        return False

    try:
        from opentelemetry import trace  # type: ignore
        from opentelemetry.sdk.trace import TracerProvider  # type: ignore
        from opentelemetry.sdk.trace.export import (  # type: ignore
            BatchSpanProcessor,
            ConsoleSpanExporter,
            SimpleSpanProcessor,
        )
        from opentelemetry.sdk.resources import Resource  # type: ignore
    except Exception:
        print("OpenTelemetry SDK not installed; spans will not be exported.", file=sys.stderr)
        return False

    exporter_kind = os.environ.get("GRAPHISTRY_DF_EXECUTOR_OTEL_EXPORTER", "console").strip().lower()
    processor = None

    if exporter_kind == "otlp":
        exporter = _make_otlp_exporter()
        if exporter is None:
            return False
        processor = BatchSpanProcessor(exporter)
    else:
        processor = SimpleSpanProcessor(ConsoleSpanExporter())

    provider = trace.get_tracer_provider()
    if not hasattr(provider, "add_span_processor"):
        service_name = os.environ.get("OTEL_SERVICE_NAME", "graphistry")
        provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
        trace.set_tracer_provider(provider)

    provider.add_span_processor(processor)
    return True


def _make_otlp_exporter() -> Optional[object]:
    endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "").strip()
    try:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (  # type: ignore
            OTLPSpanExporter,
        )
        return OTLPSpanExporter(endpoint=endpoint or None)
    except Exception:
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (  # type: ignore
                OTLPSpanExporter,
            )
            return OTLPSpanExporter(endpoint=endpoint or None)
        except Exception:
            print("OTLP exporter not available; install opentelemetry-exporter-otlp.", file=sys.stderr)
            return None

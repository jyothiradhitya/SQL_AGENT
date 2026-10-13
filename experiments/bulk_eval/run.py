"""
Runs the bulk_eval question bank through the REAL chatbot pipeline
(orchestrator -> SQL agent -> MCP server -> Postgres, live Ollama) and
grades every answer.

Resumable: results are appended to results.jsonl as each question
finishes, and already-finished ids are skipped on restart.

    PYTHONPATH=. uv run python experiments/bulk_eval/run.py --workers 6 [--limit N] [--ids Q0001,Q0002] [--out results.jsonl]
"""

import argparse
import asyncio
import contextvars
import json
import logging
import sys
import time
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from lms_chatbot.logging_config import setup_logging  # noqa: E402
from lms_chatbot.mcp_client.client import MCPClient  # noqa: E402
from lms_chatbot.orchestrator import answer_question  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from grade import grade  # noqa: E402

HERE = Path(__file__).parent
_current = contextvars.ContextVar("current_trace", default=None)

# Log lines worth keeping per question (all emitted in this process by
# the orchestrator / SQL agent, so the contextvar identifies the question).
_KEEP = ("SQL agent start", "GENERATED_SQL", "SQL_RESULT status", "NO_TOOL_CALL",
         "GAVE_UP", "ANSWER_CUT_OFF", "raised an exception", "UNKNOWN_TOOL", "failed:")


class _TraceHandler(logging.Handler):
    def emit(self, record):
        trace = _current.get()
        if trace is None:
            return
        msg = record.getMessage()
        if any(k in msg for k in _KEEP):
            trace.append(msg[:3000])


async def worker(name, queue, items_by_id, out_path, lock, stats, timeout):
    async with MCPClient() as client:
        while True:
            try:
                item = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            trace: list[str] = []
            token = _current.set(trace)
            start = time.monotonic()
            result = {"id": item["id"]}
            try:
                result["answer"] = await asyncio.wait_for(
                    answer_question(item["question"], client, []), timeout=timeout)
            except asyncio.TimeoutError:
                result["timeout"] = True
            except Exception as e:  # noqa: BLE001 - recorded, graded as ERROR
                result["exception"] = f"{type(e).__name__}: {e}"[:2000]
            finally:
                _current.reset(token)
            result["duration"] = round(time.monotonic() - start, 1)
            result["trace"] = trace
            result["verdict"] = grade(item, result)
            async with lock:
                with out_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
                stats[result["verdict"]] = stats.get(result["verdict"], 0) + 1
                done = sum(v for k, v in stats.items() if k != "_total")
                print(f"[{done}/{stats['_total']}] {item['id']} {result['verdict']:<9} "
                      f"{result['duration']:>6}s  {item['question'][:70]}", flush=True)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--ids")
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--questions", default=str(HERE / "questions.json"))
    ap.add_argument("--out", default=str(HERE / "results.jsonl"))
    args = ap.parse_args()

    setup_logging()
    handler = _TraceHandler(level=logging.INFO)
    logging.getLogger("lms_chatbot").addHandler(handler)

    items = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    if args.ids:
        wanted = set(args.ids.split(","))
        items = [x for x in items if x["id"] in wanted]
    out_path = Path(args.out)
    done = set()
    if out_path.exists():
        done = {json.loads(l)["id"] for l in out_path.read_text(encoding="utf-8").splitlines() if l.strip()}
    items = [x for x in items if x["id"] not in done]
    if args.limit:
        items = items[: args.limit]

    queue = asyncio.Queue()
    for x in items:
        queue.put_nowait(x)
    stats = {"_total": len(items)}
    lock = asyncio.Lock()
    by_id = {x["id"]: x for x in items}
    print(f"Running {len(items)} questions ({len(done)} already done) with {args.workers} workers", flush=True)
    await asyncio.gather(*(worker(f"w{i}", queue, by_id, out_path, lock, stats, args.timeout)
                           for i in range(args.workers)))
    stats.pop("_total")
    print("DONE", stats, flush=True)


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    asyncio.run(main())

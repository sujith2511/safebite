"""Intern client: the "Detective" feature. All Intern calls live here.

Intern (github.com/nace-ai/intern) is Nace's agent harness. System 2 (a DeepInfra model)
does the work; System 1 (Drex) watches every turn and checks every tool call.

When SafeBite is unsure about an allergen (CHECK / NOT SAFE, or Drex and the LLM disagree),
the app asks Intern to investigate. Intern gets a throwaway folder with the product's facts and
our allergen reference, reads and searches them with its tools, and writes a short plain-English
explanation.

SAFETY: Intern's answer is an explanation only. It never changes the verdict, and nothing
here can return "safe".
"""
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from . import config  # noqa: F401  (loads .env into os.environ)
from .synonyms import EXCLUSIONS, SYNONYMS

SYSTEM1 = "drex-latest"
SYSTEM2 = "deepinfra/Qwen/Qwen3.8-27B"
TIMEOUT_S = 120
MAX_TURNS = 8


def intern_path() -> str | None:
    return shutil.which("intern") or (
        str(Path.home() / ".local/bin/intern") if (Path.home() / ".local/bin/intern").exists() else None)


def _reference_md() -> str:
    lines = ["# Allergen reference (SafeBite dictionary)", ""]
    for allergen, names in SYNONYMS.items():
        lines.append(f"## {allergen}")
        lines.append("Hidden names: " + ", ".join(names))
        if EXCLUSIONS.get(allergen):
            lines.append("NOT this allergen: " + ", ".join(EXCLUSIONS[allergen]))
        lines.append("")
    lines += ["## Notes",
              "- Lecithin, natural flavors, spices, and 'emulsifier' may come from soy, egg, milk or other",
              "  sources when the label does not say. Unknown source means the allergen cannot be ruled out.",
              "- 'May contain' / 'processed in a facility with' means cross-contact risk.",
              "- Highly refined oils are sometimes exempt from labeling, but sensitive people should check."]
    return "\n".join(lines)


def _prompt(allergen: str, severity: str) -> str:
    name = allergen.replace("_", " ")
    return (
        f"You are SafeBite's allergen detective. The user is allergic to {name} ({severity}).\n"
        f"SafeBite could not clear this product for {name}. Investigate why.\n"
        "Steps: 1) read product.json (the label, the verdict and what each model answered). "
        f"2) search allergen_reference.md for anything related to {name}. "
        "3) answer.\n"
        "Answer in at most 4 short sentences, plain English, for a worried shopper: name the exact "
        f"ingredient or statement that is the {name} risk (or say the label is unclear), where it "
        "comes from, and what the user should do. Never say the product is safe. Do not write any files."
    )


def investigate(product: dict, allergen: str, severity: str = "severe",
                verdict: dict | None = None, answers: dict | None = None) -> dict:
    """Run Intern headlessly on one allergen of one product.

    Returns {"ok", "answer", "seconds", "turns", "system1_checks", "tool_calls", "error"}.
    """
    started = time.perf_counter()
    exe = intern_path()
    if not exe:
        return {"ok": False, "error": "intern is not installed (uv tool install ... intern-agent)",
                "answer": None, "seconds": 0.0}

    with tempfile.TemporaryDirectory(prefix="safebite_intern_") as ws:
        Path(ws, "product.json").write_text(json.dumps({
            "product": product.get("product") or product.get("product_name"),
            "ingredients": product.get("ingredients"),
            "contains_statement": product.get("contains_statement"),
            "may_contain_statement": product.get("may_contain_statement"),
            "evidence": product.get("evidence"),
            "allergen_under_investigation": allergen,
            "severity": severity,
            "safebite_verdict": verdict,
            "model_answers": answers,
        }, indent=2))
        Path(ws, "allergen_reference.md").write_text(_reference_md())

        cmd = [exe, "-p", "--json", "--system1", SYSTEM1, "--system2", SYSTEM2,
               "--cwd", ws, "--no-context-files", "--no-session", "--no-memory",
               "--max-turns", str(MAX_TURNS), _prompt(allergen, severity)]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT_S,
                                  env=os.environ.copy(), cwd=ws)
        except subprocess.TimeoutExpired:
            return {"ok": False, "error": f"Intern timed out after {TIMEOUT_S}s", "answer": None,
                    "seconds": time.perf_counter() - started}

    end, tool_calls = None, []
    for line in proc.stdout.splitlines():
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if ev.get("type") == "run_end":
            end = ev
        elif ev.get("type") == "tool_call" or (ev.get("type") == "message" and
                                               ev.get("message", {}).get("role") == "tool"):
            tool_calls.append(ev)
    seconds = time.perf_counter() - started
    if not end or not (end.get("final_text") or "").strip():
        return {"ok": False, "error": f"Intern gave no answer (exit {proc.returncode}): "
                                      f"{proc.stderr.strip()[-300:]}",
                "answer": None, "seconds": seconds}
    s1 = end.get("system1") or {}
    return {
        "ok": True,
        "answer": end["final_text"].strip(),
        "seconds": seconds,
        "turns": end.get("turns"),
        "system1_checks": s1.get("asks", 0),
        "tool_calls": len(tool_calls),
        "reason": end.get("reason"),
        "error": None,
    }

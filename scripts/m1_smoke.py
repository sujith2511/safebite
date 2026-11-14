"""M1 check: one raw call each to NDI, Drex and the LLM; print the raw JSON.

Usage:  python scripts/m1_smoke.py [path/to/label.jpg] [--only ndi|drex|llm]
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from safebite import config, drex_client, llm_client, ndi_client  # noqa: E402

SAMPLE_STATE = {
    "product": "Choco Crunch Bar",
    "ingredients": ["sugar", "wheat flour", "cocoa butter", "whey powder", "soy lecithin"],
    "contains_statement": "Contains: wheat, soy",
    "may_contain_statement": "May contain traces of peanuts",
    "evidence": {
        "milk": ["whey powder = milk (dictionary)"],
        "peanuts": ["may-contain statement mentions peanuts"],
    },
}
SAMPLE_ALLERGENS = ["milk", "peanuts", "sesame"]


def banner(name: str) -> None:
    print(f"\n{'=' * 20} {name} {'=' * 20}")


def make_synthetic_label(path: Path) -> None:
    from PIL import Image, ImageDraw, ImageFont
    img = Image.new("RGB", (1200, 520), "white")
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 34)
    except OSError:
        font = ImageFont.load_default()
    lines = ["INGREDIENTS: Sugar, wheat flour, cocoa butter,",
             "whey powder, soy lecithin, natural flavour.",
             "",
             "CONTAINS: WHEAT, MILK, SOY.",
             "May contain traces of peanuts."]
    for i, line in enumerate(lines):
        d.text((50, 50 + i * 70), line, fill="black", font=font)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


async def smoke_ndi(image_path: Path) -> None:
    banner("NDI parse")
    result = await ndi_client.raw_parse(image_path.read_bytes(), image_path.name)
    print(json.dumps(result, indent=2)[:6000])


async def smoke_drex() -> None:
    banner(f"Drex {config.DREX_MODEL}")
    questions = {a: drex_client.allergen_question(a) for a in SAMPLE_ALLERGENS}
    resp, ms = await drex_client.raw_systemone(SAMPLE_STATE, questions)
    print(json.dumps(resp, indent=2))
    print(f"client latency: {ms:.0f} ms")


async def smoke_llm() -> None:
    banner(f"LLM {config.LLM_PROVIDER} / {llm_client.model_id()}")
    user = ("Allergens: " + ", ".join(SAMPLE_ALLERGENS) + "\nProduct state:\n"
            + json.dumps(SAMPLE_STATE, indent=2)
            + '\nReply as {"<allergen>": {"label": "...", "confidence": 0.0}, ...}')
    out = await llm_client.raw_complete(user)
    print(json.dumps(out, indent=2))


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("image", nargs="?", default=str(ROOT / "data/demo_photos/synthetic_label.png"))
    ap.add_argument("--only", choices=["ndi", "drex", "llm"])
    args = ap.parse_args()
    image = Path(args.image)
    if not image.exists():
        make_synthetic_label(image)

    failures = 0
    for name, fn in [("ndi", lambda: smoke_ndi(image)), ("drex", smoke_drex), ("llm", smoke_llm)]:
        if args.only and args.only != name:
            continue
        try:
            await fn()
        except Exception as e:  # report and continue so one bad key doesn't hide the others
            failures += 1
            print(f"FAILED {name}: {type(e).__name__}: {e}")
    print(f"\nM1 smoke: {failures} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

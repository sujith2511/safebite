"""SafeBite: Streamlit app. Tabs: Scan (live demo) | Benchmark (Track 1 evidence).

Run: .venv/bin/streamlit run app.py
"""
import asyncio
import json
from pathlib import Path

import streamlit as st

from safebite import config, intern_client, llm_client, pipeline
from safebite.decide import READABLE_MIN_YES, verdict
from safebite.draw import annotate
from safebite.models import ALLERGENS

ROOT = Path(__file__).resolve().parent
PHOTOS = ROOT / "data" / "demo_photos"
DISCLAIMER = ("SafeBite is an assistant, not a guarantee. Always double-check the label and "
              "follow your doctor's advice.")
BANNERS = {
    "safe": ("#15803d", "#dcfce7", "✅ SAFE", "No problem found for your allergens."),
    "check": ("#b45309", "#fef3c7", "⚠️ CHECK CAREFULLY", "Something is unclear. Read the label yourself."),
    "not_safe": ("#b91c1c", "#fee2e2", "⛔ NOT SAFE", "This product contains or may contain your allergen."),
    "retake": ("#4b5563", "#f3f4f6", "📷 RETAKE PHOTO",
               "The label is not readable enough to decide. Please retake the photo."),
}
STATUS_TXT = {"safe": "✅ safe", "check": "⚠️ check", "not_safe": "⛔ not safe"}
NICE = {a: a.replace("_", " ").title() for a in ALLERGENS}

st.set_page_config(page_title="SafeBite", page_icon="🥜", layout="wide")


def banner(kind: str, extra: str = "") -> None:
    fg, bg, title, text = BANNERS[kind]
    st.markdown(
        f"<div style='background:{bg};border:3px solid {fg};border-radius:14px;padding:18px 22px;"
        f"margin:6px 0 14px'><div style='color:{fg};font-size:2.1rem;font-weight:800'>{title}</div>"
        f"<div style='color:#111;font-size:1.05rem'>{text} {extra}</div></div>",
        unsafe_allow_html=True)


# --- sidebar: allergy profile ------------------------------------------------------
with st.sidebar:
    st.header("Your allergy profile")
    chosen = st.multiselect("Allergens", ALLERGENS, default=["milk", "peanuts", "soy"],
                            format_func=NICE.get)
    severities = {}
    for a in chosen:
        severities[a] = st.radio(NICE[a], ["severe", "mild"], horizontal=True, key=f"sev_{a}")
    st.divider()
    use_cache = st.toggle("Use saved results (offline demo)", value=True,
                          help="Re-use earlier NDI / Drex / LLM answers for the same photo. "
                               "Latencies shown are the ones measured on the first run.")
    llm_options = [m for m in ("claude-sonnet-5-5", llm_client.model_id())
                   if config.provider_for(m)[1]] or [llm_client.model_id()]
    llm_model = st.selectbox("Compare Drex with", list(dict.fromkeys(llm_options)))
    st.caption(f"Decision model: **{config.DREX_MODEL}**")

st.title("SafeBite")
st.caption("Snap the ingredient label. SafeBite tells you if it is safe for your allergies. "
           "Powered by Nace NDI (reading), Drex (deciding) and Intern (investigating).")
tab_scan, tab_bench = st.tabs(["📷 Scan", "📊 Benchmark"])

# --- Scan tab -----------------------------------------------------------------------
with tab_scan:
    left, right = st.columns([1, 1])
    with left:
        source = st.radio("Photo", ["Demo photo", "Upload", "Camera"], horizontal=True)
        raw = None
        if source == "Demo photo":
            files = sorted(p.name for p in PHOTOS.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
            pick = st.selectbox("Choose a packet", files)
            raw = (PHOTOS / pick).read_bytes() if pick else None
        elif source == "Upload":
            up = st.file_uploader("Back of the packet", type=["jpg", "jpeg", "png"])
            raw = up.getvalue() if up else None
        else:
            cam = st.camera_input("Take a photo of the ingredients")
            raw = cam.getvalue() if cam else None
        go = st.button("Check label", type="primary", disabled=raw is None or not chosen,
                       width="stretch")
        if not chosen:
            st.info("Pick at least one allergen in the sidebar.")

    if go and raw:
        img_bytes, img = pipeline.normalize_image(raw)
        if source != "Demo photo":  # keep user photos for debugging and the cached demo
            up_dir = ROOT / "data" / "uploads"
            up_dir.mkdir(parents=True, exist_ok=True)
            (up_dir / f"{pipeline.image_key(img_bytes)[:12]}_{img.width}x{img.height}.jpg").write_bytes(img_bytes)
        with st.status("Checking label ...", expanded=True) as status:
            st.write("Reading label (NDI) and checking photo quality (Drex) ...")
            read = asyncio.run(pipeline.read(img_bytes, img, use_cache))
            scan = {"img": img, "read": read, "drex": None, "llm": None, "llm_pending": False}
            if read["readable_p_yes"] is not None and read["readable_p_yes"] >= READABLE_MIN_YES \
                    and read["ingredients"]:
                st.write("Deciding (Drex) ...")
                scan["state"] = pipeline.state_from_read(read)
                scan["drex"], scan["drex_cached"] = pipeline.decide_drex(scan["state"], use_cache)
                scan["llm_pending"] = True
            status.update(label="Done", state="complete", expanded=False)
        st.session_state["scan"] = scan
        st.session_state["intern"] = {}

    scan = st.session_state.get("scan")
    if scan:
        read = scan["read"]
        p_yes = read["readable_p_yes"] if read["ingredients"] else None
        v_drex = verdict(scan["drex"], severities, p_yes)
        with left:
            if v_drex.overall == "retake":
                w, h = scan["img"].size
                if read.get("readable_error"):
                    why = f"(Reason: {read['readable_error'][:160]})"
                elif not read["ingredients"]:
                    why = (f"(Reason: NDI found no ingredient list in this {w}x{h} photo. "
                           "Get closer so the ingredient text fills the frame.)")
                else:
                    why = (f"(Drex readability: P(readable) = {p_yes:.2f}, needs {READABLE_MIN_YES}. "
                           f"Photo {w}x{h}.)")
                if w * h < 1_000_000:
                    why += (" Tip: this photo is low resolution. On a phone, use Upload -> "
                            "Take Photo instead of Camera for a sharper image.")
                banner("retake", why)
            else:
                banner(v_drex.overall, f"(decided by Drex in {scan['drex'].latency_ms:.0f} ms)")
            marks = pipeline.marks_for(read, v_drex.per_allergen) if v_drex.overall != "retake" else []
            st.image(annotate(scan["img"], marks, read.get("block_box")),
                     caption="Red = contains, yellow = may contain. Boxes are placed from NDI's "
                             "paragraph location.", width="stretch")

        with right:
            if v_drex.overall != "retake":
                st.subheader("Drex vs frontier LLM")
                c1, c2 = st.columns(2)
                with c1:
                    r = scan["drex"]
                    st.markdown(f"**Drex Max v1.1**  \n⏱ **{r.latency_ms:,.0f} ms** · "
                                f"💲{(r.cost_usd or 0):.6f}")
                    for a in severities:
                        st.write(f"{NICE[a]}: {STATUS_TXT[v_drex.per_allergen[a]]}")
                with c2:
                    st.markdown(f"**{llm_model}**")
                    if scan["llm_pending"] or scan.get("llm_model") != llm_model:
                        with st.spinner("LLM is thinking ... (Drex already answered)"):
                            scan["llm"], scan["llm_cached"] = pipeline.decide_llm(
                                scan["state"], use_cache, model=llm_model)
                        scan["llm_pending"], scan["llm_model"] = False, llm_model
                    r = scan["llm"]
                    v_llm = verdict(r, severities, p_yes)
                    if r.error:
                        st.error(f"LLM error: {r.error[:200]} (counts as CHECK, never safe)")
                    st.markdown(f"⏱ **{r.latency_ms:,.0f} ms** · 💲{(r.cost_usd or 0):.6f}")
                    for a in severities:
                        differ = v_llm.per_allergen[a] != v_drex.per_allergen[a]
                        st.write(f"{NICE[a]}: {STATUS_TXT[v_llm.per_allergen[a]]}"
                                 + ("  🔶 **disagrees**" if differ else ""))
                if scan["llm"] and scan["llm"].latency_ms and scan["drex"].latency_ms:
                    st.success(f"Drex answered **{scan['llm'].latency_ms / scan['drex'].latency_ms:.0f}x "
                               f"faster** than the LLM on this packet.")

                with st.expander("Why? (evidence)", expanded=True):
                    for a in severities:
                        st.markdown(f"**{NICE[a]}**: {v_drex.reasons[a]}")
                        for line in scan["state"]["evidence"].get(a, []):
                            st.markdown(f"- {line}")
                    st.caption("NDI read: " + read["ingredient_text"][:400])

                st.subheader("🔍 Ask Intern to investigate")
                st.caption("Intern is Nace's agent: Qwen on DeepInfra does the work, and Drex supervises "
                           "every step. Its answer explains the result. It never changes the verdict.")
                flagged = [a for a in severities if v_drex.per_allergen[a] != "safe"]
                if not flagged:
                    st.write("Nothing to investigate: every allergen was cleared.")
                for a in flagged:
                    if st.button(f"Why is {NICE[a].lower()} flagged?", key=f"intern_{a}"):
                        with st.spinner(f"Intern is investigating {NICE[a].lower()} ..."):
                            st.session_state["intern"][a] = intern_client.investigate(
                                {"product": "Scanned packet", "ingredients": scan["state"]["ingredients"],
                                 "contains_statement": read.get("contains_statement"),
                                 "may_contain_statement": read.get("may_contain_statement"),
                                 "evidence": scan["state"]["evidence"]},
                                a, severities[a],
                                verdict={a: v_drex.per_allergen[a]},
                                answers={"drex": {x.allergen: x.label for x in scan["drex"].answers},
                                         "llm": {x.allergen: x.label for x in (scan["llm"].answers
                                                                               if scan["llm"] else [])}})
                    res = st.session_state.get("intern", {}).get(a)
                    if res:
                        if res["ok"]:
                            st.info(f"**Intern on {NICE[a].lower()}:** {res['answer']}")
                            st.caption(f"{res['seconds']:.0f} s · {res['turns']} turns · "
                                       f"{res['tool_calls']} tool calls · "
                                       f"**{res['system1_checks']} Drex supervision checks**")
                        else:
                            st.warning(f"Intern could not finish: {res['error']}")
        st.divider()
        st.caption("⚠️ " + DISCLAIMER)

# --- Benchmark tab ------------------------------------------------------------------
with tab_bench:
    res_path = ROOT / "bench" / "results.json"
    if not res_path.exists():
        st.info("No benchmark results yet. Run: .venv/bin/python bench/run_bench.py")
    else:
        R = json.loads(res_path.read_text())
        M = R["metrics"]
        d = M[config.DREX_MODEL]
        llm_names = [m for m in M if m != config.DREX_MODEL]
        llm_name = st.radio("Compare Drex with", llm_names, horizontal=True,
                            index=len(llm_names) - 1)
        q = M[llm_name]
        st.subheader(f"Drex Max v1.1 vs {llm_name} on {R['n_products']} real products")
        k = st.columns(4)
        k[0].metric("Faster (median)", f"{q['median_latency_ms'] / d['median_latency_ms']:.0f}x",
                    f"{d['median_latency_ms']} ms vs {q['median_latency_ms']:,} ms", delta_color="off")
        k[1].metric("Cheaper", f"{q['cost_per_1000_usd'] / d['cost_per_1000_usd']:.0f}x",
                    f"${d['cost_per_1000_usd']} vs ${q['cost_per_1000_usd']} per 1,000 scans",
                    delta_color="off")
        k[2].metric("Dangerous mistakes (allergen on label)",
                    f"{d['wrong_safe_visible_count']} vs {q['wrong_safe_visible_count']}",
                    f"of {d['visible_positive_pairs']} allergen-present pairs", delta_color="off")
        k[3].metric("Safe foods cleared", f"{d['safe_food_cleared']}%",
                    f"{d['safe_food_cleared'] - q['safe_food_cleared']:+.1f} pts vs LLM")
        md = (ROOT / "bench" / "results.md").read_text()
        st.markdown(md.split("\n", 1)[1])
        st.caption("Why an LLM can look 'safer' on the raw wrong-SAFE rate: a model that almost never "
                   "says SAFE cannot be wrong about it, but it is not useful either. Compare "
                   "'Safe foods correctly cleared' and the wrong-SAFE rate on allergens named on the label.")
        charts = sorted((ROOT / "charts").glob("*.png"))
        if charts:
            cols = st.columns(2)
            for i, c in enumerate(charts):
                cols[i % 2].image(str(c), width="stretch")
        with st.expander("How the benchmark works"):
            st.markdown(
                f"- **Data:** {R['n_products']} US products from Open Food Facts (official dump on "
                "Hugging Face), balanced across hidden-name, may-contain, other-allergen and allergen-free "
                "products. Products whose tags contradicted their own ingredient text were dropped.\n"
                "- **Ground truth:** allergen tag -> contains; trace tag -> may contain; else not found.\n"
                "- **Fair test:** both models get the identical input (ingredients, statements, dictionary "
                "evidence) and the same safety rules. Drex returns probabilities; the LLM states a "
                "confidence.\n"
                "- **Wrong-SAFE** = said safe when the allergen is present (the dangerous error).\n"
                "- Results are cached; the demo never re-runs the benchmark live.")
    st.caption("⚠️ " + DISCLAIMER)

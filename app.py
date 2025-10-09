# app.py
import os
import asyncio
from pathlib import Path
from typing import List, Dict, Optional
import shutil
import time
import importlib
import zlib
import zipfile

import streamlit as st
import requests
import uuid

import google.generativeai as genai


# =========================
# App Config
# =========================
st.set_page_config(page_title="PaperQA + Gemini", layout="wide")


# --- Cloud / Local detection ---
def _env(name, default=""):
    return os.getenv(name, default)


DEPLOY_ENV = (
    _env("DEPLOY_ENV", "") or ""
).lower()  # set to "cloud" or "local" to force
if DEPLOY_ENV in {"cloud", "local"}:
    IS_CLOUD = DEPLOY_ENV == "cloud"
else:
    # Streamlit Community Cloud runs under /home/appuser and mounts source at /mount/src
    IS_CLOUD = os.path.exists("/home/appuser") or os.path.exists("/mount/src")


# Separate roots: caches vs user PDFs
CACHE_ROOT = Path(
    os.environ.get(
        "PAPERQA_CACHE_DIR", "/tmp/paperqa_cache" if IS_CLOUD else ".paperqa_cache"
    )
)
CACHE_ROOT.mkdir(parents=True, exist_ok=True)

# Set env BEFORE importing PaperQA modules
os.environ.setdefault("PAPERQA_CACHE_DIR", str(CACHE_ROOT.resolve()))
os.environ.setdefault("PQA_HOME", os.environ["PAPERQA_CACHE_DIR"])
Path(os.environ["PAPERQA_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)

from paperqa import Settings, ask, agent_query
from paperqa.settings import AgentSettings
from paperqa.agents.search import get_directory_index
import paperqa as _pq

# Papers live OUTSIDE the cache root (so nuking caches won't delete PDFs)
PAPERS_ROOT = Path(
    os.environ.get("PAPERS_ROOT", "/tmp/papers" if IS_CLOUD else "user_papers")
)
PAPERS_ROOT.mkdir(parents=True, exist_ok=True)


if "session_id" not in st.session_state:
    st.session_state["session_id"] = uuid.uuid4().hex
SESSION_ID = st.session_state["session_id"]

# Per-session folder for this user's PDFs
SESSION_ROOT = PAPERS_ROOT / SESSION_ID
SESSION_ROOT.mkdir(parents=True, exist_ok=True)

# Default papers directory for the app
st.session_state.setdefault("papers_directory", str(SESSION_ROOT))


# =========================
# Helpers
# =========================
def run_async(coro):
    """Safely run an async coroutine inside Streamlit."""
    try:
        return asyncio.run(coro)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(coro)
        finally:
            loop.close()


# --- Secure Gemini config (per session) ---
import threading


@st.cache_resource
def _genai_lock():
    # one lock for the whole process (all sessions)
    return threading.Lock()


def ensure_gemini():
    """
    Configure google.generativeai using ONLY the current session's key,
    protected by a process-wide lock because configure() is global.
    """
    key = st.session_state.get("gemini_api_key")

    if not key:
        raise RuntimeError("Gemini API key not configured. Set it in the sidebar.")

    with _genai_lock():
        genai.configure(api_key=key)


import contextlib


@contextlib.contextmanager
def gemini_key_env():
    """
    Temporarily set env vars LiteLLM expects (GOOGLE_API_KEY/GEMINI_API_KEY)
    while also configuring google.generativeai. Protected by the process-wide lock.
    """
    key = st.session_state.get("gemini_api_key")
    if not key:
        raise RuntimeError("Gemini API key not configured. Set it in the sidebar.")

    with _genai_lock():
        prev_google = os.environ.get("GOOGLE_API_KEY")
        prev_gemini = os.environ.get("GEMINI_API_KEY")
        os.environ["GOOGLE_API_KEY"] = key
        os.environ["GEMINI_API_KEY"] = key

        genai.configure(api_key=key)
        try:
            yield
        finally:
            # restore previous values (or unset)
            if prev_google is None:
                os.environ.pop("GOOGLE_API_KEY", None)
            else:
                os.environ["GOOGLE_API_KEY"] = prev_google

            if prev_gemini is None:
                os.environ.pop("GEMINI_API_KEY", None)
            else:
                os.environ["GEMINI_API_KEY"] = prev_gemini


@st.dialog("🔑 How to get your Gemini API key (Google AI Studio)")
def show_api_key_modal():
    st.markdown(
        """
### Step 1 — Open Google AI Studio
👉 [**Click here to open AI Studio**](https://aistudio.google.com/projects)  
Sign in with your Google account if prompted.

### Step 2 — Create / Import a Project
If prompted:
1. Choose **Create new project** or **Import from Google Cloud project**.
2. Pick or name your project and continue.

### Step 3 — Generate the API Key
1. In the **bottom of the left sidebar**, click **Get API key**.  
2. Click **Create API key** in the top right corner → name the key and choose the cloud project you want the API key from. 
3. Click **Create Key**.
4. Copy the generated key.

### Step 4 — Use it here
Paste the key in the **sidebar** field  
> “Enter your Gemini API key”  
then click **🧪 Test key** to verify.

> Your key stays in memory only for this session and is never saved to disk.
"""
    )
    if st.button("Close", type="primary", use_container_width=True):
        st.session_state["show_api_help"] = False
        st.rerun()


def _test_gemini_key():
    try:
        ensure_gemini()  # uses st.session_state["gemini_api_key"]

        cfg_name = st.session_state.get("cfg_llm", "gemini/gemini-2.5-flash")
        bare_id, full_id = _normalize_model_id(cfg_name)  # step 3

        with _genai_lock():
            m = genai.GenerativeModel(bare_id)
            r = m.generate_content("ping")

        txt = (getattr(r, "text", "") or "").strip()
        st.success(f"API key works ✅ (model: {full_id})")
        if txt:
            st.caption((txt[:140] + "…") if len(txt) > 140 else txt)
    except Exception as e:
        st.error(f"Key test failed: {e}")


def _normalize_model_id(name: str, default: str = "gemini-2.5-flash"):
    """
    Returns (bare_id_for_sdk, full_id_for_settings).
    - SDK (google.generativeai) wants 'gemini-2.5-flash'
    - PaperQA settings can use 'gemini/gemini-2.5-flash'
    """
    if not name:
        return default, f"gemini/{default}"
    bare = name.split("/", 1)[1] if "/" in name else name
    full = name if "/" in name else f"gemini/{bare}"
    return bare, full


def _rm(path, removed: List[str]):
    try:
        p = Path(path)
        if p.is_file() and (p.suffix.lower() == ".pdf" or p.suffix.lower() == ".txt"):
            return
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
            removed.append(str(p))
        elif p.exists():
            p.unlink(missing_ok=True)
            removed.append(str(p))
    except Exception as e:
        removed.append(f"{path} (failed: {e})")


def safe_session_path(path_like: str) -> str:
    """Clamp any user-provided path to this session's private root."""
    p = Path(path_like).resolve()
    root = SESSION_ROOT.resolve()
    return str(root) if not str(p).startswith(str(root)) else str(p)


def nuke_index_artifacts(papers_dir: str) -> List[str]:
    """
    Remove PaperQA/vector DB artifacts next to PDFs and in project/cache dirs,
    but NEVER delete PDFs or the whole cache root blindly.
    """
    removed: List[str] = []
    pdir = Path(papers_dir)
    proj = Path.cwd()

    def _safe_rm_path(path: Path):
        # Never delete PDF files
        if path.is_file() and (
            path.suffix.lower() == ".pdf" or path.suffix.lower() == ".txt"
        ):
            return
        _rm(path, removed)

    # Next to PDFs and in project root (common index dirs/files)
    dir_candidates = [
        ".paperqa_index",
        ".paperqa",
        "paperqa_index",
        "index",
        "lancedb",
        "chroma",
    ]
    file_candidates = ["index.sqlite", "faiss.index", "chroma.sqlite3"]

    for base in (pdir, proj):
        for d in dir_candidates:
            _safe_rm_path(base / d)
        for f in file_candidates:
            _safe_rm_path(base / f)

        # Wipe lance/chroma/faiss artifacts found by globbing (non-recursive & recursive)
        patterns = ["*.lance", "*.lancedb", "chroma.sqlite*", "faiss.index"]
        for pat in patterns:
            for hit in base.glob(pat):
                _safe_rm_path(hit)
            for hit in base.rglob(pat):
                _safe_rm_path(hit)

    # 2) Cache root: remove ONLY known cache artifacts, not the whole directory
    cache_root = Path(
        os.environ.get("PAPERQA_CACHE_DIR", Path.home() / ".cache" / "paperqa")
    )
    if cache_root.exists():
        # Known cache dirs inside the cache root
        for d in ["index", "lancedb", "chroma", ".paperqa_index", ".paperqa"]:
            _safe_rm_path(cache_root / d)
        # Known cache files/patterns inside the cache root
        for pat in [
            "*.lance",
            "*.lancedb",
            "chroma.sqlite*",
            "faiss.index",
            "index.sqlite",
        ]:
            for hit in cache_root.glob(pat):
                _safe_rm_path(hit)
            for hit in cache_root.rglob(pat):
                _safe_rm_path(hit)

    # Extra common homes (never PDF files)
    for home_candidate in [
        Path.home() / ".paperqa",
        Path.home() / ".local" / "share" / "paperqa",
        Path.home() / ".cache" / "paperqa",
        Path.home() / ".pqa",
    ]:
        _safe_rm_path(home_candidate)

    # Clear Streamlit + Python import caches
    try:
        st.cache_data.clear()
        st.cache_resource.clear()
        removed.append("Streamlit cache cleared (data + resource)")
        _ = _genai_lock()  # <-- recreate process-wide lock right away
    except Exception as e:
        removed.append(f"Streamlit cache clear failed: {e}")

    try:
        importlib.invalidate_caches()
        removed.append("Python import caches invalidated")
    except Exception as e:
        removed.append(f"Import cache invalidate failed: {e}")

    # Re-create core cache directories for next rebuild
    cache_root.mkdir(parents=True, exist_ok=True)

    return removed


def list_pdf_stats(papers_dir: str) -> str:
    pdfs = list(Path(papers_dir).glob("*.pdf"))
    txts = list(Path(papers_dir).glob("*.txt"))
    total_bytes = sum(p.stat().st_size for p in pdfs + txts)
    lines = [
        f"PDFs: {len(pdfs)} | Texts: {len(txts)} | Total size: {total_bytes/1e6:.2f} MB"
    ]
    for p in pdfs[:20]:
        lines.append(
            f"- {p.name} | {p.stat().st_size/1e6:.2f} MB | mtime={time.ctime(p.stat().st_mtime)}"
        )
    return "\n".join(lines) if (pdfs or txts) else "No PDFs or text files found."


# =========================
# Sidebar: API & Dataset
# =========================
with st.sidebar:
    st.header("🔐 Gemini API Key")
    st.button(
        "❓ How to get a key",
        use_container_width=True,
        on_click=lambda: st.session_state.update({"show_api_help": True}),
    )

    api_key = st.text_input(
        "Enter your Gemini API key",
        value=st.session_state.get("gemini_api_key", ""),
        type="password",
        placeholder="Paste API key here",
        autocomplete="off",
        help="Stored only in this session's memory (not environment, not disk).",
    )
    if api_key:
        st.session_state["gemini_api_key"] = api_key
        st.success("Gemini API key set for this session.")
    st.button(
        "🧪 Test key",
        use_container_width=True,
        on_click=_test_gemini_key,
        disabled=not bool(st.session_state.get("gemini_api_key")),
    )

# -- model config
with st.sidebar:
    st.header("⚙️ Model & Index Settings")

    # --- Model choices ---
    LLM_CHOICES = [
        "gemini/gemini-2.5-flash",
        "gemini/gemini-2.5-pro",
    ]
    EMBED_CHOICES = [
        "gemini/text-embedding-004",
    ]

    # --- Select models ---
    st.session_state.setdefault("cfg_llm", LLM_CHOICES[0])
    st.selectbox("LLM model", LLM_CHOICES, key="cfg_llm")

    # no choices for embedding model
    st.session_state.setdefault("cfg_embed", "gemini/text-embedding-004")

    # --- Generation / agent tuning ---
    temperature = st.slider("temperature", 0.0, 1.0, 0.1, 0.05, key="cfg_temp")
    use_agent_mode = st.checkbox(
        "Use Agent Mode (agent_query)", True, key="cfg_use_agent"
    )
    search_count = st.number_input("search_count", 0, 20, 6, 1, key="cfg_search_count")
    timeout_s = st.number_input("timeout (sec)", 10, 900, 300, 10, key="cfg_timeout")

    # --- Answer settings ---
    evidence_k = st.number_input("evidence_k", 1, 50, 8, 1, key="cfg_evidence_k")
    answer_max_sources = st.number_input(
        "answer_max_sources", 1, 20, 4, 1, key="cfg_ans_max_src"
    )
    evidence_summary_length = st.text_input(
        "evidence_summary_length", "about 80 words", key="cfg_ev_sum_len"
    )
    answer_length = st.text_input(
        "answer_length", "about 150 words, but can be longer", key="cfg_ans_len"
    )
    max_concurrent_requests = st.number_input(
        "max_concurrent_requests", 1, 16, 2, 1, key="cfg_max_conc"
    )

    # --- Parsing / chunking ---
    chunk_size = st.number_input(
        "chunk_size", 512, 8192, 4000, 64, key="cfg_chunk_size"
    )
    overlap = st.number_input("overlap", 0, 2048, 200, 10, key="cfg_overlap")

    verbosity = st.select_slider(
        "verbosity", options=[0, 1, 2], value=1, key="cfg_verbosity"
    )

curr_embed = st.session_state.get("cfg_embed", "gemini/text-embedding-004")
prev_embed = st.session_state.get("_prev_embed", None)
if prev_embed is None:
    st.session_state["_prev_embed"] = curr_embed
elif prev_embed != curr_embed:
    st.session_state["_prev_embed"] = curr_embed
    st.sidebar.warning("Embedding model changed — please rebuild the index.")


def download_sample_papers() -> str:
    papers = {
        "attention_is_all_you_need.pdf": "https://arxiv.org/pdf/1706.03762.pdf",
        "bert_paper.pdf": "https://arxiv.org/pdf/1810.04805.pdf",
        "gpt3_paper.pdf": "https://arxiv.org/pdf/2005.14165.pdf",
    }
    papers_dir = Path(
        safe_session_path(st.session_state.get("papers_directory", str(SESSION_ROOT)))
    )

    papers_dir.mkdir(parents=True, exist_ok=True)

    log = []
    for filename, url in papers.items():
        filepath = papers_dir / filename
        if not filepath.exists():
            try:
                headers = {"User-Agent": "paperqa-notebook/1.0"}
                with requests.get(url, stream=True, timeout=30, headers=headers) as r:
                    r.raise_for_status()
                    with open(filepath, "wb") as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                log.append(f"✅ Downloaded: {filename}")
            except Exception as e:
                log.append(f"❌ Failed to download {filename}: {e}")
        else:
            log.append(f"📄 Already exists: {filename}")

    st.code("\n".join(log), language="text")
    return str(papers_dir.resolve())


# --- Sidebar: downloads, uploads, directory selector ---
with st.sidebar:
    # Download sample papers
    if st.button("📥 Download Sample Papers", use_container_width=True):
        with st.spinner("Downloading PDFs..."):
            papers_directory = download_sample_papers()
        st.success("Done!")
        st.session_state["papers_directory"] = papers_directory

    # Upload your own PDFs
    st.header("📤 Upload Papers")
    uploaded_files = st.file_uploader(
        "Upload one or more PDFs",
        type=["pdf"],
        accept_multiple_files=True,
    )
    if uploaded_files:
        # Save into the current papers directory (default: sample_papers)
        papers_dir = Path(
            safe_session_path(
                st.session_state.get("papers_directory", str(SESSION_ROOT))
            )
        )
        papers_dir.mkdir(exist_ok=True, parents=True)
        saved = []
        for file in uploaded_files:
            dest = papers_dir / file.name
            with open(dest, "wb") as f:
                f.write(file.getbuffer())
            saved.append(file.name)
        st.success(f"Uploaded {len(saved)} file(s): {', '.join(saved)}")
        st.session_state["uploaded_last"] = saved

    # Directory selector
    if IS_CLOUD:
        papers_directory = str(SESSION_ROOT)

    else:
        proposed = st.text_input(
            "Papers Directory",
            value=st.session_state.get("papers_directory", str(SESSION_ROOT)),
            help="Must stay under the session folder.",
        )
        papers_directory = safe_session_path(proposed)
        st.session_state["papers_directory"] = papers_directory

    st.divider()
    st.caption(f"paper-qa version: **{_pq.__version__}**")
    if st.session_state.get("show_api_help"):
        show_api_key_modal()


# =========================
# PaperQA Settings
# =========================
def create_gemini_settings(
    paper_dir: str,
    temperature: float = 0.1,
    salt: Optional[str] = None,
    force_parse_variant: bool = False,
) -> Settings:
    # Slightly perturb a harmless field so cache keys differ when we want
    answer_len = "about 150 words, but can be longer"
    if salt:
        answer_len += f" [salt:{salt}]"

    # If we force_parse_variant, slightly change parsing params (harmless but unique)
    parsing = dict(chunk_size=4000, overlap=200)
    if force_parse_variant:
        parsing = dict(chunk_size=3997, overlap=203)  # tiny perturbation to bust cache

    # Pull from sidebar (with same defaults as notebook)
    llm = st.session_state.get("cfg_llm", "gemini/gemini-2.5-flash")
    embed = st.session_state.get("cfg_embed", "gemini/text-embedding-004")
    temp = st.session_state.get("cfg_temp", 0.1)

    sc = int(st.session_state.get("cfg_search_count", 6))
    to_sec = float(st.session_state.get("cfg_timeout", 300.0))

    k = int(st.session_state.get("cfg_evidence_k", 8))
    max_src = int(st.session_state.get("cfg_ans_max_src", 4))
    ev_sum_len = st.session_state.get("cfg_ev_sum_len", "about 80 words")
    max_conc = int(st.session_state.get("cfg_max_conc", 2))

    csz = int(st.session_state.get("cfg_chunk_size", 4000))
    ovl = int(st.session_state.get("cfg_overlap", 200))
    if force_parse_variant:
        csz = max(256, csz - 3)
        ovl = ovl + 3

    return Settings(
        llm=llm,
        summary_llm=llm,
        agent=AgentSettings(agent_llm=llm, search_count=sc, timeout=to_sec),
        embedding=embed,
        temperature=temp,
        paper_directory=paper_dir,
        answer=dict(
            evidence_k=k,
            answer_max_sources=max_src,
            evidence_summary_length=ev_sum_len,
            answer_length=answer_len,
            max_concurrent_requests=max_conc,
        ),
        parsing=dict(chunk_size=csz, overlap=ovl),
        verbosity=int(st.session_state.get("cfg_verbosity", 1)),
    )


# =========================
# Index Rebuild
# =========================
async def rebuild_index_async(
    papers_dir: str, salt: Optional[str] = None, force_parse_variant: bool = False
):
    try:
        with gemini_key_env():
            return await get_directory_index(
                settings=create_gemini_settings(
                    papers_dir, salt=salt, force_parse_variant=force_parse_variant
                ),
                build=True,
            )
    except (zlib.error, zipfile.BadZipFile, EOFError) as e:
        st.warning(
            f"Index cache looked corrupt ({e}). Clearing caches and retrying once..."
        )
        nuke_index_artifacts(papers_dir)
        os.environ.setdefault("PQA_HOME", os.environ["PAPERQA_CACHE_DIR"])
        with gemini_key_env():
            return await get_directory_index(
                settings=create_gemini_settings(
                    papers_dir, salt=str(time.time_ns()), force_parse_variant=True
                ),
                build=True,
            )


# =========================
# Toolbar: single "Force Rebuild" (now nukes + rebuilds)
# =========================
col_a, col_b = st.columns([1, 2], vertical_alignment="top")

with col_a:
    if st.button(
        "📦 Force Rebuild Index",
        help="Fully clears caches and rebuilds the index from scratch",
    ):
        with st.expander("📑 Current PDF stats"):
            st.code(
                list_pdf_stats(safe_session_path(papers_directory)), language="text"
            )

        # Extra safety: reload the search module to kill in-process singletons
        try:
            import paperqa.agents.search as pqs

            importlib.reload(pqs)
        except Exception:
            pass

        with st.spinner("Deleting index/cache artifacts..."):
            removed = nuke_index_artifacts(safe_session_path(papers_directory))
            st.code("Removed:\n" + "\n".join(removed), language="text")

        with st.spinner("Rebuilding index from scratch..."):
            t0 = time.perf_counter()
            try:
                # Reassert PQA_HOME before rebuild
                os.environ.setdefault("PQA_HOME", os.environ["PAPERQA_CACHE_DIR"])
                salt = str(time.time_ns())
                _ = run_async(
                    rebuild_index_async(
                        safe_session_path(papers_directory),
                        salt=salt,
                        force_parse_variant=True,
                    )
                )
                st.success(f"Fresh index built in {time.perf_counter() - t0:.2f}s.")
            except Exception as e:
                st.error(f"Fresh rebuild failed: {e}")

with col_b:
    try:
        list_dir = Path(safe_session_path(papers_directory))
        files = [p.name for p in list_dir.glob("*.pdf")] + [
            p.name for p in list_dir.glob("*.txt")
        ]
        if files:
            st.write("**Detected PDFs and Texts:**", ", ".join(files))
        else:
            st.info("No PDFs or text files found in the selected directory.")
    except Exception as e:
        st.error(f"Could not list directory: {e}")

if not IS_CLOUD:
    st.caption(f"📂 Using papers directory: `{papers_directory}`")


# =========================
# Agent Class
# =========================
class PaperQAAgent:
    """AI Agent for literature analysis using PaperQA2 (Streamlit)."""

    def __init__(self, papers_directory: str, temperature: float = 0.1):
        self.settings = create_gemini_settings(papers_directory, temperature)
        self.papers_dir = papers_directory

    async def ask_question(self, question: str, use_agent: bool = True):
        try:
            with gemini_key_env():
                if use_agent:
                    response = await agent_query(query=question, settings=self.settings)
                else:
                    response = ask(question, settings=self.settings)
            return response
        except Exception as e:
            st.error(f"Error processing question: {e}")
            return None

    def display_answer(self, response):
        if response is None:
            st.error("No response received.")
        else:
            answer_text = getattr(response, "answer", str(response))
            st.subheader("📋 Answer")
            st.write(answer_text)

            contexts = getattr(response, "contexts", getattr(response, "context", []))
            if contexts:
                st.divider()
                st.subheader("📚 Sources Used")
                for i, context in enumerate(contexts[:3], 1):
                    context_name = getattr(
                        context, "name", getattr(context, "doc", f"Source {i}")
                    )
                    context_text = getattr(
                        context, "text", getattr(context, "content", str(context))
                    )
                    with st.expander(f"{i}. {context_name}"):
                        st.write(
                            context_text[:1000]
                            + ("..." if len(context_text) > 1000 else "")
                        )


# =========================
# Summarization Helpers (Gemini)
# =========================
def _extract_answer_and_sources(response):
    """Return (answer_text, sources_list) from a PaperQA response object."""
    if response is None:
        return "", []
    answer_text = getattr(response, "answer", str(response))
    ctxs = getattr(response, "contexts", getattr(response, "context", [])) or []
    sources = []
    for i, ctx in enumerate(ctxs, 1):
        name = getattr(ctx, "name", None) or getattr(ctx, "doc", None) or f"Source {i}"
        page = getattr(ctx, "page", None) or getattr(ctx, "page_number", None)
        if page is not None:
            sources.append(f"{name} (p. {page})")
        else:
            sources.append(f"{name}")
    return answer_text, sources


def summarize_with_gemini(
    answer_text: str,
    sources: List[str],
    model_name: str = "gemini-2.5-flash",
    temperature: float = 0.1,
) -> Dict[str, str]:
    """Return {'detailed': md_text, 'short': md_text} using Gemini, with model id normalized."""
    # Build prompt
    sources_block = (
        "\n".join([f"[{i+1}] {s}" for i, s in enumerate(sources)])
        if sources
        else "None"
    )
    prompt = f"""
You will simplify a technical answer into two user-friendly summaries.
The original answer is between <answer>. Known sources are listed in <sources>.

<answer>
{answer_text}
</answer>

<sources>
{sources_block}
</sources>

TASKS:
1) DETAILED SUMMARY WITH AUTHOR-YEAR CITATIONS:
   - Use inline citations like (Author, Year) or (Author et al., Year).
   - Only cite items that appear in the sources list and avoid fabricating details.
   - 200-400 words.
   - Output as Markdown with the heading: "### Detailed Summary (author-year)".

2) SHORT CONCISE VERSION:
   - 3-5 bullet points.
   - No citations.
   - Heading: "### Short Summary".
""".strip()

    # --- choose model & normalize ---
    cfg_name = st.session_state.get("cfg_llm") or model_name
    bare_id, _full_id = _normalize_model_id(cfg_name)  # <- step 3

    # --- call Gemini ---
    with gemini_key_env():
        model = genai.GenerativeModel(
            bare_id,
            generation_config={"temperature": float(temperature)},
        )
        resp = model.generate_content(prompt)

    text = getattr(resp, "text", "") or ""

    # Parse into two sections
    detailed_md, short_md = "", ""
    lower = text.lower()
    if "detailed summary" in lower and "short summary" in lower:
        if "\n### Short Summary" in text:
            parts = text.split("\n### Short Summary", 1)
            detailed_md = parts[0].strip()
            short_md = "### Short Summary" + (parts[1] if len(parts) > 1 else "")
        else:
            split_idx = lower.find("### short summary")
            detailed_md = text[:split_idx].strip()
            short_md = text[split_idx:].strip()
    else:
        detailed_md = text.strip()
        lines = [
            l.strip("-• ").strip()
            for l in detailed_md.splitlines()
            if l.strip() and not l.startswith("#")
        ]
        short_md = "### Short Summary\n\n" + "\n".join([f"- {l}" for l in lines[:4]])

    return {"detailed": detailed_md, "short": short_md}


# =========================
# UI Tabs
# =========================
st.markdown("## 🤖 PaperQA2 + Gemini — Streamlit App")
st.caption(
    "Upload PDFs → rebuild index → ask a question about the papers→ summarize the answer (detailed + short)."
)

tab_custom, tab_summaries, tab_tips = st.tabs(
    ["Custom Query", "Summaries", "Usage Tips"]
)

# --- Custom Query
with tab_custom:
    st.markdown("### 🎯 Interactive Custom Query")
    custom_q = st.text_input(
        "Enter your question",
        value="How do transformers handle long sequences?",
        key="custom_q",
    )
    show_sources = st.checkbox("Show sources", value=True, key="custom_sources")

    # --- Always show Ask button before the answer
    if st.button("Ask", key="custom_btn"):
        agent = PaperQAAgent(safe_session_path(papers_directory))
        with st.spinner("Querying..."):
            resp = run_async(
                agent.ask_question(
                    custom_q, use_agent=st.session_state.get("cfg_use_agent", True)
                )
            )

        if resp:
            ans = getattr(resp, "answer", str(resp))
            # build a simple sources list for persistence
            ctxs = getattr(resp, "contexts", getattr(resp, "context", [])) or []
            srcs = []
            for i, ctx in enumerate(ctxs[:10], 1):
                name = (
                    getattr(ctx, "name", None)
                    or getattr(ctx, "doc", None)
                    or f"Source {i}"
                )
                page = getattr(ctx, "page", None) or getattr(ctx, "page_number", None)
                srcs.append(f"{name} (p. {page})" if page is not None else name)

            # persist everything you need to re-render later
            st.session_state["custom_cache"] = {
                "question": custom_q,
                "answer": ans,
                "sources": srcs,
                "response": resp,  # keep full object for summaries
            }

            # also keep compatibility with your summaries tab
            st.session_state.setdefault("last_results", {})
            st.session_state["last_results"][custom_q] = resp
        else:
            st.error("Sorry, I couldn't find an answer to that question.")

    # --- Re-render last answer if present (survives tab switches & reruns)
    cache = st.session_state.get("custom_cache")
    if cache:
        st.markdown(f"**Last question:** {cache['question']}")
        st.subheader("🤖 Answer")
        st.write(cache["answer"])
        if show_sources and cache.get("sources"):
            with st.expander("📚 Sources used"):
                for i, s in enumerate(cache["sources"], 1):
                    st.write(f"{i}. {s}")

# --- Summaries
with tab_summaries:
    st.markdown("### ✨ Summarize Any Answer (Gemini)")
    st.caption("Two versions: detailed (with citations) and short concise.")

    mode = st.radio(
        "Choose source to summarize",
        ["Use a previous result", "Paste text"],
        horizontal=True,
        key="sum_mode",
    )

    text_to_summarize = ""
    sources_for_summary: List[str] = []
    selected_question = None

    if mode == "Use a previous result":
        results = st.session_state.get("last_results", {})
        cc = st.session_state.get("custom_cache")
        if cc and cc.get("response") and cc["question"] not in results:
            results = {**results, cc["question"]: cc["response"]}
        if not results:
            st.info(
                "No stored results yet. Ask a question first, or switch to 'Paste text'."
            )
        else:
            options = list(results.keys())
            selected_question = st.selectbox("Pick a question/result", options)
            resp = results.get(selected_question)
            if resp:
                answer_text, sources = _extract_answer_and_sources(resp)
                st.text_area(
                    "Preview of answer to be summarized", value=answer_text, height=160
                )
                text_to_summarize = answer_text
                sources_for_summary = sources
            else:
                st.warning("No response object found for this selection.")
    else:
        text_to_summarize = st.text_area(
            "Paste any text to summarize",
            height=180,
            placeholder="Paste long answer here...",
        )
        sources_for_summary = []

    sum_temp = st.slider(
        "Summary Style: 0 = precise, 1 = creative",
        0.0,
        1.0,
        0.1,
        0.05,
        key="sum_temp",
    )
    model_name = "gemini-2.5-flash"

    if st.button("Generate Summaries ✍️", key="sum_btn", use_container_width=True):
        if not text_to_summarize:
            st.error("Please provide text to summarize.")
        else:
            t0 = time.perf_counter()
            try:
                with st.spinner("Summarizing with Gemini..."):
                    summaries = summarize_with_gemini(
                        text_to_summarize,
                        sources_for_summary,
                        model_name=model_name,
                        temperature=sum_temp,
                    )

                st.markdown(summaries["detailed"])
                st.divider()
                st.markdown(summaries["short"])

                det_bytes = summaries["detailed"].encode("utf-8")
                short_bytes = summaries["short"].encode("utf-8")
                st.download_button(
                    "Download Detailed (Markdown)",
                    det_bytes,
                    file_name="detailed_summary.md",
                    mime="text/markdown",
                )
                st.download_button(
                    "Download Short (Markdown)",
                    short_bytes,
                    file_name="short_summary.md",
                    mime="text/markdown",
                )

                st.session_state["last_summaries"] = summaries
                st.success(f"Done in {time.perf_counter() - t0:.2f}s.")
            except Exception as e:
                st.error(f"Summarization failed: {e}")

# --- Usage Tips
with tab_tips:
    st.markdown("### 🛠️ Usage Tips")
    st.markdown(
        """
**Question Formulation**
- Be specific about what you want to know
- Ask about comparisons, mechanisms, or implications
- Use domain-specific terminology

**Model Configuration**
- Gemini 2.5 Flash is fast & reliable
- Adjust temperature (0.0-1.0) for creativity vs precision
- Use smaller `chunk_size` for finer splits

**Document Management**
- Put PDFs in your papers directory
- Use meaningful filenames
- Mix survey & method papers for broader coverage

**Performance**
- Limit concurrency on free tiers
- Lower `evidence_k` for faster answers
- Cache by keeping the app session alive
        """
    )

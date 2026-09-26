import streamlit as st
import pandas as pd
from huggingface_hub import HfApi
import requests
from requests.adapters import HTTPAdapter
import re
import concurrent.futures
import logging
from datetime import datetime, timezone
from types import SimpleNamespace
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s")

# Configure the Streamlit page
st.set_page_config(page_title="Heretic Models Explorer", page_icon="🔥", layout="wide")

st.title("🔥 Heretic Models Explorer")
st.markdown(
    "This space lists all models on Hugging Face tagged with "
    "[`heretic`](https://huggingface.co/models?other=heretic). "
    "It automatically fetches their model cards to extract **KL Divergence** and **Refusals**, "
    "allowing you to sort and compare them easily. Click on any column header to sort!"
)

# ---------------------------------------------------------------------------
# Filter config (single source of truth)
# ---------------------------------------------------------------------------
FILTER_DEFAULTS = {
    "query": "",
    "date_range": "All time",
    "unknowns": "Include unknowns",
    "kl_range": (0.0, 0.5),
    "refusal_range": (0.0, 100.0),
    "min_likes": 0,
    "min_downloads": 0,
    "show_quants": False,
    "sort_by": "KL Divergence (low → high)",
}

DATE_OPTIONS = {
    "All time": None,
    "Last week": 7,
    "Last month": 30,
    "Last 6 months": 182,
    "Last year": 365,
}

UNKNOWN_OPTIONS = (
    "Include unknowns",
    "Hide unknowns",
    "Only unknowns",
)

SORT_OPTIONS = {
    "KL Divergence (low → high)": ("KL Divergence", True),
    "KL Divergence (high → low)": ("KL Divergence", False),
    "Refusal Rate (%) (low → high)": ("Refusal Rate (%)", True),
    "Refusal Rate (%) (high → low)": ("Refusal Rate (%)", False),
    "Likes (high → low)": ("Likes", False),
    "Downloads (high → low)": ("Downloads", False),
    "Model (A → Z)": ("Model", True),
}

QUANT_PATTERNS = re.compile(r'(?i)(gguf|mlx|awq|nvfp4|gptq|exl[23]|-quant|int8|int4|oq[1-8]|mxfp[48])')
QUANT_RE = r"(?i)(gguf|mlx|awq|nvfp4|gptq|exl[23]|-quant|int8|int4|oq[1-8]|mxfp[48])"

# ---------------------------------------------------------------------------
# Network: reused Session + retry
# ---------------------------------------------------------------------------
_SESSION = None


def _get_session():
    global _SESSION
    if _SESSION is None:
        s = requests.Session()
        adapter = HTTPAdapter(pool_connections=20, pool_maxsize=20, max_retries=0)
        s.mount("https://", adapter)
        s.mount("http://", adapter)
        s.headers.update({"User-Agent": "Heretic-Models-Explorer/1.0"})
        _SESSION = s
    return _SESSION


@retry(
    retry=retry_if_exception_type((requests.exceptions.ConnectionError, requests.exceptions.Timeout, requests.exceptions.HTTPError)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10)
)
def fetch_url(url):
    response = _get_session().get(url, timeout=(3.05, 5))
    # Raise HTTPError for rate limits (429) or server errors (5xx) to trigger a retry
    if response.status_code in (429, 500, 502, 503, 504):
        response.raise_for_status()
    return response

# Fetch model's readme.md file and attempt to parse refusal and KL divergence data.
def fetch_model_info(model):
    """Fetches the README.md for a given model and dynamically extracts metrics."""
    model_id = model.id
    kl_div = None
    refusals_str = None
    refusal_rate = None
    initial_refusals_str = None

    # Download the README.md via raw URL for high-speed fetching
    url = f"https://huggingface.co/{model_id}/raw/main/README.md"
    try:
        response = fetch_url(url)
        if response.status_code == 200:
            readme_text = response.text

            # Extract KL divergence using Regex
            # Matches formats like: "KL divergence | 0.0033"
            kl_match = re.search(r"(?i)KL\s*divergence[^a-zA-Z\d\n]*?([\d\.]+)", readme_text)
            if kl_match:
                try:
                    kl_div = float(kl_match.group(1))
                except ValueError as e:
                    logging.warning(f"Could not parse KL divergence for {model_id}: {e}")

            # Extract Refusals using Regex
            # Matches formats like: "Refusals | 15/100" or "15 / 100"
            for line in readme_text.split('\n'):
                line_lower = line.lower()
                if 'refusals' in line_lower:
                    fractions = re.findall(r"(\d+)\s*/\s*(\d+)", line)
                    if not fractions:
                        continue

                    if 'initial' in line_lower and initial_refusals_str is None:
                        initial_refusals_str = f"{fractions[0][0]}/{fractions[0][1]}"
                    elif 'initial' not in line_lower and refusals_str is None:
                        refusals_str = f"{fractions[0][0]}/{fractions[0][1]}"
                        try:
                            refusal_rate = (int(fractions[0][0]) / int(fractions[0][1])) * 100
                        except (ValueError, ZeroDivisionError) as e:
                            logging.warning(f"Could not parse refusals for {model_id}: {e}")
                            continue
                        if len(fractions) >= 2 and initial_refusals_str is None:
                            initial_refusals_str = f"{fractions[1][0]}/{fractions[1][1]}"

    except requests.exceptions.RequestException as e:
        logging.warning(f"Error fetching README for {model_id}: {e}")
    except Exception as e:
        logging.error(f"Unexpected error parsing README for {model_id}: {e}")

    # Skip models without explicitly stated KL divergence or refusal data
    if kl_div is None or refusals_str is None:
        return None

    return {
        "Model ID": model_id,
        "KL Divergence": kl_div,
        "Initial Refusals": initial_refusals_str,
        "Refusal Rate (%)": refusal_rate,
        "Refusals": refusals_str,
        "Likes": getattr(model, 'likes', 0),
        "Downloads": getattr(model, 'downloads', 0),
        "URL": f"https://huggingface.co/{model_id}"
    }

# Check for reproducibility record first and then fallback to parsing data from readme.md
def get_model_data(model):
    model_id = model.id
    url = f"https://huggingface.co/{model_id}/raw/main/reproduce/reproduce.json"
    try:
        response = fetch_url(url)
        if response.status_code == 200:
            data = response.json()
            metrics = data.get("metrics")
            if isinstance(metrics, dict):
                kl_div = metrics.get("kl_divergence") or metrics.get("kl")
                refusals = metrics.get("refusals")
                base_refusals = metrics.get("base_refusals")
                n_bad = metrics.get("n_bad_prompts")
                if kl_div is not None and refusals is not None and n_bad is not None:
                    initial_refusals_str = f"{base_refusals}/{n_bad}" if base_refusals is not None else None
                    refusals_str = f"{refusals}/{n_bad}"
                    refusal_rate = (refusals / n_bad) * 100 if n_bad > 0 else 0
                    return {
                        "Model ID": model_id,
                        "KL Divergence": float(kl_div),
                        "Initial Refusals": initial_refusals_str,
                        "Refusal Rate (%)": refusal_rate,
                        "Refusals": refusals_str,
                        "Likes": getattr(model, 'likes', 0),
                        "Downloads": getattr(model, 'downloads', 0),
                        "URL": f"https://huggingface.co/{model_id}"
                    }
    except requests.exceptions.RequestException as e:
        logging.warning(f"Error fetching JSON for {model_id}: {e}")
    except (ValueError, KeyError, TypeError) as e:
        logging.warning(f"Formatting error in JSON for {model_id}: {e}")
    except Exception as e:
        logging.error(f"Unexpected error processing JSON for {model_id}: {e}")
    return fetch_model_info(model)

# ---------------------------------------------------------------------------
# Two-stage cached loading (faster initial load)
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def list_heretic_metadata():
    """Stage 1: HfApi only, no per-model fetch. Cheap and fast.

    Returns ALL models (including quants) with Created/LastModified dates
    so the caller can pre-filter by date/query BEFORE the expensive
    per-model README/reproduce.json parsing.
    """
    api = HfApi()
    # Query all models using the Hugging Face Hub `filter` parameter
    models = list(api.list_models(filter="heretic"))
    out = []
    for m in models:
        mid = m.id
        created = getattr(m, "created_at", None)
        modified = getattr(m, "last_modified", None)
        # Normalise to ISO strings (cache-safe, JSON-safe)
        try:
            created_iso = created.isoformat() if hasattr(created, "isoformat") else (str(created) if created else None)
        except Exception:
            created_iso = None
        try:
            modified_iso = modified.isoformat() if hasattr(modified, "isoformat") else (str(modified) if modified else None)
        except Exception:
            modified_iso = None
        out.append({
            "Model ID": mid,
            "Likes": getattr(m, "likes", 0) or 0,
            "Downloads": getattr(m, "downloads", 0) or 0,
            "URL": f"https://huggingface.co/{mid}",
            "Created": created_iso,
            "LastModified": modified_iso,
        })
    return out


@st.cache_data(ttl=3600, show_spinner=False)
def cached_get_model_data(model_id: str, likes: int = 0, downloads: int = 0):
    """Stage 2: per-model reasoning cache, keyed by model_id only.

    likes/downloads are accepted for back-compat but IGNORED in the key path:
    popularity is merged later via popularity_store so a like-spike doesn't
    invalidate the expensive KL/refusal parse. Pass model_id alone going forward.
    """
    proxy = SimpleNamespace(id=model_id, likes=0, downloads=0)
    return get_model_data(proxy)


@st.cache_data(ttl=3600, show_spinner=False)  # Cache for 1 hour, back-compat shim
def get_heretic_models():
    """Fetches all heretic models and their metrics concurrently."""
    meta = list_heretic_metadata()
    data = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        results = executor.map(
            lambda d: cached_get_model_data(d["Model ID"], d["Likes"], d["Downloads"]),
            meta,
        )
        for res in results:
            if res is not None:
                data.append(res)
    return data

# ---------------------------------------------------------------------------
# Filtering (pure, testable)
# ---------------------------------------------------------------------------
def _model_id_series(df: pd.DataFrame) -> pd.Series:
    col = "Model ID" if "Model ID" in df.columns else "Model"
    if col in df.columns:
        return df[col].astype("string")
    return pd.Series([""] * len(df), index=df.index, dtype="string")


def _parse_created(v):
    """Parse Created ISO string to aware datetime, or None."""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    if isinstance(v, datetime):
        dt = v
    else:
        try:
            dt = datetime.fromisoformat(str(v).replace("Z", "+00:00"))
        except Exception:
            return None
    if dt is not None and dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def filter_metadata_by_date(metadata: list, date_range_key: str, unknowns_key: str = "Include unknowns") -> list:
    """Pre-filter raw metadata list by Created date BEFORE expensive parsing.

    Unknowns handling is now an explicit user choice (not silently kept):
    - "Include unknowns": keep models with missing Created date (old default).
    - "Hide unknowns": drop them — strictest, smallest parse set.
    - "Only unknowns": parse ONLY models with missing dates (audit mode).
    """
    days = DATE_OPTIONS.get(date_range_key, None)
    if unknowns_key not in UNKNOWN_OPTIONS:
        unknowns_key = "Include unknowns"
    if days is None and unknowns_key == "Include unknowns":
        return list(metadata)
    cutoff = None
    if days is not None:
        cutoff = datetime.now(timezone.utc) - pd.Timedelta(days=days).to_pytimedelta()
    kept = []
    for d in metadata:
        dt = _parse_created(d.get("Created"))
        if dt is None:
            if unknowns_key in ("Include unknowns", "Only unknowns"):
                kept.append(d)
        elif cutoff is None:
            # All time + Hide unknowns -> drop unknowns (already skipped), keep dated
            kept.append(d)
        elif unknowns_key == "Only unknowns":
            continue  # dated rows excluded in audit mode
        elif dt >= cutoff:
            kept.append(d)
    return kept


def prefilter_metadata(metadata: list, query: str, show_quants: bool, date_range_key: str, unknowns_key: str = "Include unknowns") -> list:
    """Cheap pre-fetch filter: date (+unknowns handling) + quant + query. No network."""
    out = filter_metadata_by_date(metadata, date_range_key, unknowns_key)
    if not show_quants:
        out = [d for d in out if not QUANT_PATTERNS.search(d.get("Model ID", ""))]
    if query is not None:
        q = str(query).strip().lower()
        if q:
            out = [d for d in out if q in str(d.get("Model ID", "")).lower()]
    return out


def apply_filters(df, query, kl_range, refusal_range,
                  min_likes, min_downloads, show_quants, date_range="All time", unknowns="Include unknowns"):
    """Pure filter. Never mutates input. Handles empty df, NaNs, case-insensitive query."""
    if df is None or len(df) == 0:
        return df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame()
    mask = pd.Series(True, index=df.index)
    if query is not None:
        q = str(query).strip()
        if q:
            ids = _model_id_series(df).fillna("")
            mask &= ids.str.contains(q, case=False, na=False, regex=False)
    if kl_range is not None and "KL Divergence" in df.columns:
        try:
            lo, hi = float(kl_range[0]), float(kl_range[1])
            lo, hi = (hi, lo) if lo > hi else (lo, hi)
            kl = pd.to_numeric(df["KL Divergence"], errors="coerce")
            mask &= kl.between(lo, hi, inclusive="both")
        except (TypeError, ValueError, IndexError):
            pass
    if refusal_range is not None and "Refusal Rate (%)" in df.columns:
        try:
            lo, hi = float(refusal_range[0]), float(refusal_range[1])
            lo, hi = (hi, lo) if lo > hi else (lo, hi)
            rr = pd.to_numeric(df["Refusal Rate (%)"], errors="coerce")
            mask &= rr.between(lo, hi, inclusive="both")
        except (TypeError, ValueError, IndexError):
            pass
    for col, thresh in (("Likes", min_likes), ("Downloads", min_downloads)):
        if col in df.columns and thresh is not None:
            try:
                t = float(thresh)
            except (TypeError, ValueError):
                continue
            if pd.isna(t) or t <= 0:
                continue
            vals = pd.to_numeric(df[col], errors="coerce").fillna(0)
            mask &= vals >= t
    if not show_quants:
        ids = _model_id_series(df).fillna("")
        mask &= ~ids.str.contains(QUANT_RE, case=False, na=False, regex=True)
    # Date filter on Created column (post-fetch safety net; main saving is pre-fetch)
    days = DATE_OPTIONS.get(date_range, None)
    if unknowns not in UNKNOWN_OPTIONS:
        unknowns = "Include unknowns"
    if "Created" in df.columns and (days is not None or unknowns != "Include unknowns"):
        try:
            cutoff = None
            if days is not None:
                cutoff = datetime.now(timezone.utc) - pd.Timedelta(days=days).to_pytimedelta()
            parsed = df["Created"].apply(_parse_created)
            if unknowns == "Only unknowns":
                date_mask = parsed.apply(lambda dt: dt is None)
            elif unknowns == "Hide unknowns":
                if cutoff is None:
                    date_mask = parsed.apply(lambda dt: dt is not None)
                else:
                    date_mask = parsed.apply(lambda dt: dt is not None and dt >= cutoff)
            else:  # Include unknowns
                if cutoff is None:
                    date_mask = pd.Series(True, index=df.index)
                else:
                    date_mask = parsed.apply(lambda dt: True if dt is None else dt >= cutoff)
            mask &= date_mask.fillna(True).to_numpy() if hasattr(date_mask, "fillna") else date_mask
        except Exception:
            pass
    return df.loc[mask].copy().reset_index(drop=True)


def _qp_first(key, default=""):
    v = st.query_params.get(key, default)
    return v[0] if isinstance(v, list) and v else v


def _to_float(v, default):
    try:
        f = float(v)
        return f if not pd.isna(f) else default
    except (TypeError, ValueError):
        return default


def _to_int(v, default):
    try:
        return max(0, int(float(v)))
    except (TypeError, ValueError):
        return default


def init_filter_state_from_query_params():
    if st.session_state.get("_qp_init"):
        return
    d = FILTER_DEFAULTS
    kl_lo = _to_float(_qp_first("kl_min", d["kl_range"][0]), d["kl_range"][0])
    kl_hi = _to_float(_qp_first("kl_max", d["kl_range"][1]), d["kl_range"][1])
    rr_lo = _to_float(_qp_first("rr_min", d["refusal_range"][0]), d["refusal_range"][0])
    rr_hi = _to_float(_qp_first("rr_max", d["refusal_range"][1]), d["refusal_range"][1])
    st.session_state.setdefault("query", str(_qp_first("q", d["query"])))
    _dr = str(_qp_first("date_range", d["date_range"]))
    st.session_state.setdefault("date_range", _dr if _dr in DATE_OPTIONS else d["date_range"])
    _unk = str(_qp_first("unknowns", d["unknowns"]))
    st.session_state.setdefault("unknowns", _unk if _unk in UNKNOWN_OPTIONS else d["unknowns"])
    st.session_state.setdefault("kl_range", (min(kl_lo, kl_hi), max(kl_lo, kl_hi)))
    st.session_state.setdefault("refusal_range", (min(rr_lo, rr_hi), max(rr_lo, rr_hi)))
    st.session_state.setdefault("min_likes", _to_int(_qp_first("min_likes", d["min_likes"]), d["min_likes"]))
    st.session_state.setdefault("min_downloads", _to_int(_qp_first("min_downloads", d["min_downloads"]), d["min_downloads"]))
    st.session_state.setdefault("show_quants", str(_qp_first("show_quants", "")).lower() in ("1", "true", "yes"))
    sort = str(_qp_first("sort_by", d["sort_by"]))
    st.session_state.setdefault("sort_by", sort if sort in SORT_OPTIONS else d["sort_by"])
    st.session_state["_qp_init"] = True


def sync_filter_state_to_query_params():
    st.query_params["q"] = st.session_state.get("query", "")
    st.query_params["date_range"] = str(st.session_state.get("date_range", FILTER_DEFAULTS["date_range"]))
    st.query_params["unknowns"] = str(st.session_state.get("unknowns", FILTER_DEFAULTS["unknowns"]))
    kl = st.session_state.get("kl_range", FILTER_DEFAULTS["kl_range"])
    rr = st.session_state.get("refusal_range", FILTER_DEFAULTS["refusal_range"])
    st.query_params["kl_min"] = str(float(kl[0]))
    st.query_params["kl_max"] = str(float(kl[1]))
    st.query_params["rr_min"] = str(float(rr[0]))
    st.query_params["rr_max"] = str(float(rr[1]))
    st.query_params["min_likes"] = str(int(st.session_state.get("min_likes", 0) or 0))
    st.query_params["min_downloads"] = str(int(st.session_state.get("min_downloads", 0) or 0))
    st.query_params["show_quants"] = "1" if st.session_state.get("show_quants") else "0"
    st.query_params["sort_by"] = str(st.session_state.get("sort_by", FILTER_DEFAULTS["sort_by"]))


def _short(url: str) -> str:
    s = str(url)
    return s.split("huggingface.co/")[-1] if "huggingface.co/" in s else s


def _parse_fraction(s) -> tuple:
    """Parse '15/100' → (15, 100). Returns (None, None) on failure."""
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return (None, None)
    m = re.search(r"(\d+)\s*/\s*(\d+)", str(s))
    if not m:
        return (None, None)
    try:
        return (int(m.group(1)), int(m.group(2)))
    except (ValueError, ZeroDivisionError):
        return (None, None)


def _kl_band(kl) -> str:
    """Rough weight-impact bands. Not a verdict — higher KL = more brain drift."""
    if kl is None or (isinstance(kl, float) and pd.isna(kl)):
        return "unknown"
    try:
        kl = float(kl)
    except (TypeError, ValueError):
        return "unknown"
    if kl < 0.01:
        return "minimal drift"
    if kl < 0.05:
        return "moderate drift"
    if kl < 0.15:
        return "high drift"
    return "very high drift"


def _compare_note(kl, ref_num, ref_den, init_num, init_den) -> str:
    """One-line grain-of-salt note per model. No trophies."""
    parts = []
    # Sample-size caution
    if ref_den is None:
        parts.append("unknown sample size — weak evidence")
    elif ref_den < 50:
        parts.append(f"small sample (n={ref_den}) — take with grain of salt")
    # Drop vs base
    if init_num is not None and ref_num is not None and init_den and ref_den:
        drop = init_num - ref_num
        parts.append(f"drop {drop} (base {init_num}/{init_den} → {ref_num}/{ref_den})")
    # KL vs refusal tradeoff — the core complaint: 0% refusal + high KL = stupid
    try:
        klf = float(kl) if kl is not None else None
    except (TypeError, ValueError):
        klf = None
    try:
        rate = (ref_num / ref_den * 100) if ref_num is not None and ref_den else None
    except ZeroDivisionError:
        rate = None
    if rate is not None and klf is not None:
        if rate == 0 and klf > 0.1:
            parts.append("0 refusals BUT high KL — possibly lobotomized, check reasoning")
        elif rate == 0 and klf > 0.05:
            parts.append("0 refusals with notable drift — verify quality")
        elif rate <= 5 and klf is not None and klf < 0.02:
            parts.append("good balance: low refusal + low drift")
        elif rate > 20:
            parts.append("still refusal-prone, regardless of KL")
    elif klf is not None and klf > 0.1:
        parts.append("high KL — expect reasoning damage")
    return "; ".join(parts) if parts else "—"


# ---------------------------------------------------------------------------
# Main execution: metadata instantly, pre-filter by date, then parse subset
# ---------------------------------------------------------------------------
init_filter_state_from_query_params()


def _reset_filters():
    """Callback for Reset button — runs BEFORE widgets instantiate, so no
    StreamlitWidgetAlreadyInstantiatedError (the crash you saw)."""
    for k, v in FILTER_DEFAULTS.items():
        st.session_state[k] = v
    st.query_params.clear()
    st.session_state.metrics_done = False
    st.session_state.metrics_rows = []
    st.session_state.metrics_signature = None


if "metrics_done" not in st.session_state:
    st.session_state.metrics_done = False
if "metrics_rows" not in st.session_state:
    st.session_state.metrics_rows = []
if "metrics_signature" not in st.session_state:
    st.session_state.metrics_signature = None

# --- EARLY pre-filters: these decide HOW MANY models get parsed ---
# Shown before any expensive README/reproduce.json fetching.
st.sidebar.header("Load scope (saves parsing time)")
early_date = st.sidebar.selectbox(
    "Release date",
    list(DATE_OPTIONS.keys()),
    key="date_range",
    help="Pre-filters by Hugging Face creation date BEFORE parsing. Pick a recent window to avoid parsing ~2000 models.",
)
early_unknowns = st.sidebar.selectbox(
    "Unknown dates",
    list(UNKNOWN_OPTIONS),
    key="unknowns",
    help="Models with missing HF creation date. Include (old default), Hide (strictest, smallest parse), or Only (audit them).",
)
early_query = st.sidebar.text_input(
    "Search model ID (pre-filter)",
    key="query",
    placeholder="e.g. llama-3 heretic …",
    help="Case-insensitive substring. Applied BEFORE parsing to skip non-matching models.",
)
early_quants = st.sidebar.checkbox(
    "Show quantised models",
    key="show_quants",
    help="Untick to hide GGUF/MLX/AWQ/GPTQ/EXL2/EXL3/quant/INT8/INT4/OQ/MXFP4/MXFP8 before parsing.",
)
try:
    from src.popularity_store import load_popularity, save_popularity, snapshot_from_metadata
except ImportError:
    try:
        from popularity_store import load_popularity, save_popularity, snapshot_from_metadata
    except ImportError:
        load_popularity = lambda: {}  # noqa: E731
        save_popularity = lambda m: ""  # noqa: E731
        snapshot_from_metadata = lambda md: {}  # noqa: E731
with st.sidebar.expander("Popularity (likes/downloads)", expanded=False):
    st.caption("Stored separately in data/popularity.json — refreshable without re-parsing READMEs.")
    if st.button("🔄 Refresh popularity file", use_container_width=True):
        try:
            _live = list_heretic_metadata()
            _path = save_popularity(snapshot_from_metadata(_live))
            st.success(f"Saved {len(_live)} entries to {_path or 'data/popularity.json'}")
        except Exception as e:
            st.error(f"Could not save popularity: {e}")
    try:
        _pop = load_popularity()
        st.caption(f"{len(_pop)} models in popularity file.")
    except Exception:
        pass

col_a, col_b = st.columns([1, 4])
with col_a:
    refresh = st.button("🔄 Refresh metrics", use_container_width=True)
with col_b:
    auto = st.checkbox("Auto-load metrics", value=True)

try:
    with st.spinner("Fetching model list (fast metadata)..."):
        metadata = list_heretic_metadata()
except Exception as e:
    st.error(f"Could not list heretic models: {e}")
    st.stop()

# Cheap pre-filter BEFORE expensive parsing — this is the load-time win.
target_meta = prefilter_metadata(metadata, early_query, early_quants, early_date, early_unknowns)
st.caption(
    f"Found {len(metadata)} models (metadata cached 1h). "
    f"Scope '{early_date}' + '{early_unknowns}' + search → **{len(target_meta)} to parse**."
)
if len(target_meta) == 0:
    st.info("No models in this date/search scope — widen the date range, set Unknowns to Include, or clear search.")
    st.stop()
if len(target_meta) > 500:
    st.warning(
        f"Scope selects {len(target_meta)} models — parsing will take a while. "
        "Tip: pick 'Last month' / 'Last week', Hide unknowns, or add a search term."
    )

fetch_signature = (early_date, early_unknowns, early_query.strip().lower() if early_query else "", bool(early_quants))
if st.session_state.metrics_signature is not None and st.session_state.metrics_signature != fetch_signature:
    # Scope changed → old parsed rows no longer valid
    st.session_state.metrics_done = False
    st.session_state.metrics_rows = []

should_run = (auto and not st.session_state.metrics_done) or refresh
if refresh:
    st.session_state.metrics_done = False
    st.session_state.metrics_rows = []

if st.session_state.metrics_done and st.session_state.metrics_signature == fetch_signature:
    models_data = st.session_state.metrics_rows
elif should_run:
    status_slot = st.empty()
    progress = st.progress(0)
    rows = []
    with status_slot.status("Fetching reproduce.json → README fallback...", expanded=False) as status:
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as ex:
            fut_to_id = {
                ex.submit(cached_get_model_data, d["Model ID"], d["Likes"], d["Downloads"]): d["Model ID"]
                for d in target_meta
            }
            done = 0
            for fut in concurrent.futures.as_completed(fut_to_id):
                try:
                    res = fut.result()
                    if res is not None:
                        rows.append(res)
                except Exception as e:
                    logging.warning(f"Worker failed for {fut_to_id[fut]}: {e}")
                done += 1
                progress.progress(done / max(len(target_meta), 1))
                status.update(label=f"Parsed {done}/{len(target_meta)} — {len(rows)} with metrics...")
            status.update(label=f"Done: {len(rows)}/{len(target_meta)} with metrics", state="complete")
    # Merge Created dates back in for display/filtering + popularity
    # (likes/downloads live in popularity_store, NOT in the reasoning cache key)
    created_map = {d["Model ID"]: d.get("Created") for d in target_meta}
    meta_like_map = {d["Model ID"]: d for d in target_meta}
    try:
        _file_map = load_popularity()
    except Exception:
        _file_map = {}
    for r in rows:
        r.setdefault("Created", created_map.get(r.get("Model ID")))
        mid = r.get("Model ID")
        file_hit = (_file_map or {}).get(mid, {}) if mid else {}
        meta_hit = meta_like_map.get(mid, {}) if mid else {}
        if isinstance(file_hit, dict) and ("likes" in file_hit or "downloads" in file_hit):
            try:
                r["Likes"] = int(file_hit.get("likes", r.get("Likes", 0)) or 0)
            except (TypeError, ValueError):
                pass
            try:
                r["Downloads"] = int(file_hit.get("downloads", r.get("Downloads", 0)) or 0)
            except (TypeError, ValueError):
                pass
        elif mid in meta_like_map:
            try:
                r["Likes"] = int(meta_hit.get("Likes", r.get("Likes", 0)) or 0)
            except (TypeError, ValueError):
                pass
            try:
                r["Downloads"] = int(meta_hit.get("Downloads", r.get("Downloads", 0)) or 0)
            except (TypeError, ValueError):
                pass
    st.session_state.metrics_rows = rows
    st.session_state.metrics_done = True
    st.session_state.metrics_signature = fetch_signature
    progress.empty()
    status_slot.empty()
    models_data = rows
    st.rerun()
else:
    st.info("Tick Auto-load or press Refresh to fetch KL/Refusals.")
    st.stop()

if not models_data:
    st.warning("No models found with the 'heretic' tag in this scope.")
    st.stop()

df = pd.DataFrame(models_data)
# Make Model ID a clickable link
df["Model"] = df["URL"]
# Select and order columns for display (Created kept for filtering, shown in table too)
cols = ["Model", "KL Divergence", "Initial Refusals", "Refusals", "Refusal Rate (%)", "Likes", "Downloads"]
if "Created" in df.columns:
    cols.append("Created")
display_df = df[cols]

# --- LATE metric filters (need parsed KL/Refusals) ---
# NOTE: no writes to st.session_state widget keys here — that was the
# StreamlitWidgetAlreadyInstantiatedError. Slider max is extended to fit the
# stored value instead of writing back.
kl_max_data = float(pd.to_numeric(display_df["KL Divergence"], errors="coerce").max() or 0.5)
_cur_kl = st.session_state.get("kl_range", FILTER_DEFAULTS["kl_range"])
try:
    _cur_hi = float(_cur_kl[1])
except Exception:
    _cur_hi = 0.5
kl_slider_max = max(0.5, kl_max_data, _cur_hi)

toolbar_sort, toolbar_reset = st.columns([3, 1])
with toolbar_sort:
    sort_by = st.selectbox("Sort by", list(SORT_OPTIONS.keys()), key="sort_by")
with toolbar_reset:
    st.write("")
    st.button("Reset filters", key="reset_filters", use_container_width=True, on_click=_reset_filters)

with st.sidebar:
    st.header("Metric filters")
    kl_range = st.slider(
        "KL Divergence",
        min_value=0.0, max_value=float(kl_slider_max), step=0.001, format="%.3f",
        key="kl_range",
    )
    refusal_range = st.slider(
        "Refusal Rate (%)",
        min_value=0.0, max_value=100.0, step=0.5, format="%.1f",
        key="refusal_range",
    )
    min_likes = st.number_input("Min Likes", min_value=0, step=1, key="min_likes")
    min_downloads = st.number_input("Min Downloads", min_value=0, step=10, key="min_downloads")

sync_filter_state_to_query_params()

query = st.session_state.get("query", "")
show_quants = st.session_state.get("show_quants", False)
date_range = st.session_state.get("date_range", "All time")
unknowns = st.session_state.get("unknowns", "Include unknowns")
filtered_df = apply_filters(display_df, query, kl_range, refusal_range, min_likes, min_downloads, show_quants, date_range, unknowns)
sort_col, ascending = SORT_OPTIONS.get(sort_by, ("KL Divergence", True))
if sort_col in filtered_df.columns:
    filtered_df = filtered_df.sort_values(sort_col, ascending=ascending, kind="mergesort").reset_index(drop=True)

st.markdown(f"**Showing {len(filtered_df)} of {len(display_df)} models.**")

try:
    import altair as alt
    _ALT_OK = True
except ImportError:
    alt = None  # type: ignore
    _ALT_OK = False

tab_table, tab_compare, tab_charts = st.tabs(["Table", "Compare", "Charts"])

# 1. Table — original config preserved
with tab_table:
    if filtered_df.empty:
        st.info("No models match the current filter.")
    else:
        st.dataframe(
            filtered_df,
            column_config={
                "Model": st.column_config.LinkColumn("Model", display_text=r"https://huggingface\.co/(.*)"),
                "KL Divergence": st.column_config.NumberColumn("KL Divergence", format="%.4f"),
                "Initial Refusals": st.column_config.TextColumn("Initial Refusals"),
                "Refusals": st.column_config.TextColumn("Refusals"),
                "Refusal Rate (%)": st.column_config.NumberColumn("Refusal Rate (%)", format="%.2f%%"),
                "Likes": st.column_config.NumberColumn("Likes"),
                "Downloads": st.column_config.NumberColumn("Downloads"),
                "Created": st.column_config.TextColumn("Created"),
            },
            hide_index=True,
            width='stretch'
        )

# 2. Compare — up to 4, transposed, NO winner trophies
with tab_compare:
    if filtered_df.empty:
        st.info("No models to compare — adjust filter.")
    else:
        st.caption(
            "Grain of salt: a 0/x refusal alone proves little. "
            "A 30/30 → 0/30 on a tiny sample is weaker than 95/100 → 2/100, "
            "and zero refusals with high KL often means the weights got wrecked — "
            "uncensored but stupid. Read Refusal **with** KL + sample size."
        )
        options = filtered_df["Model"].tolist()
        selected = st.multiselect(
            "Select up to 4 models to compare",
            options=options,
            default=options[:2] if len(options) >= 2 else options,
            max_selections=4,
            format_func=_short,
        )
        if len(selected) < 2:
            st.info("Select at least 2 models to compare.")
        else:
            comp = filtered_df[filtered_df["Model"].isin(selected)].set_index("Model").reindex(selected)
            comp["KL Divergence"] = pd.to_numeric(comp["KL Divergence"], errors="coerce")
            comp["Refusal Rate (%)"] = pd.to_numeric(comp["Refusal Rate (%)"], errors="coerce")
            # Per-model honest summary (no best_kl / best_ref verdicts)
            for m in selected:
                row = comp.loc[m]
                rn, rd = _parse_fraction(row.get("Refusals"))
                inn, ind = _parse_fraction(row.get("Initial Refusals"))
                note = _compare_note(row.get("KL Divergence"), rn, rd, inn, ind)
                band = _kl_band(row.get("KL Divergence"))
                st.info(f"**{_short(m)}** — KL {band} | {note}")
            disp = comp.T.copy()
            disp.columns = [_short(c) for c in disp.columns]
            # Extra derived rows for honest comparison
            try:
                kl_vals = pd.to_numeric(comp["KL Divergence"], errors="coerce")
                disp.loc["KL band"] = [_kl_band(v) for v in kl_vals.reindex(selected).tolist()]
            except Exception:
                pass
            try:
                drops = []
                for m in selected:
                    rn, rd = _parse_fraction(comp.loc[m].get("Refusals"))
                    inn, ind = _parse_fraction(comp.loc[m].get("Initial Refusals"))
                    drops.append(str(inn - rn) if inn is not None and rn is not None else "—")
                disp.loc["Refusal drop (base − final)"] = drops
            except Exception:
                pass
            try:
                ns = []
                for m in selected:
                    _, rd = _parse_fraction(comp.loc[m].get("Refusals"))
                    ns.append(str(rd) if rd is not None else "—")
                disp.loc["Sample size (n prompts)"] = ns
            except Exception:
                pass
            for metric in list(disp.index):
                for col_url, col_short in zip(selected, disp.columns):
                    v = comp.loc[col_url, metric] if metric in comp.columns else disp.loc[metric, col_short]
                    if pd.isna(v):
                        disp.loc[metric, col_short] = "—"
                    elif metric == "KL Divergence":
                        try:
                            disp.loc[metric, col_short] = f"{float(v):.4f}"
                        except (TypeError, ValueError):
                            disp.loc[metric, col_short] = str(v)
                    elif metric == "Refusal Rate (%)":
                        try:
                            disp.loc[metric, col_short] = f"{float(v):.2f}%"
                        except (TypeError, ValueError):
                            disp.loc[metric, col_short] = str(v)
                    elif metric in ("Likes", "Downloads"):
                        try:
                            disp.loc[metric, col_short] = f"{int(float(v)):,}"
                        except Exception:
                            disp.loc[metric, col_short] = str(v)
                    else:
                        disp.loc[metric, col_short] = str(v) if v is not None else "—"
            disp.index.name = "Metric"
            st.table(disp)

# 3. Charts — altair, NaN/empty safe
with tab_charts:
    if not _ALT_OK:
        st.warning("`altair` is not installed. Add `altair` to requirements.txt and reboot.")
    elif filtered_df.empty:
        st.info("No data to chart — adjust filter.")
    else:
        chart_df = filtered_df.copy()
        chart_df["Short"] = chart_df["Model"].astype(str).str.split("huggingface.co/").str[-1]
        chart_df["KL Divergence"] = pd.to_numeric(chart_df["KL Divergence"], errors="coerce")
        chart_df["Refusal Rate (%)"] = pd.to_numeric(chart_df["Refusal Rate (%)"], errors="coerce")
        chart_df["Likes"] = pd.to_numeric(chart_df["Likes"], errors="coerce").fillna(0)
        chart_df["Downloads"] = pd.to_numeric(chart_df["Downloads"], errors="coerce").fillna(0)
        scatter_df = chart_df.dropna(subset=["KL Divergence", "Refusal Rate (%)"])
        if scatter_df.empty:
            st.info("No valid KL / Refusal Rate values to plot.")
        else:
            st.subheader("KL (x) vs Refusal Rate (y), size=Downloads")
            scatter = (
                alt.Chart(scatter_df)
                .mark_circle(opacity=0.7)
                .encode(
                    x=alt.X("KL Divergence:Q", scale=alt.Scale(zero=False)),
                    y=alt.Y("Refusal Rate (%):Q", scale=alt.Scale(zero=False)),
                    size=alt.Size("Downloads:Q", scale=alt.Scale(range=[20, 500])),
                    tooltip=["Short:N", "KL Divergence:Q", "Refusal Rate (%):Q", "Likes:Q", "Downloads:Q"],
                )
                .interactive()
            )
            st.altair_chart(scatter, use_container_width=True)
            st.subheader("Refusal Rate per model (top 20 by Likes)")
            top20 = chart_df.nlargest(20, "Likes").sort_values("Refusal Rate (%)", ascending=True)
            top20 = top20.dropna(subset=["Refusal Rate (%)"])
            if top20.empty:
                st.info("No valid Refusal Rate values for bar chart.")
            else:
                bars = (
                    alt.Chart(top20)
                    .mark_bar()
                    .encode(
                        x=alt.X("Refusal Rate (%):Q"),
                        y=alt.Y("Short:N", sort="-x", title="Model"),
                        tooltip=["Short:N", "Refusal Rate (%):Q", "KL Divergence:Q", "Likes:Q", "Downloads:Q"],
                    )
                )
                st.altair_chart(bars, use_container_width=True)
            st.subheader("Distribution of KL Divergence")
            kl_df = chart_df.dropna(subset=["KL Divergence"])
            if kl_df.empty:
                st.info("No valid KL values for histogram.")
            else:
                hist = (
                    alt.Chart(kl_df)
                    .mark_bar()
                    .encode(
                        x=alt.X("KL Divergence:Q", bin=alt.Bin(maxbins=20)),
                        y=alt.Y("count():Q", title="Models"),
                        tooltip=[alt.Tooltip("count():Q", title="Models")],
                    )
                )
                st.altair_chart(hist, use_container_width=True)

# Heretic Models Explorer
> original: https://huggingface.co/spaces/MuXodious/Heretic-Models-Explorer

> sdk: streamlit ;
> app_file: src/streamlit_app.py ;
> license: wtfpl

# Pull Request

> feat: enhance discovery + cache + comparison + add visuals + more ([c5a3505f2df7445adcf61799902d1506906cd645](https://huggingface.co/spaces/MuXodious/Heretic-Models-Explorer/commit/c5a3505f2df7445adcf61799902d1506906cd645))

**Core**
* Parsing dates and unknown scopes upfront cuts startup time.
* Isolating popularity data keeps external metadata syncs from blowing away the model cache.
* Current badge system can be improved with additional scores to uncover divergence issues that emerge from tiny sample sizes.

**How**

* Filter by `Release date` (All/Week/Month/6mo/Year) before README parsing so the app skips reading ~2,000 files on boot.
* Add `Unknown dates` scope: Include, Hide, and Only. Exposed via URL query parameter `?unknowns=` and applied across pre- and post-filtering.
* Decouple likes and downloads into `src/popularity_store.py` and `data/popularity.json`. Cached model calls ignore like counts, and sidebar refreshes update JSON without triggering a re-parse.
* Drop lowest-KL and lowest-refusal badges in model comparisons. Replace them with KL bands, net refusal drops, total sample counts, and warning flags for zero-refusal models with high divergence.
* Fix `StreamlitWidgetAlreadyInstantiatedError` by shifting reset logic into an `on_click` handler instead of setting state after render.
* Add `COPY data/` to Dockerfile with fallbacks for missing popularity stores.


**Test**
* Run `docker compose up --build`. Toggle between All and Last week with Hide/Only unknowns, then confirm the parse count drops.
* Click Reset to verify state clears without raising Streamlit errors.
* Test comparison view between a 30-prompt run at 0% refusal with high KL and a 100-prompt run with low KL; verify the first flags a divergence warning while the second renders clean metrics.

**Known and follow-up**

* Date checks rely strictly on `created_at` without a fallback for `last_modified`. Unknown scopes stay enabled by default for backwards compatibility.
* KL bands (0.01, 0.05, 0.15) and sample floor ($n < 50$) are hardcoded heuristics.
* Remove obsolete `get_heretic_models()` helper and clean up pinned dependencies in a separate PR.

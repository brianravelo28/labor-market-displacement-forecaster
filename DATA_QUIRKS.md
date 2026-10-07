# Data Quirks Log

Weird things found in this project's data, how each was found, what it would have broken, and what we did. Each entry is marked **Verified** (counted or reproduced directly) or **Observed** (seen, mechanism not confirmed).

---

## 1. FL median employment growth lands on an exact 0.0% because BLS carries estimates forward unchanged

**What it was:** The Regional Comparison tab's median employment growth rate for Florida, 2024, was exactly `+0.0%` — not just near zero.

**How it was found:** A question about whether it was a real result or a bug. Of 744 occupations present in both US and FL for 2024, **13 FL occupations report bit-for-bit identical `employment_level` between 2023 and 2024** (vs. only 2 for the same occupation set nationally). With that many exact ties clustered near the middle of the sorted distribution, both middle-ranked values land on 0.0%.

**Verified:** counted directly —
```python
fl['employment_change_yoy_pct'].eq(0).sum()  # 13 of 744
us['employment_change_yoy_pct'].eq(0).sum()  # 2 of the same 791 (superset)
```
The 13 are mostly small (40–630 employees: Mathematicians, Historians, Log Graders and Scalers, Dredge Operators, etc.), plus two larger ones (Chiropractors, 3,360; Structural Metal Fabricators, 2,150).

**What it would have broken:** "Florida employment growth: 0.0%" reads as a stagnant labor market. The real picture is far more dynamic — FL's 25th/75th percentile occupations moved -10.6% / +11.6%. The 0.0% is where a tie cluster happens to fall, not evidence of a frozen economy.

**Likely cause (Observed, not confirmed against BLS documentation):** for smaller state-level cells, when a survey cycle doesn't gather enough new sample to move the modeled estimate, the prior value appears to be carried forward. The pattern is a property of the published data, not of this pipeline.

**What we did:** A median-only bar at exactly 0.0% looks like a rendering error to anyone who doesn't know the backstory, so the chart now shows the same median bar plus asymmetric error bars spanning the 25th–75th percentile (see `regional_bar_with_iqr()` in `app/app.py`); FL's bar is still flat at the baseline, but its whisker (~-10.6% to +11.6%) shows the real spread. A box plot with full min/max whiskers was tried first and rejected: a few small occupations swing from about -82% to +300% (sampling noise), which stretched the axis until the IQR collapsed into an unreadable sliver. The IQR-scoped bars never include that tail, so it can't dominate.

---

## 2. BLS's OEWS API only ever holds the latest release

**What it was:** The original project plan assumed OEWS could be pulled as a monthly 2015–present series through the BLS API. It can't: OEWS is annual, and the API/flat-file time-series database keeps only the current release.

**How it was found / Verified (sampled):** In `oe.series` (BLS's series index) every sampled series — the national all-occupation rows and the Abilene metro rows at the top of the file — has `begin_year == end_year == 2025`. Querying the national all-occupation employment series (`OEUN000000000000000000001`) and a Florida retail-salespersons series with the default window returned data only for 2025 (period `A01`, "Annual"), with "No Data Available" messages for 2023 and 2024. (Sampled, not an exhaustive scan of every series.)

**What it would have broken:** The whole pipeline design — a 10-year panel can't be assembled from the API at all.

**What we did:** Switched to BLS's separate annual archive files (`oesm{YY}all.zip`, 2011–2025 available) and stacked them ourselves (`data/fetch_oews.py`). Cadence is annual, not monthly.

---

## 3. OEWS archives change schema between releases — and one change silently produced zero rows

**What it was:** Column names, casing, and even category *values* differ by release year. 2015: headers like `occ code` (space) and `group`, and the detailed-occupation value is `'detail '` — truncated, with a trailing space — instead of `'detailed'`. 2019: `occ_code` / `o_group`. 2024: `OCC_CODE` / `O_GROUP`. Cross-industry NAICS is `'0'` in some years and `'000000'` in others. The `PRIM_STATE` column is absent in both 2015 and 2019, and `i_group` is absent in 2015.

**How it was found / Verified:** A three-year test (2015, 2019, 2024) of the parser returned 2019 and 2024 row counts but **nothing for 2015** — no error, just a missing year in the `groupby` output (3,174 rows total, exactly the sum of the other two years). Inspecting the 2015 file showed the `'detail '` value; the filter `o_group == "detailed"` matched nothing. After the fix the same test returned 820 occupations for 2015. (Only 2015, 2019, and 2024 were inspected by hand; the other years parse correctly with the same synonym map.)

**What it would have broken:** A silently shorter panel — 2015 missing — with no exception to flag it.

**What we did:** Normalize headers (lowercase, spaces to underscores), map known synonyms to canonical names, accept both `detailed` and `detail` after stripping whitespace, and identify Florida by `AREA_TYPE == 2 & AREA == 12` (the state FIPS code), which is stable across every year regardless of the missing `PRIM_STATE`.

---

## 4. A JOLTS series ID tutorial was wrong, and the BLS API reports the mistake as "success"

**What it was:** A published OEWS/JOLTS tutorial's field-width breakdown gave a JOLTS series ID as 20 characters (5-digit area code). The real, live-working ID is **21** characters (6-digit area code): `JTS000000120000000JOR` = `JTS` + industry(6) + state(2) + area(6) + sizeclass(1) + data element(2) + rate/level(1).

**How it was found / Verified:** The first full fetch built IDs by the tutorial's widths (and, separately, with a doubled rate suffix — `JORR` — from a bug in our own code, which masked the width problem for one round). The API returned `REQUEST_SUCCEEDED` with `"Series does not exist for Series …"` messages and empty `data` arrays for every series — so the status check passed. The script then built an empty DataFrame and crashed several lines later with an unrelated-looking `KeyError: 'region'` on `sort_values`. Decomposing the known-working ID by character index showed seven zeros between the state code and `JOR`, not six.

**What it would have broken:** Two full 7-call runs against an unregistered API quota of 25 queries/day — most of a day's quota spent producing nothing. (The API later returned "daily threshold … has been reached" mid-run.)

**What we did:** Fixed the ID width, and added per-batch response caching in `data/fetch_jolts.py` so a successful call is never repeated. (A lesson worth keeping: check for the `message` list and non-empty `data`, not just `status`.)

---

## 5. BLS serves a bot-blocking page to generic browser user agents

**What it was:** Requests to `download.bls.gov` / `www.bls.gov` file URLs with a generic `Mozilla/5.0` user agent got back an HTML "Access Denied" page (automated-retrieval policy text) instead of the file. Identical requests with a self-identifying user agent (`project-name (contact@email)`) returned the real files.

**Verified:** reproduced with `curl` both ways against the same URL.

**What it would have broken:** Any scripted download — and the failure mode is an HTML page where a zip/text file was expected.

**What we did:** All fetch scripts send a self-identifying `User-Agent` built from `BLS_CONTACT_EMAIL` (`src/config.py`).

---

## 6. The same NAICS sector code carries four different titles across releases

**What it was:** In the industry-sector panel, NAICS `99` (government) has four distinct `naics_title` strings across the ten releases (BLS reworded the label, adding/removing the Postal Service qualifier and "OEWS/OES Designation").

**Verified:** `ind[ind.naics == '99']['naics_title'].nunique() == 4`.

**What it would have broken:** The sector dropdown would have shown multiple entries sharing one value (caught before it shipped), and any group-by on `naics_title` would split government into several groups.

**What we did:** Group on the code, and take the most recent year's title for display (`app/app.py`).

---

## 7. Suppression markers (`*`, `#`) make missing wages non-random

**What it was:** OEWS marks suppressed or out-of-range numeric cells with `*` and `#` strings, which the parser coerces to null. The nulls are concentrated in particular kinds of occupations, not scattered.

**Verified:** Rows with a null `wage_hourly_mean` (1,231 rows) have a median annual mean wage of $77,890 vs. $49,990 for the rest — mostly occupations BLS reports as annual salary only (postsecondary teachers, legislators, school administrators). Rows with a null `wage_annual_pct90` (826 rows) have a median annual mean wage of $172,320 vs. $50,430, e.g. Cardiologists and Pediatric Surgeons. (The `#` on upper-percentile cells of high-paid occupations such as Chief Executives was seen in the raw 2024 file; that it denotes top-coding is **Observed**, not checked against BLS's definition.)

**What it would have broken:** Any analysis of hourly or 90th-percentile wages silently drops higher-paid occupations, biasing results downward without any error.

**What we did:** Documented the per-column null shares in `docs/SCHEMA.md` and kept the dashboard on `wage_annual_mean`, which is null for only 1.0% of rows.

---

## 8. Sparse occupation × industry cells turn sampling noise into nonsense "top movers"

**What it was:** In the sector tab's top-movers tables, categorically odd occupations appeared — e.g. "Self-Enrichment Teachers" (130 employees) under Manufacturing — and huge percentage swings (+700%) on small bases.

**Verified:** The 130-employee teaching rows exist in the raw BLS data (they're real — likely in-house corporate trainers — not a parsing error). Within Manufacturing 2024 the employment distribution has a median of 2,600 but a minimum of 30.

**What it would have broken:** The tab looked like a data-integrity failure.

**What we did:** Excluded occupations under 500 employees in a sector from the ranked tables (`SECTOR_RANKING_MIN_EMPLOYMENT`). **This only partly fixes it:** the floor is applied to *current-year* employment, so a jump from a tiny base to just above 500 still gets through — e.g. Construction Laborers in Accommodation & Food Services shows 960 employees at +700% (from about 120). A floor on the smaller of the prior- and current-year counts would be stricter; not implemented yet.

---

## 9. Employment sizes are so heavy-tailed that a standard regression fails on the biggest occupations

**What it was:** Occupation employment spans 50 to ~4.6M. Only 7 of 1,276 training rows exceed 2M employees; the median training row is ~14,000.

**Verified:** A model trained on `log1p(employment_next)` mispredicted Retail Salespersons (the largest US occupation) by ~40–60% even on rows it was trained on (predicted ~2.1M vs. actual ~3.7M for 2020→2021), despite good aggregate error (MAPE ~13–20%).

**What it would have broken:** The headline forecast for the biggest, most-watched occupations — hidden behind a respectable overall metric.

**What we did:** Predict the log growth ratio instead of the level (full write-up in `docs/METHODOLOGY.md#model`); 1M+-employee RMSE fell from ~700K–940K to ~90K–180K.

# Data Quirks Log

Weird things found in this project's data, why they matter, and what we did about them.

---

## 1. FL median employment-growth-rate lands on an exact 0.0% due to BLS estimate carry-forward

**What it was:** The Regional Comparison tab's "Median employment growth rate, 2024 (YoY)" chart shows Florida at exactly `+0.0%`, not just close to zero.

**How it was found:** Brian asked whether that was a real result or a bug (2026-09-30). Checked the underlying `oews_occupation_features.csv`: of 744 occupations present in both US and FL for 2024, **13 FL occupations report bit-for-bit identical `employment_level` between 2023 and 2024** (vs. only 2 for the same occupation set nationally). With that many exact ties clustered near the middle of the sorted distribution, both middle-ranked values land on 0.0%, making the median exactly zero.

**Verified:** counted directly —
```python
fl['employment_change_yoy_pct'].eq(0).sum()  # 13 of 744
us['employment_change_yoy_pct'].eq(0).sum()  # 2 of the same 791 (superset)
```
The 13 zero-occupations are mostly small (40–630 employees: Mathematicians, Historians, Log Graders and Scalers, Dredge Operators, etc.), plus two larger ones (Chiropractors, 3,360; Structural Metal Fabricators, 2,150).

**What it would have broken (if unexplained):** Reading "Florida employment growth: 0.0%" at face value implies a stagnant labor market. The real picture is far more dynamic — FL's 25th/75th percentile occupations moved -10.6%/+11.6% that year. The 0.0% median is a coincidence of where a tie cluster falls, not evidence of a frozen economy.

**Likely cause:** BLS's OEWS estimation methodology for smaller state-level cells — when a survey cycle doesn't gather enough new sample to move the modeled estimate, BLS carries the prior year's published value forward unchanged rather than reporting a break or gap. This is a real property of the published data, not a bug in this project's pipeline.

**What we did:** Nothing to fix — documented it here and left the chart as-is, since it's an accurate reflection of the published BLS estimates. Relevant if this pattern ever needs explaining to a dashboard viewer, or shows up again elsewhere (e.g. the industry-sector tab, which already has a related small-sample noise issue — see `docs/METHODOLOGY.md`).

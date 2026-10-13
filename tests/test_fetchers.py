"""BLS fetchers: series IDs, API quirks, window splitting, and OEWS header drift."""
import io
import zipfile

import numpy as np
import pandas as pd
import pytest

import fetch_jolts
import fetch_oews


# ---- JOLTS ---------------------------------------------------------------

def test_series_ids_are_21_characters_and_unique():
    ids = fetch_jolts.build_series_ids()
    assert all(len(s) == 21 and s.startswith("JTS") for s in ids)
    assert len(ids) == len(set(ids))


def test_florida_only_has_total_nonfarm():
    ids = fetch_jolts.build_series_ids()
    fl = {s: m for s, m in ids.items() if m["region"] == "FL"}
    assert len(fl) == 5 and {m["industry_code"] for m in fl.values()} == {"000000"}
    assert all(s[9:11] == "12" for s in fl)      # state FIPS sits at characters 10-11
    assert all(s[9:11] == "00" for s, m in ids.items() if m["region"] == "US")


def test_total_series_count():
    assert len(fetch_jolts.build_series_ids()) == 85   # 16 US industries x 5 elements + 5 Florida


def response(series):
    return {"status": "REQUEST_SUCCEEDED", "message": [], "Results": {"series": series}}


def test_empty_series_is_an_error_even_though_the_api_says_success():
    ok = {"seriesID": "A", "data": [{"year": "2020"}]}
    gone = {"seriesID": "B", "data": []}
    fetch_jolts.check_no_empty_series(response([ok]))
    with pytest.raises(RuntimeError, match="B"):
        fetch_jolts.check_no_empty_series(response([ok, gone]))


def test_years_are_fetched_in_windows_of_at_most_ten(monkeypatch):
    calls = []

    def fake(series_ids, start, end):
        calls.append((start, end))
        return response([{"seriesID": s, "data": [{"year": str(start), "period": "M01", "value": "1"}] } for s in series_ids])

    monkeypatch.setattr(fetch_jolts, "fetch_batch", fake)
    points = fetch_jolts.fetch_windows(["S1", "S2"], 2015, 2026)
    assert calls == [(2015, 2024), (2025, 2026)]
    assert [p["year"] for p in points["S1"]] == ["2015", "2025"]


def test_cache_key_depends_on_series_and_years():
    a = fetch_jolts._cache_path(["S1"], 2015, 2024)
    assert a != fetch_jolts._cache_path(["S1"], 2025, 2026)
    assert a != fetch_jolts._cache_path(["S2"], 2015, 2024)
    assert a == fetch_jolts._cache_path(["S1"], 2015, 2024)


def test_cached_batch_makes_no_request(monkeypatch, tmp_path):
    monkeypatch.setattr(fetch_jolts, "JOLTS_RAW", tmp_path)
    path = fetch_jolts._cache_path(["S1"], 2015, 2024)
    path.write_text('{"cached": true}')

    def boom(*a, **k):
        raise AssertionError("network call attempted despite a cached response")

    monkeypatch.setattr(fetch_jolts.requests, "post", boom)
    assert fetch_jolts.fetch_batch(["S1"], 2015, 2024) == {"cached": True}


# ---- OEWS ----------------------------------------------------------------

def fake_archive(df):
    """An in-memory oesm{YY}all.zip holding one xlsx sheet."""
    xlsx = io.BytesIO()
    df.to_excel(xlsx, index=False)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("all_data.xlsx", xlsx.getvalue())
    buf.seek(0)
    return zipfile.ZipFile(buf)


def row(area, area_type, naics, code, group, emp, a_mean="50000"):
    return {"area": area, "area_title": "x", "area_type": area_type, "naics": naics, "naics_title": "t",
            "occ code": code, "occ title": f"title {code}", "group": group, "tot_emp": emp, "h_mean": "24",
            "a_mean": a_mean, "h_median": "23", "a_median": "48000", "a_pct10": "30000", "a_pct90": "*"}


@pytest.fixture
def archive_2015(monkeypatch):
    # Mimics the 2015 release: lowercase headers with spaces ("occ code") and a
    # "detail " group label with a trailing space.
    df = pd.DataFrame([
        row(99, 1, "000000", "11-1011", "detail ", 1000),            # national, detailed        -> kept
        row(12, 2, "000000", "11-1011", "detail ", 80),              # Florida                   -> kept
        row(6, 2, "000000", "11-1011", "detail ", 90),               # California                -> dropped
        row(99, 1, "000000", "11-0000", "major", 5000),              # aggregate group           -> dropped
        row(99, 1, "722000", "11-1011", "detail ", 400),             # industry cross-tab        -> dropped
        row(99, 1, "000000", "13-1011", "detail ", "**", "#"),       # suppressed markers        -> kept, as nulls
    ])
    monkeypatch.setattr(fetch_oews, "download_year", lambda year: fake_archive(df))


def test_2015_style_headers_and_labels_parse(archive_2015):
    out = fetch_oews.parse_year(2015)
    assert len(out) == 3
    assert set(out["region"]) == {"US", "FL"}
    assert {"occ_code", "occ_title", "tot_emp", "a_mean"} <= set(out.columns)


def test_only_florida_is_kept_among_states(archive_2015):
    out = fetch_oews.parse_year(2015)
    assert out[out["region"] == "FL"]["tot_emp"].tolist() == [80]


def test_suppression_markers_become_nulls_not_errors(archive_2015):
    out = fetch_oews.parse_year(2015).set_index(["region", "occ_code"])
    assert np.isnan(out.loc[("US", "13-1011"), "tot_emp"]) and np.isnan(out.loc[("US", "13-1011"), "a_mean"])
    assert np.isnan(out.loc[("US", "11-1011"), "a_pct90"])


def test_a_missing_required_column_fails_loudly(monkeypatch):
    df = pd.DataFrame([row(99, 1, "000000", "11-1011", "detail ", 1)]).drop(columns=["tot_emp"])
    monkeypatch.setattr(fetch_oews, "download_year", lambda year: fake_archive(df))
    with pytest.raises(ValueError, match="tot_emp"):
        fetch_oews.load_year_raw(2015)

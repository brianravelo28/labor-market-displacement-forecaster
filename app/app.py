"""
Occupation Employment & Real Wage Tracker -- Dash app.

Four tabs: Occupation Search, Regional Comparison (US vs FL), Industry &
Sector Analysis, and 2020 Shock & Recovery. Everything runs on the processed
OEWS tables in data/processed/, with CPI-U inflation-adjusted wages and JOLTS
job-market tightness merged in by src/build_macro_features.py. The dashboard is
descriptive: it reports what the data show and makes no forecasts.
"""
from pathlib import Path

import dash
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, dash_table, dcc, html

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
DATA_PATH = PROCESSED_DIR / "oews_occupation_features.csv"
INDUSTRY_DATA_PATH = PROCESSED_DIR / "oews_industry_features.csv"
SHOCK_PATH = PROCESSED_DIR / "shock_2020_response.csv"

# Below this (in either the current or the prior year), an occupation's
# employment count within a single industry sector is small enough that YoY %
# swings are dominated by OEWS sampling noise rather than a real trend -- see
# the ranking note in update_sector_tab.
SECTOR_RANKING_MIN_EMPLOYMENT = 500

# Interactive charts hide the mode bar and get a hint line instead, since
# displayModeBar=False makes drag-to-zoom/double-click-reset/legend-click
# otherwise undiscoverable. The legend sentence is only added for charts that
# actually have a legend (none of the current ones do).
CHART_HINT_BASE = "Drag to zoom into part of the chart; double-click the chart to reset the view."
CHART_HINT_LEGEND_CLAUSE = " Click a legend entry to hide that series; double-click it to isolate it."
NO_MODEBAR = {"displayModeBar": False}


def chart_hint(has_legend=False):
    text = CHART_HINT_BASE + (CHART_HINT_LEGEND_CLAUSE if has_legend else "")
    return html.P(text, style={"margin": "4px 2px 16px", "color": "#999", "fontSize": "12px", "fontStyle": "italic"})


df = pd.read_csv(DATA_PATH)
# Industry-sector extraction (Tab 3) re-parses the same large archives on a
# separate schedule and may not have finished yet -- degrade gracefully.
industry_df = pd.read_csv(INDUSTRY_DATA_PATH, dtype={"naics": str}) if INDUSTRY_DATA_PATH.exists() else None
# Observed 2020 employment response per occupation (src/build_shock_replay.py), for the 2020 tab.
shock_response = pd.read_csv(SHOCK_PATH).set_index(["region", "occ_code"]) if SHOCK_PATH.exists() else None
FIRST_YEAR, LATEST_YEAR = int(df["year"].min()), int(df["year"].max())
BASE_YEAR = 2019  # the last release before the 2020 shock


def _balanced_totals():
    """Per region: total employment of the occupations present in every release, indexed to BASE_YEAR = 100.

    Using one fixed set of occupations keeps the comparison line from moving just because
    SOC code changes added or dropped occupations along the way."""
    out = {}
    for region, g in df.dropna(subset=["employment_level"]).groupby("region"):
        years = sorted(g["year"].unique())
        present = g.groupby("occ_code")["year"].nunique()
        codes = present[present == len(years)].index
        totals = g[g["occ_code"].isin(codes)].groupby("year")["employment_level"].sum()
        out[region] = (totals / totals.loc[BASE_YEAR] * 100, len(codes))
    return out


BALANCED_TOTALS = _balanced_totals()


occ_options = (
    df[["occ_code", "occ_title"]]
    .drop_duplicates()
    .sort_values("occ_title")
    .assign(label=lambda d: d["occ_title"] + " (" + d["occ_code"] + ")")
)
DEFAULT_OCC = df.loc[df["employment_level"].idxmax(), "occ_code"]

app = dash.Dash(__name__)
app.title = "Occupation Employment & Real Wage Tracker"
server = app.server  # exposes the underlying Flask app for gunicorn (see Procfile)

tab1_content = html.Div(
    children=[
        html.Div(
            style={"display": "flex", "gap": "16px", "marginBottom": "20px", "alignItems": "center", "marginTop": "20px"},
            children=[
                dcc.Dropdown(
                    id="occ-dropdown",
                    options=[{"label": row.label, "value": row.occ_code} for row in occ_options.itertuples()],
                    value=DEFAULT_OCC,
                    style={"flex": "1", "minWidth": "320px"},
                    placeholder="Search for an occupation...",
                ),
                dcc.RadioItems(
                    id="region-toggle",
                    options=[{"label": " National (US)", "value": "US"}, {"label": " Florida", "value": "FL"}],
                    value="US",
                    inline=True,
                ),
            ],
        ),
        html.Div(id="kpi-row", style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit, minmax(132px, 1fr))", "gap": "16px", "marginBottom": "24px"}),
        dcc.Graph(id="employment-chart", config=NO_MODEBAR),
        chart_hint(),
        dcc.Graph(id="wage-chart", config=NO_MODEBAR),
        chart_hint(has_legend=True),
    ]
)

tab2_content = html.Div(
    style={"marginTop": "20px"},
    children=[
        html.P(
            f"Occupations present in both US and FL for the most recent year ({LATEST_YEAR}). "
            "'Real' wage changes are deflated with the national CPI-U (May to May), so both regions "
            "get the same inflation rate; the job openings rate is each region's JOLTS total nonfarm "
            "series. The employment growth chart's whiskers span the 25th-75th percentile (not the full range, which "
            "is dominated by a handful of small occupations with extreme swings from OEWS sampling "
            "noise) -- so a median near zero can still sit on top of a wide spread.",
            style={"color": "#666"},
        ),
        dcc.RadioItems(
            id="regional-weighting",
            options=[
                {"label": " Each occupation counts equally", "value": "occupation"},
                {"label": " Each worker counts equally (weighted by employment)", "value": "worker"},
            ],
            value="occupation",
            inline=True,
            inputStyle={"marginLeft": "16px", "marginRight": "4px"},
            style={"margin": "4px 0 8px"},
        ),
        html.P(
            "By occupation, a 500-person occupation counts as much as Retail Salespersons (millions of workers). "
            "By worker, each occupation counts in proportion to its employment in that region, so the figures "
            "describe what the typical worker rather than the typical occupation experiences. Weights are each "
            f"region's own {LATEST_YEAR} employment; the job openings rate is not affected.",
            style={"color": "#888", "fontSize": "13px", "marginTop": 0},
        ),
        html.Div(
            style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "16px"},
            children=[
                dcc.Graph(id="regional-employment-growth"),
                dcc.Graph(id="regional-wage-level"),
                dcc.Graph(id="regional-wage-decline-prevalence"),
                dcc.Graph(id="regional-real-wage-decline-prevalence"),
                dcc.Graph(id="regional-contraction-prevalence"),
                dcc.Graph(id="regional-job-openings"),
            ],
        ),
    ],
)

if industry_df is not None:
    if "naics_title" in industry_df.columns:
        # naics_title wording for some codes (e.g. "99" government) changed across
        # release years -- keep only the most recent year's title per code so the
        # dropdown doesn't show duplicate entries for the same naics value.
        sector_options = (
            industry_df.sort_values("year")
            .groupby("naics")["naics_title"]
            .last()
            .reset_index()
            .sort_values("naics_title")
        )
    else:
        sector_options = industry_df[["naics"]].drop_duplicates().assign(naics_title=lambda d: d["naics"]).sort_values("naics_title")
    DEFAULT_SECTOR = sector_options["naics"].iloc[0]
    tab3_content = html.Div(
        children=[
            dcc.Dropdown(
                id="sector-dropdown",
                options=[{"label": row.naics_title, "value": row.naics} for row in sector_options.itertuples()],
                value=DEFAULT_SECTOR,
                style={"marginTop": "20px", "marginBottom": "12px", "maxWidth": "500px"},
            ),
            dcc.RadioItems(
                id="sector-wage-basis",
                options=[
                    {"label": " Real wage growth (after CPI inflation)", "value": "real"},
                    {"label": " Nominal wage growth", "value": "nominal"},
                ],
                value="real",
                inline=True,
                inputStyle={"marginLeft": "16px", "marginRight": "4px"},
                style={"marginBottom": "16px"},
            ),
            html.Div(id="sector-jolts-row", style={"display": "flex", "flexWrap": "wrap", "gap": "16px", "marginBottom": "16px"}),
            dcc.Graph(id="sector-scatter", config=NO_MODEBAR),
            chart_hint(),
            html.Div(
                style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "16px", "marginTop": "8px"},
                children=[
                    html.Div([html.H4("Top 10 contracting"), html.Div(id="sector-top-contracting")]),
                    html.Div([html.H4("Top 10 growing"), html.Div(id="sector-top-growth")]),
                ],
            ),
        ]
    )
else:
    tab3_content = html.Div(
        style={"marginTop": "20px", "color": "#999"},
        children="Industry-sector data not yet available (still parsing OEWS archives -- rerun src/fetch_oews_industry.py then reload).",
    )

tab4_content = html.Div(
    children=[
        html.P(
            "How did occupations fare when the 2020 recession hit, and how far have they come back? Employment is "
            f"shown relative to the {BASE_YEAR} release (= 100). The 2020 response is the occupation's change in the "
            "2020 release minus how fast it had been growing in 2016-2019, so only the part plausibly due to the "
            "shock remains.",
            style={"color": "#666", "marginTop": "20px"},
        ),
        html.P(
            "Two cautions. Each OEWS release pools three years of survey data, so the 2020 release shows the "
            "shock smeared out and the rebound arrives mostly in the 2022 release. And an occupation's own "
            "year-to-year change is very noisy below about 10,000 employees, so smaller occupations show their "
            "occupation group's response instead (the card says which was used).",
            style={"color": "#888", "fontSize": "13px"},
        ),
        html.Div(
            style={"display": "flex", "gap": "16px", "marginBottom": "20px", "alignItems": "center", "marginTop": "16px"},
            children=[
                dcc.Dropdown(
                    id="shock-occ-dropdown",
                    options=[{"label": row.label, "value": row.occ_code} for row in occ_options.itertuples()],
                    value=DEFAULT_OCC,
                    style={"flex": "1", "minWidth": "320px"},
                    placeholder="Search for an occupation...",
                ),
                dcc.RadioItems(
                    id="shock-region-toggle",
                    options=[{"label": " National (US)", "value": "US"}, {"label": " Florida", "value": "FL"}],
                    value="US",
                    inline=True,
                ),
            ],
        ),
        html.Div(id="shock-kpi-row", style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit, minmax(160px, 1fr))", "gap": "16px", "marginBottom": "24px"}),
        dcc.Graph(id="shock-chart", config=NO_MODEBAR),
        chart_hint(has_legend=True),
        html.Div(
            style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "16px", "marginTop": "8px"},
            children=[
                html.Div([html.H4("Hardest hit in 2020"), html.Div(id="shock-hardest")]),
                html.Div([html.H4("Most resilient in 2020"), html.Div(id="shock-resilient")]),
            ],
        ),
        html.P(
            "The two tables list occupations with at least 10,000 employees (their own response, not their "
            "group's) in the selected region, leaving out the \"All Other\" residual categories, which don't describe a "
            "single job. Differences of a few points are within the survey's sampling noise; the large ones are the signal.",
            style={"color": "#888", "fontSize": "12px"},
        ),
    ]
)

app.layout = html.Div(
    style={"fontFamily": "Segoe UI, Arial, sans-serif", "maxWidth": "1100px", "margin": "0 auto", "padding": "24px"},
    children=[
        html.H1("Occupation Employment & Real Wage Tracker", style={"marginBottom": "4px"}),
        html.P(
            f"BLS OEWS data, {FIRST_YEAR}-{LATEST_YEAR} (annual), with CPI-U inflation adjustment and "
            "JOLTS job-market tightness.",
            style={"color": "#666", "marginTop": 0},
        ),
        dcc.Tabs(
            id="main-tabs",
            value="tab-1",
            children=[
                dcc.Tab(label="Occupation Search", value="tab-1", children=[tab1_content]),
                dcc.Tab(label="Regional Comparison (US vs FL)", value="tab-2", children=[tab2_content]),
                dcc.Tab(label="Industry & Sector Analysis", value="tab-3", children=[tab3_content]),
                dcc.Tab(label="2020 Shock & Recovery", value="tab-4", children=[tab4_content]),
            ],
        ),
    ],
)


def kpi_card(label, value, sub=None, color="#222"):
    return html.Div(
        style={
            "flex": "1 1 120px",
            "border": "1px solid #e0e0e0",
            "borderRadius": "8px",
            "padding": "14px 12px",
            "background": "#fafafa",
        },
        children=[
            html.Div(label, style={"fontSize": "13px", "color": "#777"}),
            html.Div(value, style={"fontSize": "22px", "fontWeight": "600", "color": color}),
            html.Div(sub or "", style={"fontSize": "12px", "color": "#999"}),
        ],
    )


@app.callback(
    Output("kpi-row", "children"),
    Output("employment-chart", "figure"),
    Output("wage-chart", "figure"),
    Input("occ-dropdown", "value"),
    Input("region-toggle", "value"),
)
def update_dashboard(occ_code, region):
    sub = df[(df["occ_code"] == occ_code) & (df["region"] == region)].sort_values("year")
    if sub.empty:
        empty_fig = go.Figure()
        empty_fig.update_layout(title="No data for this occupation in the selected region")
        return [kpi_card("No data", "--")], empty_fig, empty_fig

    latest = sub.iloc[-1]
    occ_title = latest["occ_title"]

    emp_growth = latest["employment_change_yoy_pct"]
    wage_growth = latest["wage_growth_yoy_pct"]
    real_wage_growth = latest["real_wage_growth_yoy_pct"]
    contraction = latest["contraction_flag_12mo"]

    kpis = [
        kpi_card("Employment", f"{latest['employment_level']:,.0f}", f"as of {int(latest['year'])}"),
        kpi_card(
            "Employment YoY",
            f"{emp_growth:+.1f}%" if pd.notna(emp_growth) else "n/a",
            color="#c0392b" if pd.notna(emp_growth) and emp_growth < 0 else "#27ae60",
        ),
        kpi_card("Mean wage", f"${latest['wage_annual_mean']:,.0f}", "annual, nominal"),
        kpi_card(
            "Wage YoY (nominal)",
            f"{wage_growth:+.1f}%" if pd.notna(wage_growth) else "n/a",
            color="#c0392b" if pd.notna(wage_growth) and wage_growth < 0 else "#27ae60",
        ),
        kpi_card(
            "Real wage YoY",
            f"{real_wage_growth:+.1f}%" if pd.notna(real_wage_growth) else "n/a",
            "after CPI-U inflation",
            color="#c0392b" if pd.notna(real_wage_growth) and real_wage_growth < 0 else "#27ae60",
        ),
        kpi_card(
            "Contraction flag",
            "YES" if contraction == 1 else "no",
            "employment down >2% YoY",
            color="#c0392b" if contraction == 1 else "#27ae60",
        ),
        kpi_card(
            "Job openings rate",
            f"{latest['jolts_exposure_openings_rate']:.1f}%" if pd.notna(latest["jolts_exposure_openings_rate"]) else "n/a",
            (f"JOLTS, {int(latest['year'])}: quits {latest['jolts_exposure_quits_rate']:.1f}%, "
             f"layoffs {latest['jolts_exposure_layoffs_rate']:.1f}%")
            if pd.notna(latest["jolts_exposure_openings_rate"]) else "",
        ),
    ]

    emp_fig = go.Figure()
    emp_fig.add_trace(go.Scatter(x=sub["year"], y=sub["employment_level"], mode="lines+markers", name="Employment"))
    emp_fig.update_layout(title=f"Employment history — {occ_title} ({region})", yaxis_title="Employment", margin=dict(t=50))

    wage_fig = go.Figure()
    wage_fig.add_trace(go.Scatter(x=sub["year"], y=sub["wage_annual_mean"], mode="lines+markers", name="Nominal"))
    wage_fig.add_trace(go.Scatter(x=sub["year"], y=sub["wage_annual_mean_real"], mode="lines+markers", name=f"Real ({LATEST_YEAR} dollars)"))
    wage_fig.update_layout(
        title=f"Annual mean wage, nominal vs. inflation-adjusted — {occ_title} ({region})",
        yaxis_title="Annual mean wage ($)",
        margin=dict(t=70),
        legend=dict(orientation="h", y=1.08, x=0),
    )

    return kpis, emp_fig, wage_fig


def weighted_quantile(values, weights, q):
    """q-quantile of `values` where each value counts `weights` times (here: employment).

    Uses midpoint cumulative weights, so equal weights reproduce the ordinary
    median. Entries missing a value or a weight are ignored."""
    ok = values.notna() & weights.notna() & (weights > 0)
    v, w = values[ok].to_numpy(dtype=float), weights[ok].to_numpy(dtype=float)
    if len(v) == 0:
        return float("nan")
    order = np.argsort(v, kind="stable")
    v, w = v[order], w[order]
    cumulative = (np.cumsum(w) - 0.5 * w) / w.sum()
    return float(np.interp(q, cumulative, v))


def share_where(series, predicate, weights=None):
    """Percent of entries satisfying `predicate`, among those with a value.

    Unweighted it counts occupations; with `weights` (employment) it is the
    share of workers in occupations that satisfy it."""
    ok = series.notna() if weights is None else series.notna() & weights.notna()
    hit = predicate(series[ok])
    if weights is None:
        return hit.mean() * 100
    return weights[ok][hit].sum() / weights[ok].sum() * 100


def share_negative(series, weights=None):
    """Percent of occupations (or of workers, with weights) whose value is negative."""
    return share_where(series, lambda s: s < 0, weights)


def region_quantile(frame, column, q, by_worker):
    if by_worker:
        return weighted_quantile(frame[column], frame["employment_level"], q)
    return frame[column].quantile(q)


def regional_bar(us_val, fl_val, title, yaxis_title, fmt="{:.1f}"):
    fig = go.Figure(
        go.Bar(
            x=["National (US)", "Florida"],
            y=[us_val, fl_val],
            marker_color=["#4a90d9", "#e07b39"],
            text=[fmt.format(us_val), fmt.format(fl_val)],
            textposition="outside",
        )
    )
    fig.update_layout(
        title=dict(text=title, font=dict(size=13)),
        yaxis_title=yaxis_title,
        margin=dict(t=60, l=50, r=20),
        showlegend=False,
    )
    return fig


def regional_bar_with_iqr(us_frame, fl_frame, column, title, yaxis_title, by_worker=False):
    # A median-only bar for this metric reads as broken when the FL median
    # lands on exactly 0.0% (13 of 744 FL occupations have identical
    # employment in consecutive releases) -- a single point estimate
    # can't show that FL still has real dispersion. A first attempt used a
    # box plot, but occupation-level YoY changes include a handful of small
    # occupations with extreme OEWS-sampling-noise swings (down to -82%, up
    # to +300%) -- full min/max whiskers stretched the axis so far that the
    # actually-informative IQR collapsed into an unreadable sliver near zero.
    # Error bars scoped to just the 25th-75th percentile sidestep that: the
    # extreme tail never enters the range calculation at all, so the chart
    # can't be dominated by it, while still visibly showing real spread
    # around the median bar instead of a lonely flat number.
    us_med, us_q1, us_q3 = (region_quantile(us_frame, column, q, by_worker) for q in (0.5, 0.25, 0.75))
    fl_med, fl_q1, fl_q3 = (region_quantile(fl_frame, column, q, by_worker) for q in (0.5, 0.25, 0.75))
    fig = go.Figure(
        go.Bar(
            x=["National (US)", "Florida"],
            y=[us_med, fl_med],
            marker_color=["#4a90d9", "#e07b39"],
            text=[f"{us_med:+.1f}%", f"{fl_med:+.1f}%"],
            textposition="outside",
            error_y=dict(
                type="data",
                symmetric=False,
                array=[us_q3 - us_med, fl_q3 - fl_med],
                arrayminus=[us_med - us_q1, fl_med - fl_q1],
                visible=True,
                color="#555",
                thickness=1.5,
                width=6,
            ),
        )
    )
    fig.update_layout(
        title=dict(text=title, font=dict(size=13)),
        yaxis_title=yaxis_title,
        margin=dict(t=60, l=50, r=20),
        showlegend=False,
    )
    return fig


@app.callback(
    Output("regional-employment-growth", "figure"),
    Output("regional-wage-level", "figure"),
    Output("regional-wage-decline-prevalence", "figure"),
    Output("regional-real-wage-decline-prevalence", "figure"),
    Output("regional-contraction-prevalence", "figure"),
    Output("regional-job-openings", "figure"),
    Input("main-tabs", "value"),
    Input("regional-weighting", "value"),
)
def update_regional_tab(_tab, weighting="occupation"):
    by_worker = weighting == "worker"
    unit = "worker" if by_worker else "occupation"
    w = (lambda frame: frame["employment_level"]) if by_worker else (lambda frame: None)
    latest_year = df["year"].max()
    latest = df[df["year"] == latest_year]

    # Restrict to occ_codes present in both regions for a fair comparison.
    common_codes = set(latest[latest["region"] == "US"]["occ_code"]) & set(latest[latest["region"] == "FL"]["occ_code"])
    latest = latest[latest["occ_code"].isin(common_codes)]
    us, fl = latest[latest["region"] == "US"], latest[latest["region"] == "FL"]

    emp_growth_fig = regional_bar_with_iqr(
        us, fl, "employment_change_yoy_pct",
        f"Employment growth rate, {latest_year} (median {unit}, IQR whiskers)",
        "% change",
        by_worker,
    )
    wage_level_fig = regional_bar(
        region_quantile(us, "wage_annual_mean", 0.5, by_worker),
        region_quantile(fl, "wage_annual_mean", 0.5, by_worker),
        f"Median annual mean wage, {latest_year} (nominal, median {unit})",
        "$",
        "${:,.0f}",
    )
    share_label = "% of workers" if by_worker else "% of occupations"
    lead = "Workers in occupations with" if by_worker else "Occupations with"
    wage_decline_fig = regional_bar(
        share_negative(us["wage_growth_yoy_pct"], w(us)),
        share_negative(fl["wage_growth_yoy_pct"], w(fl)),
        f"{lead} nominal wage decline (YoY)",
        share_label,
        "{:.0f}%",
    )
    real_wage_decline_fig = regional_bar(
        share_negative(us["real_wage_growth_yoy_pct"], w(us)),
        share_negative(fl["real_wage_growth_yoy_pct"], w(fl)),
        f"{lead} real wage decline (after CPI)" if by_worker else "Occupations with real wage decline (YoY, after CPI)",
        share_label,
        "{:.0f}%",
    )
    contraction_fig = regional_bar(
        share_where(us["contraction_flag_12mo"], lambda s: s == 1, w(us)),
        share_where(fl["contraction_flag_12mo"], lambda s: s == 1, w(fl)),
        "Workers in contracting occupations (>2% YoY)" if by_worker else "Contracting occupations (>2% YoY decline)",
        share_label,
        "{:.0f}%",
    )
    openings_fig = regional_bar(
        us["jolts_region_openings_rate"].iloc[0],
        fl["jolts_region_openings_rate"].iloc[0],
        f"Job openings rate, {latest_year} (JOLTS total nonfarm, annual avg.)",
        "% rate",
        "{:.1f}%",
    )
    return emp_growth_fig, wage_level_fig, wage_decline_fig, real_wage_decline_fig, contraction_fig, openings_fig


def movers_table(rows):
    if rows.empty:
        return html.P("No data.", style={"color": "#999"})
    return dash_table.DataTable(
        columns=[
            {"name": "Occupation", "id": "occ_title"},
            {"name": "Employment", "id": "employment_level", "type": "numeric", "format": {"specifier": ",.0f"}},
            {"name": "Emp. YoY %", "id": "employment_change_yoy_pct", "type": "numeric", "format": {"specifier": "+.1f"}},
        ],
        data=rows.to_dict("records"),
        style_cell={"fontFamily": "Segoe UI, Arial, sans-serif", "fontSize": "13px", "padding": "4px 8px"},
        style_cell_conditional=[{"if": {"column_id": "occ_title"}, "textAlign": "left"}],
        style_as_list_view=True,
    )


def sector_jolts_cards(naics_code, year):
    """JOLTS rates for the sector's closest industry (national; annual average)."""
    row = industry_df[(industry_df["naics"] == naics_code) & (industry_df["year"] == year)]
    row = row.dropna(subset=["jolts_openings_rate"])
    if row.empty:
        return [html.P("JOLTS covers nonfarm industries only, so it has no series for this sector.", style={"color": "#999", "fontSize": "13px"})]
    r = row.iloc[0]
    return [
        kpi_card("Job openings rate", f"{r['jolts_openings_rate']:.1f}%", f"JOLTS, {year} average"),
        kpi_card("Quits rate", f"{r['jolts_quits_rate']:.1f}%", f"JOLTS, {year} average"),
        kpi_card("Layoffs & discharges rate", f"{r['jolts_layoffs_rate']:.1f}%", f"JOLTS, {year} average"),
    ]


if industry_df is not None:

    @app.callback(
        Output("sector-jolts-row", "children"),
        Output("sector-scatter", "figure"),
        Output("sector-top-contracting", "children"),
        Output("sector-top-growth", "children"),
        Input("sector-dropdown", "value"),
        Input("sector-wage-basis", "value"),
    )
    def update_sector_tab(naics_code, wage_basis="real"):
        wage_col = "real_wage_growth_yoy_pct" if wage_basis == "real" else "wage_growth_yoy_pct"
        wage_label = "Real wage growth" if wage_basis == "real" else "Nominal wage growth"
        latest_year = industry_df["year"].max()
        sub = industry_df[(industry_df["naics"] == naics_code) & (industry_df["year"] == latest_year)].dropna(
            subset=["employment_change_yoy_pct", wage_col]
        )
        sector_name = sub["naics_title"].iloc[0] if len(sub) and "naics_title" in sub.columns else naics_code

        scatter_fig = go.Figure(
            go.Scatter(
                x=sub[wage_col],
                y=sub["employment_change_yoy_pct"],
                mode="markers",
                marker=dict(
                    size=(sub["employment_level"].clip(lower=1)) ** 0.5 / 8,
                    color=sub["employment_change_yoy_pct"],
                    colorscale="RdYlGn",
                    showscale=True,
                    line=dict(width=0.5, color="#333"),
                ),
                text=sub["occ_title"],
                hovertemplate="%{text}<br>" + wage_label + ": %{x:.1f}%<br>Employment change: %{y:.1f}%<extra></extra>",
            )
        )
        scatter_fig.add_hline(y=0, line_color="#ccc")
        scatter_fig.add_vline(x=0, line_color="#ccc")
        scatter_fig.update_layout(
            title=dict(text=f"{sector_name} — {wage_label.lower()} vs employment change, {latest_year} (bubble size = employment)", font=dict(size=14)),
            xaxis_title=f"{wage_label} YoY (%)",
            yaxis_title="Employment change YoY (%)",
            margin=dict(t=50),
        )

        # Occupation x industry-sector cells get very sparse for uncommon
        # combos (e.g. a handful of in-house corporate trainers at a
        # manufacturer, reported as "Self-Enrichment Teachers" under NAICS
        # Manufacturing). At n=130 a swing of a few employees reads as a
        # 40%+ move and dominates the top-movers ranking with entries that
        # look nonsensical for the sector, even though the underlying BLS
        # data is genuine. The scatter above still shows every occupation
        # (small ones are naturally de-emphasized by bubble size), but the
        # ranked tables below apply a floor so they highlight real,
        # statistically meaningful movements instead of sampling noise.
        # The floor applies to BOTH years being compared: a floor on the
        # current year alone lets a jump from a tiny base (120 -> 960
        # employees, +700%) straight through. The prior-year count is
        # recovered exactly from the YoY percentage.
        prior_employment = (sub["employment_level"] / (1 + sub["employment_change_yoy_pct"] / 100)).round()
        ranked = sub[
            (sub["employment_level"] >= SECTOR_RANKING_MIN_EMPLOYMENT)
            & (prior_employment >= SECTOR_RANKING_MIN_EMPLOYMENT)
        ]
        cols = ["occ_title", "employment_level", "employment_change_yoy_pct"]
        top_contracting = ranked.nsmallest(10, "employment_change_yoy_pct")[cols]
        top_growth = ranked.nlargest(10, "employment_change_yoy_pct")[cols]

        return sector_jolts_cards(naics_code, latest_year), scatter_fig, movers_table(top_contracting), movers_table(top_growth)


def shock_for(occ_code, region):
    """(response in percentage points, source) for the 2020 tab, or None if unavailable."""
    if shock_response is None or (region, occ_code) not in shock_response.index:
        return None
    row = shock_response.loc[(region, occ_code)]
    if pd.isna(row["response_pct"]):
        return None
    return float(row["response_pct"]), row["source"]


def response_table(rows):
    if rows.empty:
        return html.P("No data.", style={"color": "#999"})
    return dash_table.DataTable(
        columns=[
            {"name": "Occupation", "id": "occ_title"},
            {"name": f"Employment ({LATEST_YEAR})", "id": "employment_level", "type": "numeric", "format": {"specifier": ",.0f"}},
            {"name": "2020 response (pts)", "id": "response_pct", "type": "numeric", "format": {"specifier": "+.1f"}},
        ],
        data=rows.to_dict("records"),
        style_cell={"fontFamily": "Segoe UI, Arial, sans-serif", "fontSize": "13px", "padding": "4px 8px"},
        style_cell_conditional=[{"if": {"column_id": "occ_title"}, "textAlign": "left"}],
        style_as_list_view=True,
    )


def own_response_rows(region):
    """Occupations whose own (not group) 2020 response is reliable, with their latest employment."""
    if shock_response is None:
        return pd.DataFrame(columns=["occ_title", "employment_level", "response_pct"])
    own = shock_response.xs(region, level="region")
    own = own[own["source"] == "occupation"]
    latest = df[(df["region"] == region) & (df["year"] == LATEST_YEAR)].set_index("occ_code")
    rows = own.join(latest[["occ_title", "employment_level"]], how="inner").reset_index()
    # "All Other" residual categories don't describe a single job, so they are left out of the lists
    rows = rows[~rows["occ_title"].str.contains("All Other", regex=False)]
    return rows[["occ_title", "employment_level", "response_pct"]]


@app.callback(
    Output("shock-kpi-row", "children"),
    Output("shock-chart", "figure"),
    Output("shock-hardest", "children"),
    Output("shock-resilient", "children"),
    Input("shock-occ-dropdown", "value"),
    Input("shock-region-toggle", "value"),
)
def update_shock_tab(occ_code, region):
    rows = own_response_rows(region)
    hardest = response_table(rows.nsmallest(10, "response_pct"))
    resilient = response_table(rows.nlargest(10, "response_pct"))

    sub = df[(df["occ_code"] == occ_code) & (df["region"] == region)].sort_values("year").set_index("year")["employment_level"]
    if sub.empty or pd.isna(sub.get(BASE_YEAR)) or pd.isna(sub.get(BASE_YEAR + 1)):
        empty_fig = go.Figure()
        empty_fig.update_layout(title="No 2019/2020 data for this occupation in the selected region")
        return [kpi_card("No data", "--", f"needs employment in the {BASE_YEAR} and {BASE_YEAR + 1} releases")], empty_fig, hardest, resilient

    base = sub[BASE_YEAR]
    change_2020 = (sub[BASE_YEAR + 1] / base - 1) * 100
    latest = sub.get(LATEST_YEAR)
    recovery = (latest / base - 1) * 100 if pd.notna(latest) else None
    shock = shock_for(occ_code, region)

    kpis = [
        kpi_card(f"Employment, {BASE_YEAR} release", f"{base:,.0f}"),
        kpi_card(f"Change in the {BASE_YEAR + 1} release", f"{change_2020:+.1f}%", color=growth_color(change_2020)),
        kpi_card(f"{LATEST_YEAR} vs {BASE_YEAR}", f"{recovery:+.1f}%" if recovery is not None else "n/a",
                 "employment now vs. before the shock", color=growth_color(recovery) if recovery is not None else "#222"),
    ]
    if shock is not None:
        response_pts, source = shock
        note = ("this occupation's own 2020 change vs. its 2016-2019 trend" if source == "occupation"
                else "its occupation group's change (this occupation is too small for its own)")
        kpis.append(kpi_card("2020 response vs trend", f"{response_pts:+.1f} pts", note, color=growth_color(response_pts)))

    index = sub.dropna() / base * 100
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=index.index, y=index.values, mode="lines+markers", name="This occupation"))
    totals, n_codes = BALANCED_TOTALS[region]
    fig.add_trace(go.Scatter(x=totals.index, y=totals.values, mode="lines", name=f"All occupations present in every release ({n_codes})",
                             line=dict(dash="dot", color="#888")))
    fig.add_hline(y=100, line_color="#ccc")
    fig.add_vline(x=BASE_YEAR + 1, line_color="#e07b39", line_dash="dash")
    fig.update_layout(
        title=f"Employment, {BASE_YEAR} release = 100 — {df.loc[df['occ_code'] == occ_code, 'occ_title'].iloc[0]} ({region})",
        yaxis_title=f"Index ({BASE_YEAR} = 100)",
        margin=dict(t=70),
        legend=dict(orientation="h", y=1.1, x=0),
    )
    return kpis, fig, hardest, resilient


def growth_color(growth_pct):
    return "#c0392b" if growth_pct < 0 else "#27ae60"


if __name__ == "__main__":
    # Render (and most PaaS) inject PORT; gunicorn is used in production
    # instead of this block anyway (see Procfile), but keep local `python
    # app/app.py` working the same way it always has.
    import os

    port = int(os.environ.get("PORT", 8051))
    debug = os.environ.get("DASH_DEBUG", "true").lower() == "true"
    app.run(debug=debug, host="0.0.0.0", port=port)

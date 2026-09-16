"""
Occupation Displacement & Wage Compression Forecaster -- Dash app.

Tab 1 (Occupation Search & Risk Dashboard) is implemented here. It runs on
OEWS-derived features alone; real-wage and displacement-risk-score features
that depend on CPI/JOLTS are added once those fetches complete (see
src/fetch_cpi.py, src/fetch_jolts.py) -- until then the KPI cards and charts
show nominal figures only, flagged as such in the UI.
"""
from pathlib import Path

import dash
import lightgbm as lgb
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, dash_table, dcc, html

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
DATA_PATH = PROCESSED_DIR / "oews_occupation_features.csv"
INDUSTRY_DATA_PATH = PROCESSED_DIR / "oews_industry_features.csv"
MODEL_PATH = PROCESSED_DIR / "employment_forecast_model.txt"

# Below this, an occupation's employment count within a single industry
# sector is small enough that YoY % swings are dominated by OEWS sampling
# noise rather than a real trend -- see the ranking note in update_sector_tab.
SECTOR_RANKING_MIN_EMPLOYMENT = 500

# Must match src/train_model.py exactly (feature order/dtypes affect a saved
# LightGBM booster's categorical split lookups).
MODEL_FEATURE_COLS = [
    "employment_level",
    "wage_annual_mean",
    "wage_annual_median",
    "employment_change_yoy_pct",
    "wage_growth_yoy_pct",
    "employment_change_5yr_cagr",
    "wage_growth_5yr_cagr",
    "contraction_flag_12mo",
    "region",
    "occ_major_group",
]

df = pd.read_csv(DATA_PATH)
# Industry-sector extraction (Tab 3) re-parses the same large archives on a
# separate schedule and may not have finished yet -- degrade gracefully.
industry_df = pd.read_csv(INDUSTRY_DATA_PATH, dtype={"naics": str}) if INDUSTRY_DATA_PATH.exists() else None
forecast_model = lgb.Booster(model_file=str(MODEL_PATH)) if MODEL_PATH.exists() else None
occ_options = (
    df[["occ_code", "occ_title"]]
    .drop_duplicates()
    .sort_values("occ_title")
    .assign(label=lambda d: d["occ_title"] + " (" + d["occ_code"] + ")")
)
DEFAULT_OCC = df.loc[df["employment_level"].idxmax(), "occ_code"]

app = dash.Dash(__name__)
app.title = "Occupation Displacement & Wage Compression Forecaster"
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
        html.Div(id="kpi-row", style={"display": "flex", "gap": "16px", "marginBottom": "24px"}),
        dcc.Graph(id="employment-chart"),
        dcc.Graph(id="wage-chart"),
    ]
)

tab2_content = html.Div(
    style={"marginTop": "20px"},
    children=[
        html.P(
            "Occupations present in both US and FL for the most recent year (2024). "
            "'Wage decline prevalence' is nominal (YoY), not inflation-adjusted -- real-wage "
            "compression prevalence will replace it once CPI data lands.",
            style={"color": "#666"},
        ),
        html.Div(
            style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "16px"},
            children=[
                dcc.Graph(id="regional-employment-growth"),
                dcc.Graph(id="regional-wage-level"),
                dcc.Graph(id="regional-wage-decline-prevalence"),
                dcc.Graph(id="regional-contraction-prevalence"),
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
                style={"marginTop": "20px", "marginBottom": "16px", "maxWidth": "500px"},
            ),
            dcc.Graph(id="sector-scatter"),
            html.Div(
                style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "16px", "marginTop": "8px"},
                children=[
                    html.Div([html.H4("Top 10 at-risk (most contracting)"), html.Div(id="sector-top-risk")]),
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

SLIDER_MARKS = lambda lo, hi, step: {i: f"{i}%" for i in range(lo, hi + 1, step)}
# dcc.Slider centers each mark's label on its tick position, so the leftmost
# mark (e.g. "-20%") has its label centered right at the track's start --
# roughly half the text, including the leading minus sign, renders to the
# left of the slider's own box and gets clipped with no room to spare.
# Padding the slider inside its own wrapper (rather than relying on Dash's
# internal slider CSS class names, which could change) gives that overflow
# somewhere real to go.
SLIDER_WRAPPER_STYLE = {"padding": "0 24px"}

tab4_content = html.Div(
    children=[
        html.P(
            "Illustrative what-if tool: applies transparent, documented linear "
            "adjustments on top of the trained model's baseline forecast. These are "
            "simplifying assumptions, not fitted causal elasticities (no macro model "
            "exists here) -- treat results as directional, not precise.",
            style={"color": "#666", "marginTop": "20px"},
        ),
        dcc.Dropdown(
            id="scenario-occ-dropdown",
            options=[{"label": row.label, "value": row.occ_code} for row in occ_options.itertuples()],
            value=DEFAULT_OCC,
            style={"maxWidth": "500px", "marginBottom": "12px"},
        ),
        dcc.RadioItems(
            id="scenario-region-toggle",
            options=[{"label": " National (US)", "value": "US"}, {"label": " Florida", "value": "FL"}],
            value="US",
            inline=True,
            style={"marginBottom": "20px"},
        ),
        html.Div(
            style={"display": "grid", "gridTemplateColumns": "1fr", "gap": "20px", "maxWidth": "600px"},
            children=[
                html.Div([
                    html.Label("Economic contraction severity (GDP growth impact)"),
                    html.Div(
                        dcc.Slider(id="gdp-slider", min=-20, max=5, step=1, value=0, marks=SLIDER_MARKS(-20, 5, 5)),
                        style=SLIDER_WRAPPER_STYLE,
                    ),
                ]),
                html.Div([
                    html.Label("Automation impact (additional employment headwind)"),
                    html.Div(
                        dcc.Slider(id="automation-slider", min=0, max=20, step=1, value=0, marks=SLIDER_MARKS(0, 20, 5)),
                        style=SLIDER_WRAPPER_STYLE,
                    ),
                ]),
                html.Div([
                    html.Label("Immigration policy effect (labor supply adjustment)"),
                    html.Div(
                        dcc.Slider(id="immigration-slider", min=-10, max=10, step=1, value=0, marks=SLIDER_MARKS(-10, 10, 5)),
                        style=SLIDER_WRAPPER_STYLE,
                    ),
                ]),
            ],
        ),
        html.Div(id="scenario-kpi-row", style={"display": "flex", "gap": "16px", "margin": "24px 0"}),
        dcc.Graph(id="scenario-chart"),
        html.P(id="scenario-summary", style={"fontStyle": "italic", "color": "#444"}),
    ]
)

app.layout = html.Div(
    style={"fontFamily": "Segoe UI, Arial, sans-serif", "maxWidth": "1100px", "margin": "0 auto", "padding": "24px"},
    children=[
        html.H1("Occupation Displacement & Wage Compression Forecaster", style={"marginBottom": "4px"}),
        html.P(
            "BLS OEWS data, 2015-2024 (annual). Real-wage and displacement-risk "
            "scoring pending CPI/JOLTS integration.",
            style={"color": "#666", "marginTop": 0},
        ),
        dcc.Tabs(
            id="main-tabs",
            value="tab-1",
            children=[
                dcc.Tab(label="Occupation Search & Risk", value="tab-1", children=[tab1_content]),
                dcc.Tab(label="Regional Comparison (US vs FL)", value="tab-2", children=[tab2_content]),
                dcc.Tab(label="Industry & Sector Analysis", value="tab-3", children=[tab3_content]),
                dcc.Tab(label="Predictive Scenarios", value="tab-4", children=[tab4_content]),
            ],
        ),
    ],
)


def kpi_card(label, value, sub=None, color="#222"):
    return html.Div(
        style={
            "flex": "1",
            "border": "1px solid #e0e0e0",
            "borderRadius": "8px",
            "padding": "14px 16px",
            "background": "#fafafa",
        },
        children=[
            html.Div(label, style={"fontSize": "13px", "color": "#777"}),
            html.Div(value, style={"fontSize": "24px", "fontWeight": "600", "color": color}),
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
    contraction = latest["contraction_flag_12mo"]

    kpis = [
        kpi_card("Employment", f"{latest['employment_level']:,.0f}", f"as of {int(latest['year'])}"),
        kpi_card(
            "Employment YoY",
            f"{emp_growth:+.1f}%" if pd.notna(emp_growth) else "n/a",
            color="#c0392b" if pd.notna(emp_growth) and emp_growth < 0 else "#27ae60",
        ),
        kpi_card("Wage (annual mean)", f"${latest['wage_annual_mean']:,.0f}", "nominal, not inflation-adjusted"),
        kpi_card(
            "Wage YoY",
            f"{wage_growth:+.1f}%" if pd.notna(wage_growth) else "n/a",
            color="#c0392b" if pd.notna(wage_growth) and wage_growth < 0 else "#27ae60",
        ),
        kpi_card(
            "Contraction flag",
            "YES" if contraction == 1 else "no",
            "employment down >2% YoY",
            color="#c0392b" if contraction == 1 else "#27ae60",
        ),
    ]

    emp_fig = go.Figure()
    emp_fig.add_trace(go.Scatter(x=sub["year"], y=sub["employment_level"], mode="lines+markers", name="Employment"))
    emp_fig.update_layout(title=f"Employment history — {occ_title} ({region})", yaxis_title="Employment", margin=dict(t=50))

    wage_fig = go.Figure()
    wage_fig.add_trace(go.Scatter(x=sub["year"], y=sub["wage_annual_mean"], mode="lines+markers", name="Nominal annual mean wage"))
    wage_fig.update_layout(
        title=f"Wage history (nominal) — {occ_title} ({region})",
        yaxis_title="Annual mean wage ($)",
        margin=dict(t=50),
    )

    return kpis, emp_fig, wage_fig


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
    fig.update_layout(title=title, yaxis_title=yaxis_title, margin=dict(t=50), showlegend=False)
    return fig


@app.callback(
    Output("regional-employment-growth", "figure"),
    Output("regional-wage-level", "figure"),
    Output("regional-wage-decline-prevalence", "figure"),
    Output("regional-contraction-prevalence", "figure"),
    Input("main-tabs", "value"),
)
def update_regional_tab(_tab):
    latest_year = df["year"].max()
    latest = df[df["year"] == latest_year]

    # Restrict to occ_codes present in both regions for a fair comparison.
    common_codes = set(latest[latest["region"] == "US"]["occ_code"]) & set(latest[latest["region"] == "FL"]["occ_code"])
    latest = latest[latest["occ_code"].isin(common_codes)]
    us, fl = latest[latest["region"] == "US"], latest[latest["region"] == "FL"]

    emp_growth_fig = regional_bar(
        us["employment_change_yoy_pct"].median(),
        fl["employment_change_yoy_pct"].median(),
        f"Median employment growth rate, {latest_year} (YoY)",
        "%",
        "{:+.1f}%",
    )
    wage_level_fig = regional_bar(
        us["wage_annual_mean"].median(),
        fl["wage_annual_mean"].median(),
        f"Median annual mean wage, {latest_year} (nominal)",
        "$",
        "${:,.0f}",
    )
    wage_decline_fig = regional_bar(
        (us["wage_growth_yoy_pct"] < 0).mean() * 100,
        (fl["wage_growth_yoy_pct"] < 0).mean() * 100,
        "Share of occupations with nominal wage decline (YoY)",
        "% of occupations",
        "{:.0f}%",
    )
    contraction_fig = regional_bar(
        (us["contraction_flag_12mo"] == 1).mean() * 100,
        (fl["contraction_flag_12mo"] == 1).mean() * 100,
        "Share of occupations flagged as contracting (>2% YoY decline)",
        "% of occupations",
        "{:.0f}%",
    )
    return emp_growth_fig, wage_level_fig, wage_decline_fig, contraction_fig


def risk_table(rows):
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
        style_as_list_view=True,
    )


if industry_df is not None:

    @app.callback(
        Output("sector-scatter", "figure"),
        Output("sector-top-risk", "children"),
        Output("sector-top-growth", "children"),
        Input("sector-dropdown", "value"),
    )
    def update_sector_tab(naics_code):
        latest_year = industry_df["year"].max()
        sub = industry_df[(industry_df["naics"] == naics_code) & (industry_df["year"] == latest_year)].dropna(
            subset=["employment_change_yoy_pct", "wage_growth_yoy_pct"]
        )
        sector_name = sub["naics_title"].iloc[0] if len(sub) and "naics_title" in sub.columns else naics_code

        scatter_fig = go.Figure(
            go.Scatter(
                x=sub["wage_growth_yoy_pct"],
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
                hovertemplate="%{text}<br>Wage growth: %{x:.1f}%<br>Employment change: %{y:.1f}%<extra></extra>",
            )
        )
        scatter_fig.add_hline(y=0, line_color="#ccc")
        scatter_fig.add_vline(x=0, line_color="#ccc")
        scatter_fig.update_layout(
            title=f"{sector_name} — wage growth vs employment change, {latest_year} (bubble size = employment)",
            xaxis_title="Wage growth YoY (%)",
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
        ranked = sub[sub["employment_level"] >= SECTOR_RANKING_MIN_EMPLOYMENT]
        cols = ["occ_title", "employment_level", "employment_change_yoy_pct"]
        top_risk = ranked.nsmallest(10, "employment_change_yoy_pct")[cols]
        top_growth = ranked.nlargest(10, "employment_change_yoy_pct")[cols]

        return scatter_fig, risk_table(top_risk), risk_table(top_growth)


# Illustrative assumption, not a fitted causal elasticity: a 10-point GDP
# growth swing is assumed to pass through to occupation employment growth at
# 60% strength. No macro model exists in this project to estimate real
# sector-specific elasticities -- this is a transparent placeholder so the
# scenario tool is directionally useful, not a precise forecast.
GDP_PASSTHROUGH = 0.6


def predict_baseline(occ_code, region):
    sub = df[(df["occ_code"] == occ_code) & (df["region"] == region)].sort_values("year")
    if sub.empty or forecast_model is None:
        return None
    latest = sub.iloc[-1].copy()
    row = pd.DataFrame([{
        "employment_level": latest["employment_level"],
        "wage_annual_mean": latest["wage_annual_mean"],
        "wage_annual_median": latest["wage_annual_median"],
        "employment_change_yoy_pct": latest["employment_change_yoy_pct"],
        "wage_growth_yoy_pct": latest["wage_growth_yoy_pct"],
        "employment_change_5yr_cagr": latest["employment_change_5yr_cagr"],
        "wage_growth_5yr_cagr": latest["wage_growth_5yr_cagr"],
        "contraction_flag_12mo": int(latest["contraction_flag_12mo"]) if pd.notna(latest["contraction_flag_12mo"]) else 0,
        "region": str(region),
        "occ_major_group": str(occ_code)[:2],
    }])
    row["region"] = row["region"].astype("category")
    row["occ_major_group"] = row["occ_major_group"].astype("category")
    # Model predicts log growth ratio, not absolute level -- see src/train_model.py
    # for why (absolute-level regression badly mispredicts the handful of
    # multi-million-employee occupations, which are extremely sparse in training data).
    predicted_log_growth = float(forecast_model.predict(row[MODEL_FEATURE_COLS])[0])
    baseline_employment_next = float(latest["employment_level"] * np.exp(predicted_log_growth))
    baseline_growth_pct = (baseline_employment_next / latest["employment_level"] - 1) * 100
    return {
        "occ_title": latest["occ_title"],
        "current_employment": latest["employment_level"],
        "current_year": int(latest["year"]),
        "baseline_employment_next": baseline_employment_next,
        "baseline_growth_pct": baseline_growth_pct,
    }


def risk_level(growth_pct):
    if growth_pct < -10:
        return "HIGH", "#c0392b"
    if growth_pct < -2:
        return "MODERATE", "#e08e0b"
    return "LOW", "#27ae60"


@app.callback(
    Output("scenario-kpi-row", "children"),
    Output("scenario-chart", "figure"),
    Output("scenario-summary", "children"),
    Input("scenario-occ-dropdown", "value"),
    Input("scenario-region-toggle", "value"),
    Input("gdp-slider", "value"),
    Input("automation-slider", "value"),
    Input("immigration-slider", "value"),
)
def update_scenario_tab(occ_code, region, gdp_pct, automation_pct, immigration_pct):
    baseline = predict_baseline(occ_code, region)
    if baseline is None:
        empty_fig = go.Figure()
        empty_fig.update_layout(title="No data / model unavailable for this selection")
        return [], empty_fig, "No data available."

    scenario_growth_pct = (
        baseline["baseline_growth_pct"]
        + gdp_pct * GDP_PASSTHROUGH
        - automation_pct
        + immigration_pct
    )
    scenario_employment = baseline["current_employment"] * (1 + scenario_growth_pct / 100)

    base_level, base_color = risk_level(baseline["baseline_growth_pct"])
    scenario_level, scenario_color = risk_level(scenario_growth_pct)

    kpis = [
        kpi_card("Baseline forecast growth", f"{baseline['baseline_growth_pct']:+.1f}%", "model prediction, no scenario"),
        kpi_card("Scenario forecast growth", f"{scenario_growth_pct:+.1f}%", "with sliders applied", color=scenario_color),
        kpi_card("Baseline risk level", base_level, color=base_color),
        kpi_card("Scenario risk level", scenario_level, color=scenario_color),
    ]

    fig = go.Figure(
        go.Bar(
            x=["Current", "Baseline forecast", "Scenario forecast"],
            y=[baseline["current_employment"], baseline["baseline_employment_next"], scenario_employment],
            marker_color=["#999", "#4a90d9", "#e07b39"],
            text=[f"{v:,.0f}" for v in [baseline["current_employment"], baseline["baseline_employment_next"], scenario_employment]],
            textposition="outside",
        )
    )
    fig.update_layout(
        title=f"Employment: current vs baseline vs scenario — {baseline['occ_title']} ({region})",
        yaxis_title="Employment",
        margin=dict(t=50),
    )

    summary = (
        f"Under this scenario (GDP {gdp_pct:+d}%, automation headwind {automation_pct}pp, "
        f"immigration effect {immigration_pct:+d}%), {baseline['occ_title']} risk moves from "
        f"{base_level} to {scenario_level}."
        if base_level != scenario_level
        else f"Under this scenario, {baseline['occ_title']} risk stays {scenario_level}."
    )

    return kpis, fig, summary


if __name__ == "__main__":
    # Render (and most PaaS) inject PORT; gunicorn is used in production
    # instead of this block anyway (see Procfile), but keep local `python
    # app/app.py` working the same way it always has.
    import os

    port = int(os.environ.get("PORT", 8051))
    debug = os.environ.get("DASH_DEBUG", "true").lower() == "true"
    app.run(debug=debug, host="0.0.0.0", port=port)

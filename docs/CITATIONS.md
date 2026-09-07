# Data Sources & References

## BLS Occupational Employment and Wage Statistics (OEWS)
- Program overview: https://www.bls.gov/oes/
- Annual "All Data" archives (used by `data/fetch_oews.py`): https://www.bls.gov/oes/tables.htm
- Technical/series-format documentation: https://download.bls.gov/pub/time.series/oe/oe.txt
- Standard Occupational Classification (SOC) system: https://www.bls.gov/soc/

## BLS Job Openings and Labor Turnover Survey (JOLTS)
- Program overview: https://www.bls.gov/jlt/
- Technical/series-format documentation: https://download.bls.gov/pub/time.series/jt/jt.txt
- North American Industry Classification System (NAICS): https://www.bls.gov/bls/naics.htm

## BLS Consumer Price Index (CPI)
- Program overview: https://www.bls.gov/cpi/
- Series used: `CUUR0000SA0` (CPI-U, All Urban Consumers, US city average, all items)

## BLS Public Data API
- Developer docs: https://www.bls.gov/developers/
- API v2 signature reference: https://www.bls.gov/developers/api_signature_v2.htm
- Registration (raises the unregistered 25-queries/day limit to 500/day): https://data.bls.gov/registrationEngine/

## Tooling
- LightGBM: https://lightgbm.readthedocs.io/
- Plotly Dash: https://dash.plotly.com/

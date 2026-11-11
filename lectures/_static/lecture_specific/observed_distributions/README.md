# Data for the observed distributions lecture

The lecture reads every file from a URL. The copies here are backups, in case a source
changes or disappears.

## Files published with this book

The lecture reads these from this repository on GitHub.

`gdp_by_country_2025.csv` holds GDP in current US dollars for 2025, one row per country.

- **Source:** World Bank, World Development Indicators, series NY.GDP.MKTP.CD, retrieved
  8 November 2026 from
  https://api.worldbank.org/v2/country/all/indicator/NY.GDP.MKTP.CD?date=2025&format=json
  (database last updated 8 October 2026). Licensed under CC BY 4.0.
- **Changes:** regional and income-group aggregates were removed, using the region field of
  https://api.worldbank.org/v2/country, leaving 186 economies with a 2025 value. Country
  names and codes are the World Bank's own.
- **Gaps:** 31 economies had no 2025 value at retrieval, mostly small territories but
  also Afghanistan, Cuba and Eritrea.

`amazon_monthly_close.csv` holds the monthly closing price of Amazon shares (AMZN), in
US dollars, from January 2000 to December 2023, one row per month.

- **Source:** Yahoo Finance, retrieved 10 November 2026 with `yfinance` 1.8.0 via
  `yf.download('AMZN', '2000-1-1', '2024-1-1', interval='1mo')`, column `Close`.
- **Changes:** prices are adjusted for stock splits (yfinance's default `auto_adjust=True`)
  and rounded to six decimal places. Stored here because the lecture should not depend on
  a live download from Yahoo Finance.

## Files from QuantEcon's data repository

The lecture reads these from https://github.com/QuantEcon/data-lectures, where each has a
manifest recording its provenance. The copies here are byte-identical backups, taken on
10 November 2026 at data-lectures commit `43f32e0e`.

- `ames_house_prices.csv`: residential house sales in Ames, Iowa, 2006-2010, from the Ames
  Housing data of De Cock (2011), Journal of Statistics Education. Also used in the
  bivariate distributions lecture.
- `japan_deaths_by_age.csv`: deaths in Japan by single year of age, 2023, from the United
  Nations World Population Prospects 2024. Licensed under CC BY 3.0 IGO.
- `us_household_net_worth_2022.csv`: US household net worth with survey weights, from the
  Federal Reserve's Survey of Consumer Finances 2022. Public domain.
- `epl_match_goals.csv`: full-time scores in the English Premier League, 2015-16 to
  2024-25, from openfootball/football.json. Public domain.
- `japan_earthquakes.csv`: earthquakes of magnitude 5 and above in the region around
  Japan, 2000-2024, from the US Geological Survey's ANSS Comprehensive Earthquake Catalog.
  Public domain.

The heights data used in this lecture are stored with the probability distributions
lecture, in `../prob_dist/`.

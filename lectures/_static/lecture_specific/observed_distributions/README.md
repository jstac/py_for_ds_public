# Data for the observed distributions lecture

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
  and rounded to six decimal places. Stored here so that the lecture does not depend on a
  live download.

# Data for the working with data lecture

The lecture reads every data set from its source online. The copies here are backups, in
case a source changes or disappears.

`gdp_population_by_country_2000.csv` is the lecture's running example, with population,
total GDP and two expenditure shares for seven countries in the year 2000.

- **Source:** a byte-identical copy of the file of the same name in
  https://github.com/QuantEcon/data-lectures, whose manifest records its provenance. It
  is cut from `test_pwt.csv` in that repo.
- **Caveat:** the values match no downloadable release of the Penn World Tables, so they
  are for illustration only.

`unemployment_rate_us.csv` is the monthly US unemployment rate, series UNRATE from FRED,
Federal Reserve Bank of St. Louis, retrieved 10 November 2026 from
https://fred.stlouisfed.org/graph/fredgraph.csv?id=UNRATE, unchanged. It runs from
January 1948 to October 2026. The underlying data are from the US Bureau of Labor
Statistics.

`government_debt_usa_aus.csv` is central government debt as a percent of GDP for the
United States and Australia, 2005-2015, World Bank series GC.DOD.TOTL.GD.ZS, retrieved
10 November 2026 from
https://api.worldbank.org/v2/country/USA;AUS/indicator/GC.DOD.TOTL.GD.ZS?date=2005:2015&format=json
(database last updated 8 October 2026). It is reshaped into one row per economy and year.
Licensed under CC BY 4.0. The lecture fetches the same series with `wbgapi`.

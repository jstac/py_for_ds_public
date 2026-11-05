# Data for the working with data lecture

`gdp_per_capita.csv` holds GDP per capita for China, Japan and South Korea, in
international dollars at 2021 prices, adjusted for inflation and differences in living
costs.

- **Source:** Our World in Data,
  https://ourworldindata.org/grapher/gdp-per-capita-worldbank, based on data from
  Eurostat, the OECD, the IMF and the World Bank. Licensed under CC BY 4.0.
- **Changes:** columns were renamed and shortened, and values were rounded to the nearest
  dollar.

`gdp_population_by_country_2000.csv` is the lecture's running example, with population,
total GDP and two expenditure shares for seven countries in the year 2000.

- **Source:** a byte-identical copy of the file of the same name in
  https://github.com/QuantEcon/data-lectures, whose manifest records its provenance. It
  is cut from `test_pwt.csv` in that repo.
- **Caveat:** the values match no downloadable release of the Penn World Tables, so they
  are for illustration only.

`pandas_vs_rest.png`, `pandas_share_prices.png` and `pandas_indices_pctchange.png` are
copied from `lectures/_static/lecture_specific/pandas/` in
QuantEcon/lecture-python-programming.

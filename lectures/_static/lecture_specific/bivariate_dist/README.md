# Data for the bivariate distributions lecture

The lecture reads every file from a URL. The copies here are backups, in case a source
changes or disappears.

## Files published with this book

The lecture reads this file from this repository on GitHub.

`state_votes_life_expectancy.csv` has one row for each of the 50 states and the District
of Columbia, with Donald Trump's share of the 2024 presidential vote and life expectancy
at birth in 2021.

- **Vote share:** Federal Election Commission, Official 2024 Presidential General Election
  Results, https://www.fec.gov/resources/cms-content/documents/2024presgeresults.pdf. The
  share is Trump's votes divided by total votes, in percent, rounded to two decimal places.
  The state figures sum to the document's totals of 77,302,580 and 155,238,302.
- **Life expectancy:** National Center for Health Statistics, U.S. State Life Expectancy by
  Sex, 2021, https://data.cdc.gov/d/it4f-frdc, total population. Public domain. This was
  the latest year available as a data file, and it was a pandemic year. Values for 2022
  are in NVSR 74-12.

## Files from QuantEcon's data repository

The lecture reads this file from https://github.com/QuantEcon/data-lectures, where it has
a manifest recording its provenance. The copy here is a byte-identical backup, taken on
10 November 2026 at data-lectures commit `43f32e0e`.

- `life-expectancy-vs-gdp-per-capita.csv`: the "Life expectancy vs. GDP per capita"
  grapher export from Our World in Data,
  https://ourworldindata.org/grapher/life-expectancy-vs-gdp-per-capita, stored verbatim.

The Ames house price data used in this lecture are stored with the observed distributions
lecture, in `../observed_distributions/`.

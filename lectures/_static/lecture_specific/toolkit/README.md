# Data for the toolkit lecture

`japan_prefectures.geojson` holds simplified boundaries and 2020 population for
Japan's 47 prefectures.

- **Boundaries:** Natural Earth, 1:10m admin 1 states and provinces, which is in the
  public domain (https://www.naturalearthdata.com/about/terms-of-use/). Polygons were
  simplified with a tolerance of 0.015 degrees, very small islands were dropped, and
  coordinates were rounded to three decimal places.
- **Population:** 2020 Population Census of Japan, Statistics Bureau of Japan, taken by
  prefecture from Wikipedia's "List of Japanese prefectures by population". The values sum
  to 126,146,099, which matches the Bureau's published total of 126,146 thousand. The
  individual prefecture values have not been checked against the Bureau's own tables.

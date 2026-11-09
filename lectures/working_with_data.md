---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.7
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Working with Data

```{index} single: Python; Pandas
```

[pandas](https://pandas.pydata.org/) is the standard library for working with tables of data in Python.

AI tools are trained on masses of pandas code, and most of what you do with AI and data in Python will involve substantial amounts of it.

As such, we at least need to understand the basics.

Just as [NumPy](https://numpy.org/) provides the basic array data type plus core array operations, pandas

1. defines fundamental structures for working with data and
1. endows them with methods that facilitate operations such as
    * reading in data
    * adjusting indices
    * working with dates and time series
    * sorting, grouping, re-ordering and general data munging [^mung]
    * dealing with missing values, etc., etc.

```{note}
pandas does not contain sophisticated statistical routines.

Rather, it is a tool for simple data analysis and a foundation for many other packages that work with pandas DataFrames, such as [statsmodels](https://www.statsmodels.org/) and [scikit-learn](https://scikit-learn.org/).
```

This lecture gives a fast-paced introduction to pandas.

In addition to what’s in Anaconda, this lecture needs the following library:

```{code-cell} ipython3
:tags: [hide-output]

!pip install --upgrade wbgapi
```

We use the following imports.

```{code-cell} ipython3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
```

## Series

```{index} single: Pandas; Series
```

Two important data types defined by pandas are  `Series` and `DataFrame`.

You can think of a `Series` as a "column" of data, such as a collection of observations on a single variable.

A `DataFrame` is like a `Series` but it has multiple columns -- analogous to a spreadsheet.

Let's start with Series.

We begin by creating a series from a Python dictionary.

The keys are the ticker symbols of four listed companies, and the values are made-up daily returns on their shares, in percent.

```{code-cell} ipython3
returns = {'AMZN': 1.2, 'AAPL': -0.7, 'MSFT': 0.4, 'GOOG': -1.5}
s = pd.Series(returns, name='daily returns')
s
```

The keys have become the *index* of the series.

Let's compare this with a NumPy array containing the same numbers.

```{code-cell} ipython3
a = s.to_numpy()
a
```

The figure below shows the two side by side.

```{code-cell} ipython3
---
mystnb:
  image:
    width: 60%
  figure:
    caption: A NumPy array and a pandas Series
    name: array_vs_series
tags: [hide-input]
---
value_color, index_color = '#cde2fb', '#d3f0e4'
edge_color, muted_color = '#7a8794', '#5f6b76'
n = len(s)
values = [f'{x:.1f}' for x in s]


def draw_cells(ax, x, width, entries, color, header):
    "Draw a column of cells starting at x, with a header above it."
    for i, entry in enumerate(entries):
        ax.add_patch(Rectangle((x, n - 1 - i), width, 1, facecolor=color,
                               edgecolor=edge_color))
        ax.text(x + width / 2, n - 0.5 - i, entry, ha='center', va='center')
    ax.text(x + width / 2, n + 0.15, header, ha='center', color=muted_color)


fig, ax = plt.subplots(figsize=(8, 3.4))

# The array: positions are implicit, so they are not drawn as cells
ax.text(1.5, n + 0.9, 'NumPy array', ha='center', weight='bold')
for i in range(n):
    ax.text(0.65, n - 0.5 - i, str(i), ha='center', va='center', color=muted_color)
ax.text(0.65, n + 0.15, 'position', ha='center', color=muted_color)
draw_cells(ax, 1.3, 1.7, values, value_color, 'value')

# The series: the index is stored next to the values
ax.text(6.75, n + 0.9, 'pandas Series', ha='center', weight='bold')
draw_cells(ax, 4.6, 1.6, s.index, index_color, 'index')
draw_cells(ax, 6.2, 2.7, values, value_color, s.name)

ax.set_xlim(0, 9)
ax.set_ylim(-0.1, n + 1.4)
ax.set_aspect('equal')
ax.axis('off')
plt.show()
```

We find a number in a NumPy array by its position.

```{code-cell} ipython3
print(a[0])
```

In a pandas series, we typically use its label.

```{code-cell} ipython3
print(s['AMZN'])
```

`Series` are built on top of NumPy arrays and support many similar operations.

For example, we can convert the returns from percentages to fractions.

```{code-cell} ipython3
s / 100
```

We can also use NumPy functions

```{code-cell} ipython3
np.abs(s)
```

But `Series` provide more than NumPy arrays.

For one thing, they have some additional (statistically oriented) methods

```{code-cell} ipython3
s.describe()
```

And because of the index, `Series` are like Python dictionaries
(with the restriction that the items in the dictionary all have the same
type---in this case, floats).

In fact, you can use much of the same syntax as Python dictionaries

```{code-cell} ipython3
s['AMZN'] = 0
s
```

```{code-cell} ipython3
'AAPL' in s
```

## DataFrames

```{index} single: Pandas; DataFrames
```

While a `Series` is a single column of data, a `DataFrame` is several columns, one for each variable.

It is used to manage and analyze data that are naturally viewed as rows and columns. 

Let's look at an example that reads data from the CSV file `gdp_population_by_country_2000.csv`.

It holds data on seven countries in the year 2000, in the layout of the [Penn World Tables](https://www.rug.nl/ggdc/productivity/pwt/).

The numbers are for illustration only, since they do not match any downloadable release of those tables.

The dataset contains the following indicators 

| Variable Name | Description |
| :-: | :-: |
| POP | Population (in thousands) |
| tcgdp | GDP (millions of international dollars) |
| cc | Consumption Share of GDP Per Capita (%) |
| cg | Government Consumption Share of GDP Per Capita (%) |


We'll read this in from a URL using the `pandas` function `read_csv`.

```{code-cell} ipython3
url = ('https://github.com/QuantEcon/data-lectures/raw/main/lectures/'
       'gdp_population_by_country_2000.csv')
df = pd.read_csv(url)
type(df)
```

Here's the content of the file

```{code-cell} ipython3
df
```

The leftmost column, `0, 1, ..., 6`, is the index of the DataFrame.

It is not in the file: pandas created it because the file does not supply one, and you will see this often when reading CSV files.

The country names make a more useful index, so let's use them instead.

```{code-cell} ipython3
df = df.set_index('country')
df
```

The country names are now the index, and `country` is no longer one of the columns.

The figure below shows the parts of this DataFrame.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: The parts of a DataFrame
    name: dataframe_parts
tags: [hide-input]
---
column_color = '#e8e3f4'
n = len(df)
index_width, width, gap = 3.1, 2.6, 0.35
x_index = 2.0
x_values = x_index + index_width + gap
x_end = x_values + 4 * width + 3 * gap
x_center = (x_values + x_end) / 2
header_y = n + 0.2


def draw_header(x, cell_width, name, color):
    "Draw a cell holding a name, above a column of cells."
    ax.add_patch(Rectangle((x, header_y), cell_width, 1, facecolor=color,
                           edgecolor=edge_color))
    ax.text(x + cell_width / 2, header_y + 0.5, name, ha='center', va='center')


def arrow(start, end):
    "Draw a thin arrow between two points."
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle='-|>', color=muted_color, lw=1.2))


def label(x, y, text, **kwargs):
    "Write a label in the muted text color."
    ax.text(x, y, text, color=muted_color, ha='center', va='center', **kwargs)


fig, ax = plt.subplots(figsize=(8, 5))

# Positions are implicit, so they are not drawn as cells
for i in range(n):
    label(1.0, n - 0.5 - i, str(i))
label(1.0, header_y + 0.5, 'position', fontsize=8.5)

# The index
draw_cells(ax, x_index, index_width, df.index, index_color, '')
draw_header(x_index, index_width, df.index.name, index_color)
label(x_index + index_width / 2, -0.95, 'df.index', family='monospace')
arrow((x_index + index_width / 2, -0.65), (x_index + index_width / 2, -0.05))

# Each column is a Series with its own data type, and they share the index
for j, column in enumerate(df.columns):
    x = x_values + j * (width + gap)
    style = '{:,.0f}' if column in ('POP', 'tcgdp') else '{:.1f}'
    entries = [style.format(v) for v in df[column]]
    draw_cells(ax, x, width, entries, value_color, '')
    draw_header(x, width, column, column_color)
    ax.add_patch(FancyBboxPatch((x - 0.1, -0.12), width + 0.2, header_y + 1.24,
                                boxstyle='round,pad=0,rounding_size=0.18',
                                fill=False, linestyle='--',
                                edgecolor=muted_color))
    label(x + width / 2, -0.55, str(df[column].dtype),
          family='monospace', fontsize=8.5)
    ax.plot([x + width / 2, x_center], [header_y + 1.12, header_y + 2.0],
            ls=':', color=muted_color, lw=1)

label(x_center, header_y + 2.3, 'each column is a Series')

# The column labels
label(x_end + 2.15, header_y + 0.5, 'df.columns', family='monospace')
arrow((x_end + 0.9, header_y + 0.5), (x_end + 0.15, header_y + 0.5))

# The two axes
arrow((0.3, n), (0.3, 0.2))
label(-0.1, n / 2, 'axis=0', family='monospace', fontsize=8.5, rotation=90)
arrow((x_index + 1.0, header_y + 2.0), (x_index + 3.0, header_y + 2.0))
label(x_index + 0.25, header_y + 2.0, 'axis=1', family='monospace', fontsize=8.5)

ax.set_xlim(-0.4, x_end + 3.4)
ax.set_ylim(-1.3, header_y + 2.7)
ax.axis('off')
plt.show()
```

Each column is a `Series` with its own data type, and all the columns share the same index.

This figure, the earlier comparison of an array and a `Series`, and the merge figure below follow the diagrams in Lev Maximov's [Pandas Illustrated](https://medium.com/better-programming/pandas-illustrated-the-definitive-visual-guide-to-pandas-c31fa921a43), which is a good visual guide to the library.

### A first look at a dataset

Seven rows fit on the screen, but most real datasets do not.

So the first step with a new dataset is usually to run a few standard checks.

The `shape` attribute gives the number of rows and columns.

```{code-cell} ipython3
df.shape
```

The `head` method shows the first few rows.

```{code-cell} ipython3
df.head(3)
```

The `info` method lists each column with its data type and its number of non-missing values.

```{code-cell} ipython3
df.info()
```

The `describe` method gives summary statistics for each numeric column.

```{code-cell} ipython3
df.describe()
```

These checks catch many problems early, such as a column of numbers that was read in as text.

### Select data by position and label

In practice, one thing that we do all the time is to find, select and work with a subset of the data of our interests. 

To select rows and columns by position, the `iloc` attribute should be used with the format `.iloc[rows, columns]`.

```{code-cell} ipython3
df.iloc[2:5, 0:2]
```

To select rows and columns by label, the `loc` attribute can be used in a similar way

```{code-cell} ipython3
df.loc[['India', 'Israel', 'Malawi'], ['POP', 'tcgdp']]
```

We can also use it to pick out a single number

```{code-cell} ipython3
print(df.loc['India', 'POP'])
```

To select whole columns, we can pass a list containing their names

```{code-cell} ipython3
df[['POP', 'tcgdp']]
```

A single column is a `Series`, with the same index as the DataFrame

```{code-cell} ipython3
df['POP']
```

For column names without spaces, `df.POP` does the same thing.

### Select data by conditions

Instead of indexing rows and columns using integers and names, we can also obtain a sub-dataframe of our interests that satisfies certain (potentially complicated) conditions.

This section demonstrates various ways to do that.

The most straightforward way is with the `[]` operator.

```{code-cell} ipython3
df[df.POP >= 20000]
```

To understand what is going on here, notice that `df.POP >= 20000` returns a series of boolean values.

```{code-cell} ipython3
df.POP >= 20000
```

In this case, `df[___]` takes a series of boolean values and only returns rows with the `True` values.

Take one more example,

```{code-cell} ipython3
df[df.index.isin(['Argentina', 'India', 'South Africa']) & (df.POP > 40000)]
```

In fact, there is another way of doing the same thing, with more natural syntax.


```{code-cell} ipython3
df.query("country in ['Argentina', 'India', 'South Africa'] and POP > 40000")
```

The string is a Python expression that pandas evaluates, when called with `df.query`.

The syntax inside the string is quite easy to read.

Here's another example.

Consider this arithmetic operation between different columns.

```{code-cell} ipython3
df[(df.cc + df.cg >= 80) & (df.POP <= 20000)]
```

It becomes more readable if we use `df.query`.

```{code-cell} ipython3
df.query("cc + cg >= 80 and POP <= 20000")
```

We can also use `df.query` to select the country with the largest consumption share `cc`.

```{code-cell} ipython3
df.query("cc == cc.max()")
```

When selecting rows and columns together, we typically use `.loc`.

For example, in the following code the first argument is the condition and the second is the list of columns we want to return.

```{code-cell} ipython3
df.loc[(df.cc + df.cg >= 80) & (df.POP <= 20000), ['POP', 'tcgdp']]
```

We can save any selection to a file for further analysis.

```{code-block} python3
:class: no-execute

df_subset = df[['POP', 'tcgdp']]
df_subset.to_csv('population_and_gdp.csv')
```

### The apply method

Another widely used pandas method is `df.apply()`. 

It applies a function to each row/column and returns a series. 

This function can be some built-in functions like the `max` function, a `lambda` function, or a user-defined function.

Here is an example using the `max` function

```{code-cell} ipython3
df.apply(max)
```

This line of code applies the `max` function to each column.

A `lambda` function is often used with `df.apply()`.

For example, this adds the consumption and government shares in each row

```{code-cell} ipython3
df.apply(lambda row: row.cc + row.cg, axis=1)
```

```{note}
For the `.apply()` method
- axis = 0 -- apply function to each column (variables)
- axis = 1 -- apply function to each row (observations)
- axis = 0 is the default parameter
```

There is an easier way to add those two shares.

```{code-cell} ipython3
df.cc + df.cg
```

But `apply` is useful for more complex operations.

For example, suppose that a donor is deciding which countries can receive aid.

Its rule has two parts:

* A country with at most 40 million people qualifies if its GDP per capita is below 10,000 dollars.
* A country with more than 40 million people qualifies if its GDP per capita is below 5,000 dollars, because it has more domestic resources to draw on.

We can write this rule as a function that takes one row and returns `True` or `False`.

```{code-cell} ipython3
def eligible(row):
    # tcgdp is in millions and POP is in thousands
    gdp_percap = row.tcgdp * 1e3 / row.POP
    limit = 5000 if row.POP > 40_000 else 10_000
    return gdp_percap < limit
```

Applying this function to each row gives a series of boolean values.

```{code-cell} ipython3
df.apply(eligible, axis=1)
```

We can use this series with `.loc[]` to select the eligible countries and the columns we want.

```{code-cell} ipython3
df.loc[df.apply(eligible, axis=1), ['POP', 'tcgdp']]
```

South Africa misses out because of its size, even though it is poorer than Argentina.

### Make changes in DataFrames

The ability to make changes in dataframes is important to generate a clean dataset for future analysis.


**1.** We can simply use `.loc[]` to specify the column that we want to modify, and assign values

For example, suppose we learn that the figure for India's government share is unreliable.

We can mark it as missing, so that we don't use it by accident.

```{code-cell} ipython3
df.loc['India', 'cg'] = np.nan
df
```

**2.** We can use the `.apply()` method to modify *rows/columns as a whole*

For example, suppose we want to report GDP in billions rather than millions.

```{code-cell} ipython3
def update_row(row):
    # convert tcgdp from millions to billions
    row.tcgdp = row.tcgdp / 1000
    return row

df.apply(update_row, axis=1)
```

This returns a new DataFrame, so `df` itself is unchanged.

**3.** We can use the `.map()` method to modify all *individual entries* in the dataframe altogether.

```{code-cell} ipython3
# Round all decimal numbers to 2 decimal places
df.map(lambda x: round(x, 2))
```

### Missing values

Real datasets usually have gaps, which pandas marks with `NaN`.

So far `df` has one, India's government share, so let's insert a few more.

```{code-cell} ipython3
for idx in list(zip([0, 3, 5, 6], [2, 3, 2, 3])):
    df.iloc[idx] = np.nan

df
```

The `zip()` function here creates pairs of values from the two lists (i.e. [0,2], [3,3] ...)

The first step is to find the gaps.

The `isna` method marks each missing value as `True`, and summing counts them in each column.

```{code-cell} ipython3
df.isna().sum()
```

One option is to drop every row that has a missing value.

```{code-cell} ipython3
df.dropna()
```

Notice how much data this loses: only two of the seven countries remain.

Another option is to fill the gaps with a fixed value.

```{code-cell} ipython3
df.fillna(0)
```

This is clearly wrong here, since no country has a consumption share of zero.

A common alternative is single imputation, which fills each gap with the mean of its column.

```{code-cell} ipython3
df.fillna(df.mean())
```

None of these calls changes `df`, since each one returns a new DataFrame.

Missing value imputation is a big area in data science involving various machine learning techniques.

There are also more [advanced tools](https://scikit-learn.org/stable/modules/impute.html) in Python to impute missing values.

### New columns, sorting and plotting

For the rest of this section we only need population (`POP`) and total GDP (`tcgdp`), which have no gaps.

```{code-cell} ipython3
df = df[['POP', 'tcgdp']]
df
```

Let's give the columns slightly better names

```{code-cell} ipython3
df.columns = 'population', 'total GDP'
df
```

The `population` variable is in thousands, let's revert to single units

```{code-cell} ipython3
df['population'] = df['population'] * 1e3
df
```

Next, we're going to add a column showing real GDP per capita, multiplying by 1,000,000 as we go because total GDP is in millions

```{code-cell} ipython3
df['GDP percap'] = df['total GDP'] * 1e6 / df['population']
df
```

One of the nice things about pandas `DataFrame` and `Series` objects is that they have methods for plotting and visualization that work through Matplotlib.

For example, we can easily generate a bar plot of GDP per capita

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: GDP per capita, in alphabetical order
    name: gdp_percap_alphabetical
---
ax = df['GDP percap'].plot(kind='bar')
ax.set_xlabel('country', fontsize=12)
ax.set_ylabel('GDP per capita', fontsize=12)
plt.show()
```

At the moment the data frame is ordered alphabetically on the countries---let's change it to GDP per capita

```{code-cell} ipython3
df = df.sort_values(by='GDP percap', ascending=False)
df
```

Plotting as before now yields

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: GDP per capita, sorted
    name: gdp_percap_sorted
---
ax = df['GDP percap'].plot(kind='bar')
ax.set_xlabel('country', fontsize=12)
ax.set_ylabel('GDP per capita', fontsize=12)
plt.show()
```

### Merging and grouping

Most real analyses combine data from more than one source.

Suppose we have a second table that assigns each country to a region.

```{code-cell} ipython3
regions = pd.DataFrame({
    'country': ['Argentina', 'Australia', 'India', 'Israel',
                'Malawi', 'South Africa', 'USA'],
    'region': ['Americas', 'Oceania', 'Asia', 'Asia',
               'Africa', 'Africa', 'Americas'],
})
regions
```

We can combine the two tables with `merge`, matching rows on the country name.

```{code-cell} ipython3
merged = df.merge(regions, on='country')
merged
```

The result has `country` as an ordinary column again, with a new integer index.

Now look closely at the rows: the United States has disappeared.

The second table calls it `USA`, so the row has no match, and by default `merge` keeps only the rows that match.

No error or warning tells us this has happened.

One useful habit is to compare the number of rows before and after a merge.

```{code-cell} ipython3
print(len(df), len(merged))
```

Another is to use `how='left'`, which keeps every row of the first table and shows the gap as `NaN`.

```{code-cell} ipython3
df.merge(regions, on='country', how='left')
```

The figure below shows both merges, with the unmatched rows highlighted.

Blue cells come from `df` and green cells come from `regions`.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Two ways to merge, with population in millions
    name: merge_inner_left
tags: [hide-input]
---
unmatched_color = '#fbdccb'
source_colors = {'population': value_color, 'region': index_color}


def draw_cell(x, y, width, text, color):
    "Draw one table cell with its lower left corner at (x, y)."
    ax.add_patch(Rectangle((x, y), width, 1, facecolor=color,
                           edgecolor=edge_color))
    ax.text(x + width / 2, y + 0.5, text, ha='center', va='center', fontsize=9)


def as_text(frame):
    "Return a table of strings, with population in millions."
    table = frame.reset_index() if frame.index.name else frame.copy()
    if 'population' in table:
        table['population'] = (table['population'] / 1e6).map('{:,.1f}'.format)
    return table.fillna('NaN')


def merge_color(row, column):
    "Color each cell by its source table, and highlight unmatched rows."
    if row['country'] in ('United States', 'USA'):
        return unmatched_color
    return source_colors.get(column, 'white')


def draw_table(x, top, table, widths, title, cell_color):
    "Draw a table with its top left corner at (x, top)."
    ax.text(x, top + 0.45, title, family='monospace', fontsize=9, va='center')
    for j, column in enumerate(table.columns):
        left = x + sum(widths[:j])
        draw_cell(left, top - 1, widths[j], column, column_color)
        for i, entry in enumerate(table[column]):
            color = cell_color(table.iloc[i], column)
            draw_cell(left, top - 2 - i, widths[j], entry, color)


columns = ['country', 'population', 'region']
inner = as_text(df.merge(regions, on='country'))[columns]
left = as_text(df.merge(regions, on='country', how='left'))[columns]
wide = [3.1, 2.2, 2.2]

fig, ax = plt.subplots(figsize=(8, 6.8))

draw_table(1.0, 19, as_text(df[['population']]), [3.1, 2.2], 'df', merge_color)
draw_table(10.2, 19, regions, [3.1, 2.2], 'regions', merge_color)

draw_table(0, 8.6, inner, wide, "df.merge(regions, on='country')", merge_color)
label(3.75, 0.1, 'United States is dropped, with no warning', fontsize=9)

draw_table(8.7, 8.6, left, wide, "df.merge(regions, on='country', how='left')",
           merge_color)

arrow((3.6, 10.7), (3.6, 9.7))
arrow((12.45, 10.7), (12.45, 9.7))

ax.set_xlim(-0.1, 16.3)
ax.set_ylim(-0.4, 20)
ax.axis('off')
plt.show()
```

Let's fix the name and merge again.

```{code-cell} ipython3
regions['country'] = regions['country'].replace('USA', 'United States')
merged = df.merge(regions, on='country')
merged
```

Now we can use the new column with `groupby`, which splits the rows into groups and computes something for each one.

For example, here is the total population of each region, in millions.

```{code-cell} ipython3
merged.groupby('region')['population'].sum() / 1e6
```

The figure below shows the steps: the rows are split by region, each group is summed, and the results are combined into one table.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Split, sum and combine, with population in millions
    name: groupby_steps
tags: [hide-input]
---
region_colors = {'Africa': '#fbe9bd', 'Americas': '#fbdccb',
                 'Asia': '#d3f0e4', 'Oceania': '#cde2fb'}
widths = [2.1, 2.2]
table = as_text(merged[['region', 'population']])
totals = merged.groupby('region')['population'].sum() / 1e6


def region_color(row, column):
    "Color each cell by its region."
    return region_colors[row['region']]


def draw_rows(x, y, rows):
    "Draw region and population cells downward from a top edge at y."
    for i, (region, value) in enumerate(rows):
        draw_cell(x, y - 1 - i, widths[0], region, region_colors[region])
        draw_cell(x + widths[0], y - 1 - i, widths[1], value,
                  region_colors[region])


fig, ax = plt.subplots(figsize=(8, 4.4))

# The merged table
draw_table(0, 8.25, table, widths, 'merged', region_color)

# Split the rows into one group per region, and sum each group
x_split, x_sum, x_combine = 5.8, 11.4, 14.5
y = 8.5
ax.text(x_split, y + 0.45, 'split', fontsize=9, family='monospace', va='center')
ax.text(x_sum, y + 0.45, 'sum', fontsize=9, family='monospace', va='center')
arrow((4.5, 4.25), (5.6, 4.25))
for region, rows in table.groupby('region'):
    draw_rows(x_split, y, list(zip(rows['region'], rows['population'])))
    center = y - len(rows) / 2
    draw_cell(x_sum, center - 0.5, 1.7, f'{totals[region]:,.1f}',
              region_colors[region])
    arrow((x_split + sum(widths) + 0.1, center), (x_sum - 0.1, center))
    y -= len(rows) + 0.5

# Combine the results into one table
combined = totals.reset_index()
combined['population'] = combined['population'].map('{:,.1f}'.format)
draw_table(x_combine, 6.75, combined, widths, 'combine', region_color)
arrow((x_sum + 1.8, 4.25), (x_combine - 0.1, 4.25))

ax.set_xlim(-0.1, x_combine + sum(widths) + 0.1)
ax.set_ylim(-0.3, 9.4)
ax.axis('off')
plt.show()
```

### Silent failures

Code that runs without an error message can still give the wrong answer.

The merge that lost the United States is one example.

Here is another.

Suppose that Argentina's population is missing, and that we fill the gap with the mean, just as we did for missing values above.

```{code-cell} ipython3
raw = pd.read_csv(url).set_index('country')
raw.loc['Argentina', 'POP'] = np.nan
filled = raw.fillna(raw.mean())

# tcgdp is in millions and POP is in thousands
gdp_percap = filled['tcgdp'] * 1e3 / filled['POP']

comparison = df[['GDP percap']].copy()
comparison['with imputed population'] = gdp_percap
comparison
```

Argentina's GDP per capita is now far too low, because India and the United States pull the mean population well above Argentina's true population.

Nothing warned us: the code ran, the result looks like any other table, and no missing values remain to show that something was changed.

Filling in a missing value is a modeling choice, and every number computed from that value inherits its error.

## Online data sources

```{index} single: Data Sources
```

Someone has to collect every number that we analyze.

Most data used in policy work is produced by national statistical agencies, central banks and international organizations, using surveys, censuses and administrative records such as tax filings.

Before using a series, it is worth asking who produced it, how it was measured and whether it has been revised.

Many of these sources can be read directly from Python.

### FRED

[FRED](https://fred.stlouisfed.org/) is a vast collection of time series data maintained by the St. Louis Fed.

For example, it holds the US [unemployment rate](https://fred.stlouisfed.org/series/UNRATE), which the Bureau of Labor Statistics estimates from a monthly household survey.

Since `read_csv` accepts a URL, reading this series takes one line.

```{code-cell} ipython3
fred_url = 'https://fred.stlouisfed.org/graph/fredgraph.csv?id=UNRATE'
data = pd.read_csv(fred_url, index_col=0, parse_dates=True)
data.head()
```

Here `index_col=0` makes the first column the index, and `parse_dates=True` tells pandas that it holds dates.

Because the index holds dates, we can select a range of years with `.loc`.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: US unemployment rate
    name: us_unemployment
---
ax = data.loc['2006':'2012'].plot(legend=False, lw=2)
ax.set_xlabel('year', fontsize=12)
ax.set_ylabel('unemployment rate (%)', fontsize=12)
plt.show()
```

pandas can also [read many other formats](https://pandas.pydata.org/docs/user_guide/io.html), including Excel, JSON and Parquet files, and it can connect to databases.

### The World Bank

The World Bank [collects and organizes data](https://data.worldbank.org/indicator) on a huge range of indicators.

The [wbgapi](https://pypi.org/project/wbgapi/) library can fetch data from its databases, and this [blog post](https://blogs.worldbank.org/en/opendata/introducing-wbgapi-new-python-package-accessing-world-bank-data) gives an introduction.

For example, [here's](https://data.worldbank.org/indicator/GC.DOD.TOTL.GD.ZS) some data on government debt as a ratio to GDP.

```{code-cell} ipython3
import wbgapi as wb
wb.series.info('GC.DOD.TOTL.GD.ZS')
```

The next code fetches the data and plots time series for the US and Australia.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Government debt as a ratio to GDP
    name: government_debt
---
govt_debt = wb.data.DataFrame('GC.DOD.TOTL.GD.ZS',
                              economy=['USA', 'AUS'],
                              time=range(2005, 2016))
govt_debt = govt_debt.T    # move years from columns to rows for plotting
ax = govt_debt.plot(lw=2)
ax.set_xlabel('year', fontsize=12)
ax.set_ylabel('government debt (% of GDP)', fontsize=12)
plt.show()
```

## Exercises

```{exercise-start}
:label: wwd_ex1
```

You ask an AI tool to analyze the country data in `gdp_population_by_country_2000.csv`, and it returns the code below.

```{code-block} python3
table = pd.read_csv(url).set_index('country')
table = table.query("cg > cg.mean()")
table = table.loc[:, ['POP', 'cg']].sort_values('cg', ascending=False)
table
```

Without running the code, say in one sentence what table it produces.

Then run it to check your answer.

```{exercise-end}
```

```{solution-start} wwd_ex1
:class: dropdown
```

The code lists the countries whose government share is above the average of the seven countries, showing their population and government share, with the largest share first.

```{code-cell} ipython3
table = pd.read_csv(url).set_index('country')
table = table.query("cg > cg.mean()")
table = table.loc[:, ['POP', 'cg']].sort_values('cg', ascending=False)
table
```

```{solution-end}
```

```{exercise-start}
:label: wwd_ex2
```

You ask an AI tool for the total GDP of each region, and it returns the code below.

```{code-block} python3
gdp = pd.read_csv(url)
regions = pd.DataFrame({
    'country': ['Argentina', 'Australia', 'India', 'Israel', 'Israel',
                'Malawi', 'South Africa', 'United States'],
    'region': ['Americas', 'Oceania', 'Asia', 'Asia', 'Middle East',
               'Africa', 'Africa', 'Americas'],
})
merged = gdp.merge(regions, on='country')
merged.groupby('region')['tcgdp'].sum()
```

The code runs without an error.

Can you trust the answer?

Find the problem and fix it.

```{exercise-end}
```

```{solution-start} wwd_ex2
:class: dropdown
```

As in the lecture, we start by comparing the number of rows before and after the merge.

```{code-cell} ipython3
gdp = pd.read_csv(url)
regions = pd.DataFrame({
    'country': ['Argentina', 'Australia', 'India', 'Israel', 'Israel',
                'Malawi', 'South Africa', 'United States'],
    'region': ['Americas', 'Oceania', 'Asia', 'Asia', 'Middle East',
               'Africa', 'Africa', 'Americas'],
})
merged = gdp.merge(regions, on='country')
print(len(gdp), len(merged))
```

The merge has added a row, so some country must appear twice in `regions`.

```{code-cell} ipython3
regions[regions['country'].duplicated(keep=False)]
```

Israel is listed under two regions, so its GDP is counted twice and the regional totals add up to more than the total GDP of the seven countries.

```{code-cell} ipython3
print(gdp['tcgdp'].sum(), merged['tcgdp'].sum())
```

To fix this, we give each country one region.

Which region to use for Israel is a judgment call, and here we choose Asia.

We also pass `validate='many_to_one'`, which makes `merge` raise an error if a country appears twice in `regions`.

```{code-cell} ipython3
regions = regions[regions['region'] != 'Middle East']
merged = gdp.merge(regions, on='country', validate='many_to_one')
merged.groupby('region')['tcgdp'].sum()
```

```{solution-end}
```

```{exercise-start}
:label: wwd_ex3
```

Using the corrected `merged` table from {ref}`wwd_ex2`, compute GDP per capita for each region in two ways:

1. the average of GDP per capita across the countries in the region
1. the total GDP of the region divided by its total population

You can write this yourself or ask an AI tool to do it.

Why are the two answers so different for Asia?

Which one would you report, and when?

```{exercise-end}
```

```{solution-start} wwd_ex3
:class: dropdown
```

Here is one way to compute both.

```{code-cell} ipython3
merged['gdp_percap'] = merged['tcgdp'] * 1e3 / merged['POP']
average = merged.groupby('region')['gdp_percap'].mean()

totals = merged.groupby('region')[['tcgdp', 'POP']].sum()
whole_region = totals['tcgdp'] * 1e3 / totals['POP']

pd.DataFrame({'average of countries': average,
              'region as a whole': whole_region})
```

The average of countries gives India and Israel equal weight, even though India has more than 160 times as many people.

The second number is close to India's own figure, because India makes up nearly all of the region's population.

Neither answer is wrong, because they answer different questions, as in {doc}`python_ai`.

The first describes a typical country, while the second describes the income of a typical person.

```{solution-end}
```

[^mung]: Wikipedia defines munging as cleaning data from one raw form into a structured, purged one.

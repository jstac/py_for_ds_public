---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.6
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# The Toolkit: What Is Possible

In this lecture we look at the hardware that runs our code, and then take a visual tour of what Python's main libraries can do.

You do not need to memorize any of the code.

The aim is to know what is possible, so that you can ask an AI for it and judge what comes back.

Most of the code behind the figures is hidden, and you can click to show it.

In addition to what's in Anaconda, this lecture needs JAX, which we discuss below, and a library containing the trade data that we use.

```{code-cell} ipython3
---
tags: [hide-output]
---
!pip install jax
!pip install quantecon_book_networks
```

We use these imports.

```{code-cell} ipython3
import json
import os
import urllib.request

import jax
import jax.numpy as jnp
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import statsmodels.api as sm
from matplotlib.collections import PolyCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from scipy.integrate import quad
from scipy.stats import norm
from sklearn.cluster import KMeans
import quantecon_book_networks.data as qbn_data
import quantecon_book_networks.input_output as qbn_io

# Colors used in the figures below
blue, orange, aqua, yellow = '#2a78d6', '#eb6834', '#1baf7a', '#eda100'
blues = LinearSegmentedColormap.from_list(
    'blues', ['#cde2fb', '#6da7ec', '#2a78d6', '#184f95', '#0d366b'])
```

## Hardware

Data science increasingly depends on specialized hardware, so we start with the basic parts of a computer.

### Processors and memory

The [central processing unit](https://en.wikipedia.org/wiki/Central_processing_unit) (CPU) is the part of a computer that carries out calculations.

Modern CPUs contain several [cores](https://en.wikipedia.org/wiki/Multi-core_processor), each of which can work on a separate task.

[Memory](https://en.wikipedia.org/wiki/Random-access_memory), or RAM, is the fast workspace that holds the data the CPU is currently using.

Think of the CPU as a cook and memory as the kitchen counter: a bigger counter lets the cook work with more ingredients at once.

Long-term storage on a disk is more like the pantry, which is much larger but slower to reach.

Each number in a typical NumPy array takes up 8 bytes of memory, so we can estimate the memory that a data set needs.

```{code-cell} ipython3
x = np.zeros(1_000_000)    # An array of one million numbers
x.nbytes / 1e6             # Memory used, in megabytes
```

A table with 10 million rows and 50 columns of such numbers therefore needs about 4 gigabytes, which is a real fraction of a typical laptop's memory.

If your data do not fit in memory, your program will slow down dramatically or crash.

### Accelerators

An [accelerator](https://en.wikipedia.org/wiki/Hardware_acceleration) is a specialized chip that does certain calculations much faster than a CPU.

The best-known type is the [graphics processing unit](https://en.wikipedia.org/wiki/Graphics_processing_unit) (GPU), originally built to draw images for computer games.

A CPU has a few powerful cores that can handle almost any task, while a GPU has thousands of simpler units that perform the same operation on many numbers at once.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: "CPU and GPU cores (schematic, not to scale)"
    name: cpu_gpu
tags: [hide-input]
---
fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))

for ax, (rows, cols, title) in zip(axes, [(2, 4, 'CPU: a few powerful cores'),
                                          (16, 32, 'GPU: many simple units')]):
    for i in range(rows):
        for j in range(cols):
            ax.add_patch(Rectangle((j + 0.1, i + 0.1), 0.8, 0.8, color=blue))
    ax.set_xlim(-0.2, cols + 0.2)
    ax.set_ylim(-0.2, rows + 0.2)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)

plt.show()
```

Another type is Google's [tensor processing unit](https://en.wikipedia.org/wiki/Tensor_Processing_Unit) (TPU), a chip [designed for](https://docs.cloud.google.com/tpu/docs/intro-to-tpu) the large matrix calculations in machine learning.

Accelerators help only when a task can be split into many similar pieces.

They also have their own memory, so data must be moved onto them before they can work.

### Why this matters for AI and statistics

Training and running a [neural network](https://en.wikipedia.org/wiki/Neural_network_(machine_learning)) comes down to an enormous number of matrix calculations.

These are exactly the calculations that accelerators are good at, which is why modern AI runs mostly on them.

It is also why the AI tools you use run on a company's servers rather than on your laptop.

Accelerators matter for statistics too.

For example, the [Bayesian analysis](https://en.wikipedia.org/wiki/Bayesian_statistics) that we study in {doc}`bayes_intro` often relies on running many simulations side by side, which suits a GPU well.

### Which library runs on what

Most policy work runs perfectly well on a laptop CPU, but it helps to know which libraries target which hardware.

| Library | Main use | Runs on |
|---|---|---|
| NumPy, SciPy | Arrays and scientific routines | CPU |
| pandas | Tables of data | CPU |
| Matplotlib, seaborn | Figures | CPU |
| statsmodels | Statistical models | CPU |
| scikit-learn | Machine learning | CPU, with [experimental GPU support](https://scikit-learn.org/stable/modules/array_api.html) |
| [Polars](https://docs.pola.rs/) | Tables of data | All CPU cores, with optional NVIDIA GPU support |
| [cuDF](https://docs.nvidia.com/cudf/latest/) | Tables of data, in the style of pandas | NVIDIA GPU |
| [PyTorch](https://pytorch.org/) | Deep learning | CPU, GPU |
| [TensorFlow](https://www.tensorflow.org/) | Deep learning | CPU, GPU, TPU |
| [JAX](https://docs.jax.dev/en/latest/installation.html) | Fast array calculations in the style of NumPy | CPU, GPU, TPU |
| [NumPyro](https://num.pyro.ai/) | Bayesian analysis | CPU, GPU (built on JAX) |

{ref}`Google Colab <overview:colab>` offers access to GPUs and TPUs, and you can try them under Runtime, then Change runtime type.

## NumPy

[NumPy](https://numpy.org/) is the foundation of numerical work in Python, and it provides fast operations on arrays and matrices.

### Fast array operations

Here we add up the squares of a million random numbers, first with a Python loop.

```{code-cell} ipython3
rng = np.random.default_rng(1234)
x = rng.uniform(size=1_000_000)
```

```{code-cell} ipython3
%%time
total = 0
for v in x:
    total = total + v**2
```

Now we do the same calculation with NumPy.

```{code-cell} ipython3
%%time
total = np.sum(x**2)
```

Compare the two "Wall time" figures, which show how long each one took.

NumPy is much faster because it passes the work to precompiled code that runs without the overhead of a Python loop.

Operating on a whole array at once like this is called [vectorization](https://en.wikipedia.org/wiki/Array_programming).

### Matrices and linear equations

Many problems reduce to solving a system of linear equations.

Suppose that demand and supply for a good satisfy $q = 10 - p$ and $q = 2 + p$, where $q$ is quantity and $p$ is price.

We can rewrite these two equations as

$$
\begin{aligned}
q + p & = 10 \\
q - p & = 2
\end{aligned}
$$

Next we collect the coefficients in a [matrix](https://en.wikipedia.org/wiki/Matrix_(mathematics)), and the unknowns and the right-hand side in vectors.

$$
\begin{bmatrix}
1 & 1 \\
1 & -1
\end{bmatrix}
\begin{bmatrix}
q \\
p
\end{bmatrix}
=
\begin{bmatrix}
10 \\
2
\end{bmatrix}
$$

This has the form $A x = b$, and multiplying each row of $A$ by $x$ gives back one of the original equations.

NumPy can now solve the system directly.

```{code-cell} ipython3
A = np.array([[1, 1],
              [1, -1]])
b = np.array([10, 2])

np.linalg.solve(A, b)
```

So the quantity is 6 and the price is 4, which is where the two lines cross.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: "Solving two equations graphically"
    name: supply_demand
tags: [hide-input]
---
q = np.linspace(2, 10, 100)

fig, ax = plt.subplots()
ax.plot(q, 10 - q, lw=2, color=blue, label='demand')
ax.plot(q, q - 2, lw=2, color=orange, label='supply')
ax.plot(6, 4, 'o', ms=9, color='black', label='solution')
ax.annotate('$q = 6$, $p = 4$', xy=(6, 4), xytext=(6.5, 4.0))
ax.set_xlabel('quantity')
ax.set_ylabel('price')
ax.legend()
plt.show()
```

The same command works for much larger systems, such as one with 8,000 equations and 8,000 unknowns.

```{code-cell} ipython3
n = 8_000
A = rng.normal(size=(n, n))
b = rng.normal(size=n)
```

```{code-cell} ipython3
%%time
x = np.linalg.solve(A, b)
```

Calculations like this one are the kind of work that accelerators speed up.

[JAX](https://docs.jax.dev/) lets us run them on an accelerator with almost no change to the code, because `jnp` is its version of NumPy.

First we ask which hardware JAX can see.

```{code-cell} ipython3
jax.devices()
```

This page was built on a machine with no accelerator, so JAX reports a CPU.

Here is the same calculation in JAX, which we run once as a warm-up.

```{code-cell} ipython3
x = jnp.linalg.solve(A, b).block_until_ready()
```

The first run is slower, because JAX [compiles](https://en.wikipedia.org/wiki/Just-in-time_compilation) the calculation for the hardware, so a fair timing needs a second run.

```{code-cell} ipython3
%%time
x = jnp.linalg.solve(A, b).block_until_ready()
```

We need `block_until_ready` because JAX otherwise moves on to the next line while the calculation is still running.

Compare this with the NumPy timing above: at this size, JAX often wins even on a CPU.

On an accelerator, the gap can be far larger: in one test on a desktop GPU, a system of this size took about 0.04 seconds, compared to 0.6 seconds for NumPy on that machine's CPU.

These comparisons flatter JAX, since it uses lower-precision numbers by default, while NumPy uses higher precision.

To try this yourself, choose a GPU in Colab under Runtime, then Change runtime type, and run these cells again.

## SciPy

[SciPy](https://scipy.org/) builds on NumPy, with routines for probability distributions, optimization, numerical integration and more.

As an example, consider finding the area under a curve, which is called [numerical integration](https://en.wikipedia.org/wiki/Numerical_integration).

The simplest approach is to cover the area with rectangles and add up their areas, as in the figure below.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: "Approximating an area with rectangles"
    name: integration
tags: [hide-input]
---
ϕ = norm()
grid = np.linspace(-3, 3, 200)
a, b = -2, 2


def rectangle_area(n):
    width = (b - a) / n
    midpoints = a + width * (np.arange(n) + 0.5)
    return midpoints, width, np.sum(ϕ.pdf(midpoints) * width)


fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), sharey=True)

fill = grid[(grid >= a) & (grid <= b)]
axes[0].fill_between(fill, ϕ.pdf(fill), color=blue, alpha=0.4)
axes[0].set_title('area to compute')

for ax, n in zip(axes[1:], (6, 24)):
    midpoints, width, area = rectangle_area(n)
    ax.bar(midpoints, ϕ.pdf(midpoints), width=width,
           color=blue, alpha=0.4, edgecolor='white')
    ax.set_title(f'{n} rectangles: area {area:.4f}')

for ax in axes:
    ax.plot(grid, ϕ.pdf(grid), lw=2, color='black')
    ax.set_xlabel('$z$')

axes[0].set_ylabel('density')
plt.show()
```

The approximation improves as the rectangles get narrower.

SciPy's `quad` function uses a more sophisticated version of this idea.

Here it computes the same area, under the [normal density](https://en.wikipedia.org/wiki/Normal_distribution) between $-2$ and $2$.

```{code-cell} ipython3
value, error = quad(ϕ.pdf, -2, 2)
value
```

## Visualizing data

Python has excellent libraries for visualization, and asking an AI for a figure is one of the best uses of it.

### Comparing countries with seaborn

[Seaborn](https://seaborn.pydata.org/) produces statistical figures from a table of data with very little code.

We use the [Gapminder](https://www.gapminder.org/) data, which contains life expectancy, income and population by country.

```{code-cell} ipython3
gapminder = px.data.gapminder()
gapminder_2007 = gapminder[gapminder['year'] == 2007]
gapminder_2007[['country', 'continent', 'lifeExp', 'pop', 'gdpPercap']].head()
```

The next figure shows how life expectancy rises with income, with a separate panel for each continent and larger dots for larger populations.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: "Income and life expectancy, 2007"
    name: gapminder_scatter
---
g = sns.relplot(data=gapminder_2007, x='gdpPercap', y='lifeExp',
                col='continent', col_wrap=3, height=2.6,
                size='pop', sizes=(10, 400), color=blue, alpha=0.6,
                legend=False, facet_kws={'despine': False})
g.set(xscale='log')
g.set_titles('{col_name}')
g.set_axis_labels('GDP per capita (log scale)', 'life expectancy')
plt.show()
```

A violin plot shows the whole distribution within each group, which a simple average would hide.

Here we leave out Oceania, which has only two countries in these data.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: "Life expectancy by continent, 2007"
    name: gapminder_violin
---
data = gapminder_2007[gapminder_2007['continent'] != 'Oceania']

fig, ax = plt.subplots()
sns.violinplot(data=data, x='continent', y='lifeExp', color=blue,
               inner='quartile', cut=0, density_norm='width', ax=ax)
ax.set_xlabel('continent')
ax.set_ylabel('life expectancy')
plt.show()
```

A heat map shows many numbers at once, using color for their size.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: "Mean life expectancy by continent and year"
    name: gapminder_heatmap
---
table = gapminder.pivot_table(index='continent', columns='year', values='lifeExp')

fig, ax = plt.subplots(figsize=(9, 3.2))
sns.heatmap(table, cmap=blues, annot=True, fmt='.0f', linewidths=1,
            cbar_kws={'label': 'years'}, ax=ax)
ax.set_xlabel('year')
ax.set_ylabel('continent')
plt.show()
```

### Time series

A [time series](https://en.wikipedia.org/wiki/Time_series) records a quantity over time.

Here is life expectancy in four countries from 1952, where the sharp fall for China coincides with the [famine of 1959 to 1961](https://en.wikipedia.org/wiki/Great_Chinese_Famine).

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: "Life expectancy in four countries"
    name: life_expectancy
tags: [hide-input]
---
fig, ax = plt.subplots()

for country, color in zip(['Japan', 'United States', 'China', 'India'],
                          [blue, orange, aqua, yellow]):
    data = gapminder[gapminder['country'] == country]
    ax.plot(data['year'], data['lifeExp'], lw=2, color=color, label=country)
    ax.annotate(country, xy=(2007, data['lifeExp'].iloc[-1]),
                xytext=(5, 0), textcoords='offset points', va='center')

ax.set_xlim(1950, 2025)
ax.set_xlabel('year')
ax.set_ylabel('life expectancy')
ax.legend(loc='lower right')
plt.show()
```

Next is the concentration of carbon dioxide in the atmosphere, measured weekly at Mauna Loa in Hawaii.

The left panel shows the long-run rise, while the right panel zooms in to reveal a seasonal cycle.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: "Atmospheric carbon dioxide at Mauna Loa"
    name: co2
tags: [hide-input]
---
co2 = sm.datasets.co2.load_pandas().data['co2'].interpolate()

fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))

axes[0].plot(co2, lw=2, color=blue)
axes[0].set_title('1958 to 2001')
axes[1].plot(co2['1998':'2001'], lw=2, color=blue)
axes[1].set_title('1998 to 2001')
axes[1].xaxis.set_major_locator(mdates.YearLocator())
axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

for ax in axes:
    ax.set_xlabel('date')
    ax.set_ylabel('CO$_2$ (parts per million)')

plt.tight_layout()
plt.show()
```

### Maps

A [choropleth map](https://en.wikipedia.org/wiki/Choropleth_map) uses color to show a quantity across regions.

The next figure shows the population of Japan's 47 prefectures in the [2020 census](https://www.stat.go.jp/english/data/kokusei/2020/summary.html), using boundaries from [Natural Earth](https://www.naturalearthdata.com/).

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: "Population of Japan by prefecture, 2020"
    name: japan_map
tags: [hide-input]
---
path = '_static/lecture_specific/toolkit/japan_prefectures.geojson'
url = ('https://raw.githubusercontent.com/jstac/py_for_ds_public/main/lectures/'
       + path)

if os.path.exists(path):
    with open(path) as f:
        japan = json.load(f)
else:
    with urllib.request.urlopen(url) as f:
        japan = json.load(f)

polygons, populations, centers = [], [], {}

for feature in japan['features']:
    geometry = feature['geometry']
    parts = geometry['coordinates']
    if geometry['type'] == 'Polygon':
        parts = [parts]
    population = feature['properties']['population'] / 1e6
    for part in parts:
        polygons.append(np.array(part[0]))
        populations.append(population)
    largest = max(parts, key=lambda part: len(part[0]))
    centers[feature['properties']['name']] = np.array(largest[0]).mean(axis=0)

fig, ax = plt.subplots(figsize=(8, 8))

shapes = PolyCollection(polygons, array=np.array(populations), cmap=blues,
                        edgecolor='white', linewidth=0.5)
ax.add_collection(shapes)
fig.colorbar(shapes, ax=ax, shrink=0.6, label='population (millions)')

for name, offset in {'Tokyo': (2.2, -1.4), 'Osaka': (-1.2, -2.6),
                     'Hokkaido': (2.2, 1.6), 'Okinawa': (1.4, 1.0)}.items():
    ax.annotate(name, xy=centers[name], xytext=centers[name] + np.array(offset),
                arrowprops={'arrowstyle': '-', 'color': 'gray'})

ax.set_xlim(127, 147)
ax.set_ylim(25.5, 46)
ax.set_aspect(1 / np.cos(np.radians(36)))    # Correct for the shape of the Earth
ax.axis('off')
plt.show()
```

The dark region around Tokyo shows how concentrated the population is.

Notice that large prefectures such as Hokkaido look important on a map even when they have relatively few people, which is a common way for maps to mislead.

## Networks

A [network](https://en.wikipedia.org/wiki/Graph_(discrete_mathematics)) is a set of nodes connected by links, and the [NetworkX](https://networkx.org/) library works with them in Python.

The next figure shows international trade in large commercial aircraft in 2019, using data from {cite:t}`sargent2024economic`.

Each node is a country, and each arrow is a flow of exports.

Larger nodes export more, and darker nodes are more central to the network.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: "International trade in commercial aircraft, 2019"
    name: aircraft_network
tags: [hide-input]
---
data = qbn_data.introduction()
network = data['aircraft_network']
positions = data['aircraft_network_pos']

centrality = nx.eigenvector_centrality(network)
sizes = qbn_io.normalise_weights(qbn_io.node_total_exports(network), 5000)
widths = qbn_io.normalise_weights(qbn_io.edge_weights(network), 6)
shares = {node: c / max(centrality.values()) for node, c in centrality.items()}
colors = [blues(shares[node]) for node in network]

fig, ax = plt.subplots(figsize=(9, 9))
nx.draw_networkx_edges(network, positions, width=widths, edge_color='gray',
                       alpha=0.4, arrows=True, ax=ax)
nx.draw_networkx_nodes(network, positions, node_size=sizes,
                       node_color=colors, edgecolors='white', ax=ax)

# Use white text on the dark nodes, so that their labels can be read
for font_color, keep in (('white', lambda s: s > 0.5), ('black', lambda s: s <= 0.5)):
    labels = {node: node for node in network if keep(shares[node])}
    nx.draw_networkx_labels(network, positions, labels=labels, font_size=8,
                            font_color=font_color, ax=ax)

ax.axis('off')
plt.show()
```

Small networks can be just as revealing, and they take very little code.

This one shows marriage ties between leading families in fifteenth-century Florence, a classic example from {cite:t}`padgett1993robust`, and it ships with NetworkX.

We size each family by [betweenness centrality](https://en.wikipedia.org/wiki/Betweenness_centrality), which measures how often it lies on the shortest path between other families.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: "Marriage ties in fifteenth-century Florence"
    name: florentine
---
families = nx.florentine_families_graph()
centrality = nx.betweenness_centrality(families)
sizes = [300 + 6000 * centrality[family] for family in families]

positions = nx.spring_layout(families, seed=1234)

fig, ax = plt.subplots(figsize=(8, 6))
nx.draw_networkx_edges(families, positions, edge_color='gray', ax=ax)
nx.draw_networkx_nodes(families, positions, node_size=sizes,
                       node_color='#9ec5f4', edgecolors=blue, ax=ax)
nx.draw_networkx_labels(families, positions, font_size=9, ax=ax)
ax.axis('off')
plt.show()
```

The Medici stand out as the most central family.

We return to networks in {doc}`networks`.

## pandas and Polars

[pandas](https://pandas.pydata.org/) is the standard library for working with tables of data, and we study it in {doc}`working_with_data`.

For example, this line finds the median life expectancy on each continent.

```{code-cell} ipython3
gapminder_2007.groupby('continent')['lifeExp'].median()
```

[Polars](https://docs.pola.rs/) does a similar job with a different design, and it uses all of your CPU's cores by default.

It can be much faster on large data sets, but pandas remains the most common choice, and it is what most AI tools produce by default.

## scikit-learn

[scikit-learn](https://scikit-learn.org/) is the standard library for machine learning outside of deep learning.

As an example, we ask its [k-means](https://en.wikipedia.org/wiki/K-means_clustering) algorithm to divide the countries into three groups, using only income and life expectancy.

The algorithm is never told what the groups should be.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: "Three clusters of countries, 2007"
    name: kmeans
---
features = np.column_stack([np.log(gapminder_2007['gdpPercap']),
                            gapminder_2007['lifeExp'] / 10])

model = KMeans(n_clusters=3, n_init=10, random_state=1234)
labels = model.fit_predict(features)

fig, ax = plt.subplots()

for k, (color, marker) in enumerate(zip([blue, orange, aqua], ['o', 's', '^'])):
    group = gapminder_2007[labels == k]
    ax.scatter(group['gdpPercap'], group['lifeExp'], s=50, color=color,
               marker=marker, edgecolor='white', label=f'cluster {k + 1}')

ax.set_xscale('log')
ax.set_xlabel('GDP per capita (log scale)')
ax.set_ylabel('life expectancy')
ax.legend()
plt.show()
```

The groups look sensible, but the algorithm will always return three groups when asked, whether or not three real groups exist.

Deciding whether they mean anything is your job, as we discuss in {doc}`classification_forecasting`.

## Git and GitHub

Finally, we turn to a tool for keeping track of changes to code.

[Git](https://en.wikipedia.org/wiki/Git) is a version control system, which records each saved version of a project, called a commit.

[GitHub](https://github.com/) is a website that stores these projects online, so that people can share them and work on them together.

Most of the libraries above keep their code on GitHub, and so does [this book](https://github.com/jstac/py_for_ds_public).

Version control matters even more when an AI is editing your work, since it shows you exactly what changed.

This view is called a [diff](https://en.wikipedia.org/wiki/Diff), where red lines were removed and green lines were added.

```{code-block} diff
- if m > 25:
+ if m >= 25:
      count = count + 1
```

Here an AI has changed `>` to `>=`, a small edit that changes the answer, as we saw in {doc}`python_ai`.

If you do not like a change, then you can go back to the previous commit, which makes it safer to let an AI make large edits.

A [branch](https://en.wikipedia.org/wiki/Branching_(version_control)) is a separate line of work within a project, which leaves the main version, usually called `main`, untouched.

If the work on a branch turns out well, you merge it into `main`, and if not, you delete the branch and nothing is lost.

The figure below shows both outcomes, where each circle is a commit.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: "Branching, merging and deleting a branch"
    name: git_branches
tags: [hide-input]
---
fig, ax = plt.subplots(figsize=(9, 2.8))

main = np.arange(8)
merged_x, merged_y = [1, 2, 3, 4], [0, 1, 1, 0]
deleted_x, deleted_y = [5, 6, 7], [0, -1, -1]

ax.plot(main, np.zeros(8), '-o', lw=2, ms=10, color=blue, zorder=2)
ax.plot(merged_x, merged_y, '-', lw=2, color=orange, zorder=1)
ax.plot(merged_x[1:3], merged_y[1:3], 'o', ms=10, color=orange)
ax.plot(deleted_x, deleted_y, '--', lw=2, color='gray', zorder=1)
ax.plot(deleted_x[1:], deleted_y[1:], 'o', ms=10, mfc='white', mec='gray', mew=2)

ax.annotate('main', xy=(7, 0), xytext=(14, 0), textcoords='offset points',
            va='center')
ax.annotate('work that is merged', xy=(2.5, 1), xytext=(0, 12),
            textcoords='offset points', ha='center')
ax.annotate('work that is deleted', xy=(6.5, -1), xytext=(0, -20),
            textcoords='offset points', ha='center')

ax.set_xlim(-0.5, 8.3)
ax.set_ylim(-1.7, 1.7)
ax.axis('off')
plt.show()
```

Branches are helpful even when you work alone, particularly with an AI.

For example, you can ask an AI to try a risky change or a new approach on its own branch, and then review the diff before deciding whether to keep it.

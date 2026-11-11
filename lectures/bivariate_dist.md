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

# Bivariate Distributions, Correlation and Simple Regression

```{index} single: Bivariate Distributions
```

So far we have studied one variable at a time, such as the distribution of house prices, in {doc}`prob_dist`, {doc}`observed_distributions` and {doc}`mle_intro`.

Often we are interested in how two variables relate to each other.

For example, do larger houses tend to sell for more?

In this lecture we study joint distributions and correlation, and then fit a line through data using [simple linear regression](https://en.wikipedia.org/wiki/Simple_linear_regression).

We end with a warning about what such a line can and cannot tell us.

We use the following imports:

```{code-cell} ipython3
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.stats
from scipy.optimize import minimize
import seaborn as sns

np.set_printoptions(legacy='1.25')   # print scalars as plain numbers
```

To motivate what follows, here is a [scatter plot](https://en.wikipedia.org/wiki/Scatter_plot) of the sale prices and floor areas of the Ames houses that we studied in {doc}`observed_distributions`.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: House price against floor area
    name: fig:bivariate-ames-scatter
tags: [hide-input]
---
url = ('https://github.com/QuantEcon/data-lectures/raw/main/'
       'lectures/ames_house_prices.csv')
houses = pd.read_csv(url)
price = houses['price']
area = houses['living_area_sqft']

fig, ax = plt.subplots()
ax.scatter(area, price, alpha=0.3, s=10)
ax.set_xlabel('living area (square feet)')
ax.set_ylabel('sale price (US$)')
plt.show()
```

Each point is one house, and larger houses clearly tend to sell for more.

This is the kind of pattern that a joint distribution can describe.

## Joint and marginal distributions

We start with the discrete case and then turn to densities.

### Discrete case

Let $X$ and $Y$ be two discrete random variables, taking values in finite sets $S_X$ and $S_Y$.

Their [joint distribution](https://en.wikipedia.org/wiki/Joint_probability_distribution) is described by the function

$$
p(x,y) = \mathbb P\{X = x, Y = y\}
$$

As with a single variable, the values of $p$ are nonnegative and sum to one, now over both variables.

Let's build an example from the house price data.

Let $X = 1$ if a house's floor area is above the sample mean and $X=0$ otherwise, and define $Y$ the same way for price.

In the code below, `area > area.mean()` gives `True` or `False` for each house, and `astype(int)` converts these to 1 and 0.

The pandas function `crosstab` then counts the houses with each combination of $X$ and $Y$, and `normalize=True` turns the counts into fractions.

```{code-cell} ipython3
X = (area > area.mean()).astype(int)
Y = (price > price.mean()).astype(int)
joint = pd.crosstab(X, Y, normalize=True)
joint.index.name = 'x (area above mean)'
joint.columns.name = 'y (price above mean)'
joint
```

Each cell is the fraction of houses with that combination of $X$ and $Y$.

Given a joint distribution, we can recover the distribution of $X$ on its own by summing out $Y$:

$$
p_X(x) = \sum_{y \in S_Y} p(x,y)
$$

and symmetrically for $p_Y$.

We call $p_X$ and $p_Y$ the [marginal distributions](https://en.wikipedia.org/wiki/Marginal_distribution) of $X$ and $Y$.

In our table, the marginals are the row and column sums:

```{code-cell} ipython3
p_X, p_Y = joint.sum(axis=1), joint.sum(axis=0)
p_X, p_Y
```

Notice that the marginals alone do not tell us the whole story.

Knowing the fraction of houses with above-mean area and the fraction with above-mean price, separately, says nothing about whether the *same* houses tend to have both.

For that we need the joint distribution.

### Continuous case

Some variables are continuous rather than discrete.

For a continuous pair $(X, Y)$, the joint distribution is described by a joint density $p(x, y)$, which we can picture as a surface over the $(x, y)$ plane.

The probability that $(X, Y)$ lands in a given region is the volume under this surface over that region, just as, for a single variable, a probability is an area under the density.

As in the discrete case, we can recover the density of $X$ on its own, called the marginal density, from the joint density.

```{admonition} Details: joint and marginal densities
:class: dropdown

A joint density is a nonnegative function $p$ on $\mathbb R^2$ with

$$
\int_{-\infty}^\infty \int_{-\infty}^\infty p(x,y) \, dx \, dy = 1
$$

We say that $(X,Y)$ has joint density $p$ if, for every region $A$ in the plane,

$$
\mathbb P\{(X,Y) \in A\} = \iint_A p(x,y) \, dx \, dy
$$

Marginal densities come from integrating out the other variable, in place of the sum in the discrete case:

$$
p_X(x) = \int_{-\infty}^\infty p(x,y) \, dy
$$
```

## Independence and correlation

$X$ and $Y$ are [independent](https://en.wikipedia.org/wiki/Independence_(probability_theory)) if the joint distribution is the product of the marginals:

$$
p(x,y) = p_X(x) \, p_Y(y) \qquad \text{for all } x, y
$$

Independence means that learning the value of $X$ tells us nothing about $Y$.

We have already relied on this idea, informally when we discussed {ref}`the role of independence <observed_distributions:independence>` in {doc}`observed_distributions`, and through the IID assumption in {doc}`mle_intro`.

For example, $X$ and $Y$ in our house table are not independent.

To see this, consider the houses with below-mean floor area and below-mean price, which is the case $x = 0$ and $y = 0$.

Here is $p(0, 0)$, followed by the product of the marginals $p_X(0) \, p_Y(0)$.

```{code-cell} ipython3
joint.loc[0, 0], p_X[0] * p_Y[0]
```

About 48% of houses are in this case, while independence would require about 34%.

So $X$ and $Y$ are not independent, as we would expect, since larger houses tend to be more expensive.

Independence is a yes-or-no property: it does not tell us how strong a relationship is.

To measure the strength and direction of dependence, we use the [covariance](https://en.wikipedia.org/wiki/Covariance)

$$
\mathrm{Cov}(X,Y) = \mathbb E \left[ (X - \mu_X)(Y - \mu_Y) \right]
$$

where $\mu_X = \mathbb E[X]$ and $\mu_Y = \mathbb E[Y]$ are the means of $X$ and $Y$.

The covariance is positive when $X$ and $Y$ tend to be above their means together, and negative when one tends to be above its mean while the other is below.

If $X$ and $Y$ are independent, then $\mathrm{Cov}(X,Y) = 0$.

```{note}
The converse is false: zero covariance does not imply independence.

It only rules out *linear* dependence, as {ref}`bivariate_ex2` shows.
```

The covariance is measured in the units of $X$ times the units of $Y$, which makes it hard to interpret.

We therefore usually standardize it into the [correlation coefficient](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient)

$$
\rho = \mathrm{Corr}(X,Y) = \frac{\mathrm{Cov}(X,Y)}{\sigma_X \sigma_Y}
$$

Correlation is unit-free and always lies in $[-1, 1]$, with the extreme values attained only when $Y$ is an exact linear function of $X$.

To estimate correlation from data $\{(x_i, y_i)\}$, we replace population quantities with sample ones, which gives the sample correlation

$$
r = \frac{1}{n} \sum_{i=1}^n
    \left( \frac{x_i - \bar x}{s_x} \right)
    \left( \frac{y_i - \bar y}{s_y} \right)
$$

where $\bar x$, $\bar y$, $s_x$ and $s_y$ are the sample means and standard deviations, as defined in {doc}`observed_distributions`.

Here is a function that computes $r$ directly from this formula.

(Setting `ddof=0` makes `std` divide by $n$, as in our definition of $s_x$ and $s_y$.)

```{code-cell} ipython3
def sample_corr(x, y):
    "The sample correlation of x and y."
    x_standardized = (x - x.mean()) / x.std(ddof=0)
    y_standardized = (y - y.mean()) / y.std(ddof=0)
    return np.mean(x_standardized * y_standardized)
```

Here is the correlation between floor area and price.

```{code-cell} ipython3
sample_corr(area, price)
```

Libraries also provide this calculation, for example as NumPy's `np.corrcoef`, which you will often see in code written by AI tools.

(bivariate:normal)=
## The bivariate normal distribution

Just as the {ref}`normal distribution <prob_dist:normal>` is the most widely used distribution for a single variable, the [bivariate normal distribution](https://en.wikipedia.org/wiki/Multivariate_normal_distribution) is the most widely used joint distribution.

It has five parameters: the two means $\mu_X, \mu_Y$, the two standard deviations $\sigma_X, \sigma_Y$, and the correlation $\rho \in (-1,1)$ between $X$ and $Y$.

```{admonition} Details: the bivariate normal density
:class: dropdown

The density of the bivariate normal distribution is

$$
p(x,y) = \frac{1}{2\pi \sigma_X \sigma_Y \sqrt{1-\rho^2}}
\exp\left(
-\frac{1}{2(1-\rho^2)}
\left[
\frac{(x-\mu_X)^2}{\sigma_X^2}
- \frac{2\rho (x-\mu_X)(y-\mu_Y)}{\sigma_X \sigma_Y}
+ \frac{(y-\mu_Y)^2}{\sigma_Y^2}
\right]
\right)
$$
```

SciPy provides this distribution as `scipy.stats.multivariate_normal`, which takes the two means and a [covariance matrix](https://en.wikipedia.org/wiki/Covariance_matrix).

The covariance matrix has the variances $\sigma_X^2$ and $\sigma_Y^2$ on its diagonal, and the covariance $\rho \sigma_X \sigma_Y$ in the other two entries.

```{code-cell} ipython3
def bivariate_normal(μ_x, μ_y, σ_x, σ_y, ρ):
    cov = [[σ_x**2, ρ * σ_x * σ_y],
           [ρ * σ_x * σ_y, σ_y**2]]
    return scipy.stats.multivariate_normal([μ_x, μ_y], cov)
```

Here is the density as a surface over the $(x,y)$ plane.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: The bivariate normal density surface
    name: fig:bivariate-normal-surface
tags: [hide-input]
---
x_grid = np.linspace(-3, 3, 100)
y_grid = np.linspace(-3, 3, 100)
X_mesh, Y_mesh = np.meshgrid(x_grid, y_grid)
pos = np.dstack((X_mesh, Y_mesh))

u = bivariate_normal(0, 0, 1, 1, 0.6)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot_surface(X_mesh, Y_mesh, u.pdf(pos), cmap='viridis', linewidth=0)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('density')
plt.show()
```

This is the same kind of bell-shaped hill as the univariate normal density, built over a plane instead of a line.

It is usually more convenient to look straight down at the hill and draw its [contour lines](https://en.wikipedia.org/wiki/Contour_line), the way a topographic map does.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Bivariate normal contours by correlation
    name: fig:bivariate-normal-contours
tags: [hide-input]
---
fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)
for ax, ρ in zip(axes, (-0.8, 0.0, 0.8)):
    u = bivariate_normal(0, 0, 1, 1, ρ)
    ax.contour(X_mesh, Y_mesh, u.pdf(pos), levels=6, cmap='viridis')
    ax.set_title(rf'$\rho={ρ}$')
    ax.set_xlabel('x')
    ax.set_aspect('equal')
axes[0].set_ylabel('y')
plt.show()
```

When $\rho = 0$ the contours are circles, and otherwise they are tilted ellipses.

It can be shown that the marginal of $X$ is $N(\mu_X, \sigma_X^2)$ and the marginal of $Y$ is $N(\mu_Y, \sigma_Y^2)$.

Neither marginal depends on $\rho$: the correlation describes how $X$ and $Y$ move *together*, and that information is lost when we look at either variable alone.

For the bivariate normal, $\rho = 0$ implies that $X$ and $Y$ are independent, which, as noted above, is not true of distributions in general.

## Building joint distributions

Why do two variables end up correlated?

A very common mechanism is that $Y$ is built partly from $X$:

$$
Y = a X + b + W
$$

where $W$ is noise, independent of $X$, with mean zero and standard deviation $\sigma_W$.

The correlation between $X$ and $Y$ is then

$$
\mathrm{Corr}(X,Y) = \frac{a \sigma_X}{\sqrt{a^2 \sigma_X^2 + \sigma_W^2}}
$$

```{admonition} Derivation: the correlation of X and Y
:class: dropdown

Since $W$ has mean zero, $\mu_Y = a \mu_X + b$, so $Y - \mu_Y = a (X - \mu_X) + W$.

Hence

$$
\mathrm{Cov}(X,Y)
= \mathbb E [ (X - \mu_X) (a (X - \mu_X) + W) ]
= a \sigma_X^2 + \mathbb E [ (X - \mu_X) W ]
= a \sigma_X^2
$$

where the last term is zero because $W$ is independent of $X$ and has mean zero.

Similarly,

$$
\mathbb V[Y]
= \mathbb E [ (a (X - \mu_X) + W)^2 ]
= a^2 \sigma_X^2 + \sigma_W^2
$$

since the cross term $2 a \, \mathbb E [ (X - \mu_X) W ]$ is again zero.

Dividing the covariance by $\sigma_X \sigma_Y = \sigma_X \sqrt{a^2 \sigma_X^2 + \sigma_W^2}$ gives the formula.
```

The correlation is driven entirely by the relative size of the signal $a X$ and the noise $W$.

Let's see this in a picture, fixing $a=1$ and $\sigma_X = 1$ and increasing $\sigma_W$.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Correlation falls as noise grows
    name: fig:bivariate-signal-noise
---
rng = np.random.default_rng(1234)
N = 500
a = 1.0

fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)
for ax, σ_w in zip(axes, (0.2, 1.0, 3.0)):
    x = rng.standard_normal(N)
    w = rng.normal(scale=σ_w, size=N)
    y = a * x + w
    ρ = a / np.sqrt(a**2 + σ_w**2)
    ax.scatter(x, y, alpha=0.5, s=10)
    ax.set_title(rf'$\sigma_W={σ_w}$, $\rho={ρ:.2f}$')
    ax.set_xlabel('x')
axes[0].set_ylabel('y')
plt.show()
```

As the noise grows, the cloud of points spreads out and the correlation falls, even though the underlying mechanism never changes.

If $X$ and $W$ are both normal, then the pair $(X,Y)$ built this way is bivariate normal.

## When is a pair jointly normal?

It is tempting to think that, if $X$ and $Y$ are each individually normal, then the pair $(X,Y)$ must be bivariate normal.

This is false.

Let $X$ be standard normal, and construct $Y$ by

$$
Y = \begin{cases} X & \text{if } |X| < 1 \\ -X & \text{if } |X| \geq 1 \end{cases}
$$

Since the standard normal density is symmetric about zero, flipping the sign of $X$ whenever $|X| \geq 1$ does not change its distribution, so $Y$ is also standard normal.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Normal marginals, but not jointly normal
    name: fig:bivariate-not-normal
---
N = 2000
x = rng.standard_normal(N)
y = np.where(np.abs(x) < 1, x, -x)

g = sns.jointplot(x=x, y=y, height=5, alpha=0.4, s=8)
g.set_axis_labels('x', 'y')
plt.show()
```

The histograms on the margins have the familiar bell shape, yet the joint distribution is concentrated on two crossing lines, nothing like the elliptical contours of the bivariate normal.

The lesson is that checking each variable individually is not enough: we need to look at the joint distribution directly.

## Simple regression

We now turn to fitting a line through data.

Let's take a small data set in which $y_i$ is the daily sales of an ice-cream cart in dollars and $x_i$ is the day's temperature in Celsius.

```{code-cell} ipython3
sales = pd.DataFrame({
    'temperature': [32, 21, 24, 35, 10, 11, 22, 21, 27, 2],
    'sales': [2000, 1000, 1500, 2500, 500, 900, 1100, 1500, 1800, 250]
})
x, y = sales['temperature'], sales['sales']
```

We would like to choose a line

$$
\hat y_i = \alpha + \beta x_i
$$

that fits these data well, where $\hat y_i$ is the value of $y$ that the line predicts at $x_i$.

### Fitting by hand

Let's try two guesses, with the distance from each point to the line drawn in red.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Two lines fitted by hand
    name: fig:hand-fits
---
def plot_fit(ax, α, β, label):
    "Plot the sales data, the line and the residuals."
    ax.scatter(x, y)
    x_vals = np.array([x.min(), x.max()])
    ax.plot(x_vals, α + β * x_vals, 'g', lw=2, label=label)
    ax.vlines(x, α + β * x, y, color='r', lw=1)
    ax.set_xlabel('temperature (Celsius)')
    ax.set_ylabel('sales (US$)')
    ax.legend()

fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
plot_fit(axes[0], 5, 10, r'$\alpha=5, \beta=10$')
plot_fit(axes[1], 5, 65, r'$\alpha=5, \beta=65$')
plt.show()
```

The first line is poor and the second looks reasonable.

But which line is best depends on what we mean by "best", so we need a criterion.

### Choosing a criterion

The red segments are the [residuals](https://en.wikipedia.org/wiki/Errors_and_residuals)

$$
e_i = y_i - \alpha - \beta x_i
$$

and a good line should make them small.

The most common criterion, called [ordinary least squares](https://en.wikipedia.org/wiki/Ordinary_least_squares) (OLS), minimizes the sum of squared residuals

$$
C(\alpha, \beta) = \sum_{i=1}^n (y_i - \alpha - \beta x_i)^2
$$

### Optimizing

The values of $\alpha$ and $\beta$ that minimize $C$ are

$$
\hat \beta
    = \frac{\sum_{i=1}^n (x_i - \bar x)(y_i - \bar y)}{\sum_{i=1}^n (x_i - \bar x)^2}
    = r \, \frac{s_y}{s_x}
\qquad \text{and} \qquad
\hat \alpha = \bar y - \hat \beta \bar x
$$

```{admonition} Derivation: the OLS formulas
:class: dropdown

Setting the derivative of $C$ with respect to $\alpha$ to zero and dividing by $-2$ gives

$$
\sum_{i=1}^n (y_i - \alpha - \beta x_i) = 0
$$

Dividing by $n$ and rearranging gives $\alpha = \bar y - \beta \bar x$.

Setting the derivative with respect to $\beta$ to zero and dividing by $-2$ gives

$$
\sum_{i=1}^n x_i (y_i - \alpha - \beta x_i) = 0
$$

Substituting for $\alpha$ gives

$$
\sum_{i=1}^n x_i (y_i - \bar y) = \beta \sum_{i=1}^n x_i (x_i - \bar x)
$$

Since $\sum_i (y_i - \bar y) = 0$ and $\sum_i (x_i - \bar x) = 0$, we can subtract $\bar x$ from $x_i$ on both sides without changing either sum, which gives the formula for $\hat \beta$.

Dividing the numerator and denominator by $n$ shows that $\hat \beta = r s_x s_y / s_x^2 = r s_y / s_x$.
```

So the slope is the sample correlation $r$, rescaled by the units of the two variables.

```{code-cell} ipython3
r = sample_corr(x, y)
β_hat = r * y.std(ddof=0) / x.std(ddof=0)
α_hat = y.mean() - β_hat * x.mean()
α_hat, β_hat
```

NumPy's `np.polyfit(x, y, 1)` gives the same answer.

It fits a polynomial of degree 1, which is a straight line, to the points $(x_i, y_i)$, and returns the slope first and then the intercept.

```{code-cell} ipython3
np.polyfit(x, y, 1)
```

Let's check this visually, by computing the sum of squared residuals $C(\alpha, \beta)$ on a grid of values and drawing its contour lines.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Contours of the sum of squared residuals
    name: fig:ssr-contours
---
def ssr(α, β):
    return ((y - α - β * x)**2).sum()

α_vals = np.linspace(-500, 500, 200)
β_vals = np.linspace(40, 90, 200)
C_values = np.array([[ssr(α, β) for α in α_vals] for β in β_vals]) / 1000

# draw contours at 5%, 20%, 50%, ... above the minimum value
levels = ssr(α_hat, β_hat) / 1000 * np.array([1.05, 1.2, 1.5, 2, 3, 5, 8])

fig, ax = plt.subplots()
filled = ax.contourf(α_vals, β_vals, C_values, levels=levels,
                     cmap='viridis_r', extend='both', alpha=0.8)
lines = ax.contour(α_vals, β_vals, C_values, levels=levels,
                   colors='black', linewidths=1)
ax.clabel(lines, fmt='%.0f', fontsize=9)
fig.colorbar(filled, ax=ax, label='sum of squared residuals (thousands)')
ax.plot(α_hat, β_hat, 'ko', label=r'$(\hat \alpha, \hat \beta)$')
ax.set_xlabel(r'$\alpha$')
ax.set_ylabel(r'$\beta$')
ax.legend()
plt.show()
```

The contours form nested ellipses around $(\hat \alpha, \hat \beta)$, so the sum of squared residuals rises as we move away from it in any direction.

The ellipses are long and tilted because a larger intercept can be partly offset by a smaller slope, so many lines fit the data almost equally well along that direction.

### Regression and the conditional mean

The [conditional mean](https://en.wikipedia.org/wiki/Conditional_expectation) $\mathbb E[Y \mid X = x]$ is the average value of $Y$ among the cases where $X = x$, and it is our best guess of $Y$ when we know that $X = x$.

For a {ref}`bivariate normal <bivariate:normal>` pair, the conditional mean is a straight line in $x$:

$$
\mathbb E[Y \mid X=x] = \mu_Y + \rho \frac{\sigma_Y}{\sigma_X} (x - \mu_X)
$$

This has the same form as the OLS line, with $\hat \alpha + \hat \beta x = \bar y + r (s_y / s_x)(x - \bar x)$.

So OLS estimates the conditional mean when the data are bivariate normal, and it remains a useful summary of a linear relationship when they are not.

Let's fit a line to the house data in logs, which makes both variables closer to normal, as we saw for prices in {numref}`fig:log-price-hist`.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Log price against log area, with the OLS line
    name: fig:log-ames-ols
---
log_area, log_price = np.log(area), np.log(price)
slope, intercept = np.polyfit(log_area, log_price, 1)
x_vals = np.array([log_area.min(), log_area.max()])

fig, ax = plt.subplots()
ax.scatter(log_area, log_price, alpha=0.2, s=8)
ax.plot(x_vals, intercept + slope * x_vals, 'k--', lw=2, label='OLS line')
ax.set_xlabel('log(living area)')
ax.set_ylabel('log(sale price)')
ax.legend()
plt.show()
```

### Least squares and maximum likelihood

In {doc}`mle_intro` we chose parameters by maximizing the likelihood, so it is natural to ask how least squares relates to that approach.

Suppose that

$$
y_i = \alpha + \beta x_i + \epsilon_i
$$

where the errors $\epsilon_i$ are IID and normal, with mean zero and standard deviation $\sigma$.

Then each $y_i$ is normal with mean $\alpha + \beta x_i$, and replacing $x_i - \mu$ by $y_i - \alpha - \beta x_i$ in the normal log likelihood {eq}`eq:normal_loglik` gives

$$
\ell(\alpha, \beta, \sigma)
= - \frac{n}{2} \ln (2 \pi) - n \ln \sigma
  - \frac{1}{2 \sigma^2} \sum_{i=1}^n (y_i - \alpha - \beta x_i)^2
$$

Only the last term depends on $\alpha$ and $\beta$, so, whatever the value of $\sigma$, maximizing $\ell$ over $\alpha$ and $\beta$ is the same as minimizing the sum of squared residuals $C(\alpha, \beta)$.

So, when the errors are normal, ordinary least squares *is* maximum likelihood.

### Other criteria

Least squares is not the only reasonable criterion.

Squaring means that a large residual counts very heavily, so a single unusual observation, called an [outlier](https://en.wikipedia.org/wiki/Outlier), can pull the whole line towards it.

An alternative is [least absolute deviations](https://en.wikipedia.org/wiki/Least_absolute_deviations) (LAD), which minimizes

$$
\sum_{i=1}^n | y_i - \alpha - \beta x_i |
$$

This criterion has no simple formula for its minimizer, so we minimize it numerically with SciPy's `minimize` function, as we did when {ref}`fitting the gamma distribution <mle_numerical>` in {doc}`mle_intro`.

To compare the two, suppose that we add one unusual day to the ice-cream data: a hot day when the cart broke down and sales were only 100 dollars.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: OLS and LAD lines with an outlier
    name: fig:ols-vs-lad
---
x_new = np.append(x, 30)
y_new = np.append(y, 100)

def lad(p):
    α, β = p
    return np.abs(y_new - α - β * x_new).sum()

result = minimize(lad, x0=[0, 50], method='Nelder-Mead')
α_lad, β_lad = result.x
β_ols, α_ols = np.polyfit(x_new, y_new, 1)

x_vals = np.array([x_new.min(), x_new.max()])

fig, ax = plt.subplots()
ax.scatter(x_new, y_new)
ax.scatter(30, 100, color='r', label='outlier')
ax.plot(x_vals, α_ols + β_ols * x_vals, lw=2, label='OLS')
ax.plot(x_vals, α_lad + β_lad * x_vals, lw=2, label='LAD')
ax.set_xlabel('temperature (Celsius)')
ax.set_ylabel('sales (US$)')
ax.legend()
plt.show()
```

```{code-cell} ipython3
β_ols, β_lad
```

One bad day pulls the OLS slope down from about 64 to about 49, while the LAD slope stays at about 62.5, as it was without the outlier.

This is the same contrast as between the mean and the {ref}`median <observed_distributions:quantiles>` in {doc}`observed_distributions`, and indeed LAD estimates the conditional median rather than the conditional mean.

## An extended example

Let's use regression on a question of policy interest: how does life expectancy vary with income across countries?

The data come from [Our World in Data](https://ourworldindata.org/grapher/life-expectancy-vs-gdp-per-capita), and we use 2018, the most recent year in which this data set has both series.

```{code-cell} ipython3
url = ('https://github.com/QuantEcon/data-lectures/raw/main/'
       'lectures/life-expectancy-vs-gdp-per-capita.csv')
columns = ['Code', 'Year', 'Life expectancy at birth (historical)',
           'GDP per capita']
gdp_life = pd.read_csv(url, usecols=columns)[columns].dropna()
gdp_life.columns = ['code', 'year', 'life_expectancy', 'gdppc']

# keep 2018, and drop the world aggregate, which is not a country
gdp_life = gdp_life[(gdp_life['year'] == 2018)
                    & ~gdp_life['code'].str.startswith('OWID')]
gdp_life['log_gdppc'] = np.log10(gdp_life['gdppc'])
len(gdp_life)
```

Here are the data, with GDP per capita on both its original scale and a log scale.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Life expectancy against GDP per capita, 2018
    name: fig:preston-curve
---
slope, intercept = np.polyfit(gdp_life['log_gdppc'],
                              gdp_life['life_expectancy'], 1)
x_vals = np.array([gdp_life['log_gdppc'].min(), gdp_life['log_gdppc'].max()])

fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
axes[0].scatter(gdp_life['gdppc'], gdp_life['life_expectancy'], s=12, alpha=0.6)
axes[0].set_xticks(np.arange(0, 200_000, 50_000))
axes[0].set_xlabel('GDP per capita')
axes[0].set_ylabel('life expectancy (years)')
axes[1].scatter(gdp_life['log_gdppc'], gdp_life['life_expectancy'],
                s=12, alpha=0.6)
axes[1].plot(x_vals, intercept + slope * x_vals, 'k--', lw=2)
axes[1].set_xlabel('log10 of GDP per capita')
plt.show()
```

On the original scale the relationship is strongly curved, a pattern known as the [Preston curve](https://en.wikipedia.org/wiki/Preston_curve).

Taking logs makes it close to linear, so a straight line is a much better summary.

The correlation rises accordingly.

```{code-cell} ipython3
sample_corr(gdp_life['gdppc'], gdp_life['life_expectancy']), \
sample_corr(gdp_life['log_gdppc'], gdp_life['life_expectancy'])
```

```{code-cell} ipython3
slope
```

Since we used base 10 logs, the slope says that a country with ten times the GDP per capita has, on average, about 13 more years of life expectancy.

Notice how carefully that sentence is worded, since it describes a pattern across countries and not the effect of raising a country's income.

## Correlation and causality

Suppose we find that, across days, ice-cream sales and drownings at the beach rise and fall together.

It would be odd to ban ice cream to prevent drownings.

Both rise in hot weather, when more people buy ice cream and more people go swimming.

A third factor that affects both variables like this is called a [confounder](https://en.wikipedia.org/wiki/Confounding), and it can produce a [correlation without any causal link](https://en.wikipedia.org/wiki/Correlation_does_not_imply_causation) between the two.

Confounders are not always this easy to spot.

Suppose a survey finds that women who own horses live longer than women who don't.

Owning a horse is expensive, so horse owners are likely to be wealthier and to have better access to health care, and those may be what matters.

Here is a real example, using data on the 50 US states and the District of Columbia (DC).

For each one we have Donald Trump's percentage share of the 2024 presidential vote, from the Federal Election Commission, and life expectancy at birth in 2021, from the US National Center for Health Statistics.

```{code-cell} ipython3
url = ('https://raw.githubusercontent.com/jstac/py_for_ds_public/main/lectures/'
       '_static/lecture_specific/bivariate_dist/state_votes_life_expectancy.csv')
states = pd.read_csv(url)
states.head()
```

Let's plot the data.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Life expectancy against Trump vote share, by state
    name: fig:votes-longevity-all
---
fig, ax = plt.subplots()
ax.scatter(states['trump_share'], states['life_expectancy'], alpha=0.7)
for _, row in states[states['code'].isin(['DC', 'HI', 'MS'])].iterrows():
    ax.annotate(row['code'], (row['trump_share'], row['life_expectancy']),
                xytext=(5, 5), textcoords='offset points')
ax.margins(y=0.1)
ax.set_xlabel('Trump share of the 2024 vote (%)')
ax.set_ylabel('life expectancy in 2021 (years)')
plt.show()
```

DC sits far to the left of everything else, and it is a single city rather than a state, so let's see what happens without it.

```{code-cell} ipython3
no_dc = states[states['code'] != 'DC']

r_all = sample_corr(states['trump_share'], states['life_expectancy'])
r_no_dc = sample_corr(no_dc['trump_share'], no_dc['life_expectancy'])
slope, intercept = np.polyfit(no_dc['trump_share'], no_dc['life_expectancy'], 1)

r_all, r_no_dc, slope
```

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: The same data without DC, with the OLS line
    name: fig:votes-longevity-fit
---
x_vals = np.array([no_dc['trump_share'].min(), no_dc['trump_share'].max()])

fig, ax = plt.subplots()
ax.scatter(no_dc['trump_share'], no_dc['life_expectancy'], alpha=0.7)
ax.plot(x_vals, intercept + slope * x_vals, 'k--', lw=2)
ax.set_xlabel('Trump share of the 2024 vote (%)')
ax.set_ylabel('life expectancy in 2021 (years)')
plt.show()
```

The correlation is about $-0.53$ with DC and $-0.66$ without it, so this one choice about the data matters, and any report should state it.

Without DC, states with a vote share 10 percentage points higher have life expectancy about 1.6 years lower, on average.

What can we conclude from this?

Not that voting one way or the other affects how long people live, since these data cannot tell us that.

Many factors differ across states and are related to both variables, including income, education, age, smoking, health insurance and region.

The year 2021 also falls in the pandemic, whose effects differed across states.

Moreover, each point is a whole state, so the figure says nothing directly about individuals, and in particular it does not show that any individual's vote is related to their lifespan.

Drawing conclusions about individuals from group data like this is called the [ecological fallacy](https://en.wikipedia.org/wiki/Ecological_fallacy).

Learning about causes takes more than a strong correlation: it takes a research design, such as a [randomized controlled trial](https://en.wikipedia.org/wiki/Randomized_controlled_trial), that rules out explanations like these.

As discussed in {doc}`overview`, some policy questions need only a good prediction, while others need causal evidence of this kind {cite}`kleinberg2015prediction`.

## Exercises

```{exercise-start}
:label: bivariate_ex1
```

You ask an AI tool how much life expectancy rises with GDP per capita, using the `gdp_life` data above, and it returns the code below.

```{code-block} python3
slope, intercept = np.polyfit(gdp_life['life_expectancy'],
                              gdp_life['log_gdppc'], 1)
print(slope)
```

It then reports that a tenfold rise in GDP per capita goes with about 0.056 more years of life expectancy.

The code runs without an error, but what is wrong?

Is the slope of one regression simply the inverse of the slope of the other?

```{exercise-end}
```

```{solution-start} bivariate_ex1
:class: dropdown
```

The arguments are the wrong way around: `np.polyfit` takes $x$ first, so this code regresses log GDP per capita on life expectancy.

```{code-cell} ipython3
wrong, _ = np.polyfit(gdp_life['life_expectancy'], gdp_life['log_gdppc'], 1)
right, _ = np.polyfit(gdp_life['log_gdppc'], gdp_life['life_expectancy'], 1)
wrong, right, 1 / right
```

The slope that the AI tool found is in units of log GDP per capita per year of life expectancy, so its statement is wrong.

The two slopes are also not inverses of each other, because the two regressions minimize different things: vertical distances in one case and horizontal distances in the other.

In fact their product is $r^2$, so they would be inverses only if the points lay exactly on a line.

```{solution-end}
```

```{exercise-start}
:label: bivariate_ex2
```

Let $X$ be standard normal and let $Y = X^2$.

Simulate a large sample, plot it and compute the sample correlation.

Are $X$ and $Y$ independent?

What does this tell us about reading a correlation close to zero?

```{exercise-end}
```

```{solution-start} bivariate_ex2
:class: dropdown
```

```{code-cell} ipython3
x_sim = rng.standard_normal(10_000)
y_sim = x_sim**2

fig, ax = plt.subplots()
ax.scatter(x_sim, y_sim, alpha=0.2, s=5)
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()

sample_corr(x_sim, y_sim)
```

The correlation is close to zero, and indeed the population covariance is $\mathbb E[X^3] = 0$.

Yet $X$ and $Y$ are far from independent, since $X$ determines $Y$ exactly.

Correlation only measures linear dependence, so we should always look at the data as well as the number.

```{solution-end}
```

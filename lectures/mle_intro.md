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

# Maximum Likelihood Estimation

```{index} single: Maximum Likelihood Estimation
```

In {doc}`observed_distributions` we fitted distributions to data by the method of moments, choosing parameters so that the fitted distribution reproduces the mean and variance of the data.

In this lecture we study a second approach, called [maximum likelihood estimation](https://en.wikipedia.org/wiki/Maximum_likelihood_estimation), which is the standard method for choosing the parameters of a statistical model.

The idea is to choose the parameter values under which the data we actually observed are most probable.

```{note}
Regarding the history of the idea, [Carl Friedrich Gauss](https://en.wikipedia.org/wiki/Carl_Friedrich_Gauss) used a version of it in 1809, and [Ronald Fisher](https://en.wikipedia.org/wiki/Ronald_Fisher) developed it systematically between 1912 and 1922 {cite}`fisher1922mathematical`.

See {cite:t}`aldrich1997fisher` and {cite:t}`stigler2007epic` for more details.
```

In this lecture we

* define the likelihood function and maximize it, using calculus when a formula for the maximizer exists and numerical methods when it does not,
* see why we work with the logarithm of the likelihood,
* compare maximum likelihood with the method of moments for the Pareto distribution, whose heavy tail causes trouble for the method of moments, and
* check by simulation that maximum likelihood estimates approach the true value as the sample grows.

Mathematical derivations are placed in dropdown boxes, which you can open if you want the details.

We use the following imports.

```{code-cell} ipython3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
from scipy.optimize import minimize

np.set_printoptions(legacy='1.25')   # print scalars as plain numbers
```

## The likelihood function

We start with an example that we meet again in {doc}`bayes_intro`.

### The model

A lender makes a series of small loans, each of which is either repaid or not.

We write $Y_i = 1$ if loan $i$ defaults and $Y_i = 0$ if it is repaid.

We assume that the outcomes are [independent and identically distributed](https://en.wikipedia.org/wiki/Independent_and_identically_distributed_random_variables) (IID) draws from the {doc}`Bernoulli distribution <prob_dist>` with

$$
\mathbb{P}\{Y_i = 1\} = \theta
\quad \text{and} \quad
\mathbb{P}\{Y_i = 0\} = 1 - \theta
$$

The default probability $\theta$ is unknown, and our job is to estimate it from the observed outcomes $y_1, \ldots, y_n$.

Both cases can be written as the single expression

$$
\mathbb{P}\{Y_i = y\} = \theta^y (1 - \theta)^{1 - y}
\quad \text{for} \quad y \in \{0, 1\}
$$

(To check this, set $y = 1$ and then $y = 0$.)

### The likelihood

Since the outcomes are independent, the probability of observing the whole sample $y_1, \ldots, y_n$ is the product of the individual probabilities:

$$
L(\theta) = \prod_{i=1}^n \theta^{y_i} (1 - \theta)^{1 - y_i}
$$ (eq:bern_lik)

The function $L$ is called the [likelihood function](https://en.wikipedia.org/wiki/Likelihood_function).

It holds the observed data fixed and treats the probability of those data as a function of the parameter.

The maximum likelihood estimate $\hat \theta$ is the value of $\theta$ that maximizes $L(\theta)$.

In other words, it is the default probability under which the outcomes we actually observed are most probable.

### Plotting the likelihood

Let's simulate some data, drawing 20 loan outcomes from the Bernoulli distribution with a true default probability of 0.15.

```{code-cell} ipython3
rng = np.random.default_rng(14)
θ_true = 0.15                     # the "unknown" true value
n = 20
y = scipy.stats.bernoulli(θ_true).rvs(n, random_state=rng)
y
```

Here is a function that computes the likelihood {eq}`eq:bern_lik` directly from its definition as a product.

```{code-cell} ipython3
def likelihood(θ, y):
    return np.prod(θ**y * (1 - θ)**(1 - y))
```

Now we evaluate it on a grid of values for $\theta$ and plot the result.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Likelihood function for the default probability
    name: fig:mle-bernoulli-lik
---
θ_grid = np.linspace(0.001, 0.999, 500)
L_values = [likelihood(θ, y) for θ in θ_grid]

fig, ax = plt.subplots()
ax.plot(θ_grid, L_values, lw=2)
ax.set_xlabel(r'$\theta$')
ax.set_ylabel('likelihood')
plt.show()
```

The grid point with the largest likelihood approximates the maximum likelihood estimate.

```{code-cell} ipython3
θ_grid[np.argmax(L_values)]
```

This is very close to the fraction of loans that defaulted.

```{code-cell} ipython3
y.mean()
```

In fact the maximum likelihood estimate is exactly the fraction that defaulted, $\hat \theta = k / n$, where $k$ is the number of defaults, as we show {ref}`below <mle_log_likelihood>`.

Notice that the estimate is not equal to the true value 0.15, since with only 20 loans the fraction that defaulted is a noisy guide to $\theta$.

### Likelihood is not probability

It is important to be clear about what $L$ is.

For a fixed $\theta$, the expression {eq}`eq:bern_lik` is the probability of one particular data set.

As a function of $\theta$, with the data held fixed, it is not a probability distribution over $\theta$, and the area under it need not equal one.

That is why it has a different name.

(We return to probabilities over parameters in {doc}`bayes_intro`.)

### More data

Let's see what happens to the likelihood as we observe more loans.

Recall that $k$ is the number of defaults among the $n$ loans.

Each default contributes a factor $\theta$ to the product {eq}`eq:bern_lik`, and each repayment contributes $1 - \theta$, so

$$
L(\theta) = \theta^k (1 - \theta)^{n - k}
$$

We draw samples of 20, 100 and 500 loans and plot the likelihood for each one.

The curves have very different heights, so each one gets its own panel and vertical scale.

In each panel, the point where the likelihood is highest is the maximum likelihood estimate $\hat \theta = k / n$ for that sample, marked with a dot.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Likelihood functions for increasing sample sizes
    name: fig:mle-bernoulli-lik-n
---
fig, axes = plt.subplots(3, 1, sharex=True, figsize=(7, 7))
for ax, n in zip(axes, (20, 100, 500)):
    y = scipy.stats.bernoulli(θ_true).rvs(n, random_state=rng)
    k = y.sum()
    θ_hat = k / n
    ax.plot(θ_grid, θ_grid**k * (1 - θ_grid)**(n - k), lw=2)
    ax.plot(θ_hat, θ_hat**k * (1 - θ_hat)**(n - k), 'ko',
            label=r'$\hat \theta = k / n$')
    ax.axvline(θ_true, color='k', ls=':', lw=1, label=r'true $\theta$')
    ax.set_title(rf'$n = {n}$, $\hat \theta = {θ_hat:.2f}$')
    ax.set_ylabel('likelihood')
axes[0].legend()
axes[-1].set_xlim(0, 0.5)
axes[-1].set_xlabel(r'$\theta$')
plt.tight_layout()
plt.show()
```

As $n$ grows, the estimate $\hat \theta$ moves towards the true value, and the likelihood concentrates around it, since fewer values of $\theta$ are consistent with what we observe.

Notice also the vertical scales: the likelihood shrinks rapidly as $n$ grows, since it is a product of $n$ numbers less than one, a point we return to {ref}`below <mle_numerical>`.

(mle_log_likelihood)=
## The log likelihood

In this section we introduce the logarithm of the likelihood, which makes the likelihood much easier to maximize with calculus.

We will see in the {ref}`next section <mle_numerical>` that it is also essential for computation.

The log likelihood is

$$
\ell(\theta) = \ln L(\theta)
$$

Since the logarithm is strictly increasing, $L$ and $\ell$ have the same maximizer, so we can maximize $\ell$ instead of $L$.

The advantage is that the logarithm turns products into sums.

In general, if $x_1, \ldots, x_n$ are IID draws from a distribution with density or probability mass function $f(x; \theta)$, then

$$
\ell(\theta) = \sum_{i=1}^n \ln f(x_i; \theta)
$$

For the loan example, maximizing this sum with calculus confirms that $\hat \theta = k / n$.

```{admonition} Derivation: the loan example
:class: dropdown

Using $\ln (ab) = \ln a + \ln b$ and $\ln (a^b) = b \ln a$, the log likelihood is

$$
\ell(\theta)
= \sum_{i=1}^n \left[ y_i \ln \theta + (1 - y_i) \ln (1 - \theta) \right]
= k \ln \theta + (n - k) \ln (1 - \theta)
$$

where $k = \sum_i y_i$ is the number of defaults.

Its derivative is

$$
\ell'(\theta) = \frac{k}{\theta} - \frac{n - k}{1 - \theta}
$$

Assuming $0 < k < n$, setting the derivative to zero and multiplying through by $\theta (1 - \theta)$ gives $k - n \theta = 0$, so $\theta = k / n$.

This is a maximum, since the second derivative
$- k / \theta^2 - (n - k) / (1 - \theta)^2$ is negative everywhere on $(0, 1)$.
```

### The normal distribution

For a normal distribution with mean $\mu$ and standard deviation $\sigma$, the log likelihood is

$$
\ell(\mu, \sigma)
= - \frac{n}{2} \ln (2 \pi) - n \ln \sigma
  - \frac{1}{2 \sigma^2} \sum_{i=1}^n (x_i - \mu)^2
$$ (eq:normal_loglik)

It is maximized by the sample mean and the sample standard deviation:

$$
\hat \mu = \frac{1}{n} \sum_{i=1}^n x_i
\quad \text{and} \quad
\hat \sigma = \left( \frac{1}{n} \sum_{i=1}^n (x_i - \hat \mu)^2 \right)^{1/2}
$$

```{admonition} Derivation: the normal distribution
:class: dropdown

The density of the normal distribution is

$$
f(x; \mu, \sigma)
= \frac{1}{\sigma \sqrt{2 \pi}}
  \exp \left( - \frac{(x - \mu)^2}{2 \sigma^2} \right)
$$

Taking logs and summing over the observations gives {eq}`eq:normal_loglik`.

Only the last term depends on $\mu$, and setting its derivative with respect to $\mu$ to zero gives $\sum_i (x_i - \mu) = 0$, so $\hat \mu$ is the sample mean.

Setting the derivative with respect to $\sigma$ to zero gives

$$
- \frac{n}{\sigma} + \frac{1}{\sigma^3} \sum_{i=1}^n (x_i - \hat \mu)^2 = 0
$$

and solving for $\sigma$ gives $\hat \sigma$.
```

These are exactly the method of moments estimates from {prf:ref}`obs_mom_normal`, so for the normal distribution the two methods agree.

The same is true of the loan example, since the mean of the Bernoulli distribution is $\theta$, and matching it to the sample mean also gives $k / n$.

The two methods do not always agree, as we see {ref}`below <mle_vs_mom>`.

Let's look at the log likelihood for the heights of US adult women, which we fitted with a normal distribution in {doc}`prob_dist`.

```{code-cell} ipython3
url = ('https://github.com/QuantEcon/data-lectures/raw/main/'
       'lectures/us_adult_heights.csv')
heights = pd.read_csv(url)
female = heights[heights['sex'] == 'female']['height_cm'].to_numpy()
```

With two parameters, the log likelihood is a surface over the $(\mu, \sigma)$ plane, which we display with contour lines, as in a topographic map.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Log likelihood for the normal distribution
    name: fig:mle-normal-contour
---
def normal_log_likelihood(μ, σ, x):
    n = len(x)
    return (- n / 2 * np.log(2 * np.pi) - n * np.log(σ)
            - ((x - μ)**2).sum() / (2 * σ**2))

μ_hat, σ_hat = female.mean(), female.std()   # NumPy's std divides by n

μ_grid = np.linspace(μ_hat - 1.5, μ_hat + 1.5, 100)
σ_grid = np.linspace(σ_hat - 1.2, σ_hat + 1.2, 100)
ℓ_values = [[normal_log_likelihood(μ, σ, female) for μ in μ_grid]
            for σ in σ_grid]

fig, ax = plt.subplots()
ax.contour(μ_grid, σ_grid, ℓ_values, levels=20, cmap='viridis')
ax.plot(μ_hat, σ_hat, 'ko', label='maximum')
ax.set_xlabel(r'$\mu$')
ax.set_ylabel(r'$\sigma$')
ax.legend()
plt.show()
```

The log likelihood is highest at $(\hat \mu, \hat \sigma)$, marked by the black dot.

Here are the values.

```{code-cell} ipython3
μ_hat, σ_hat
```

The formula {eq}`eq:normal_loglik` returns in {doc}`bivariate_dist`, where it connects maximum likelihood to regression.

(mle_numerical)=
## Numerical maximization

In the examples so far we could find the maximizer with calculus (see the dropdown boxes for details).

For many distributions this is not possible, and we have to maximize the log likelihood numerically.

The grid search that we used for the loan example is one numerical method, but it becomes expensive with several parameters, since the number of grid points multiplies with each one.

Instead we use an optimization routine, which searches for the maximum by moving uphill from a starting point.

As an example, let's fit the gamma distribution to the Ames house prices, which we fitted by the method of moments in {doc}`observed_distributions`.

```{code-cell} ipython3
url = ('https://github.com/QuantEcon/data-lectures/raw/main/'
       'lectures/ames_house_prices.csv')
houses = pd.read_csv(url)
price = houses['price'].to_numpy() / 1_000    # price in thousands of dollars
```

SciPy describes the gamma distribution by a shape $a$ and a scale $s$, which correspond to $\alpha$ and $1 / \beta$ in our notation from {doc}`prob_dist`.

### Why the log likelihood is essential

For this data set the log likelihood is not just convenient but essential.

The likelihood is a product of 2,930 densities, one for each house, and this product is too small for a computer to represent.

Here is the likelihood at $a = 6$ and $s = 30$, which is close to the best fit.

```{code-cell} ipython3
u = scipy.stats.gamma(6, scale=30)
np.prod(u.pdf(price))
```

The true value is positive, but it is smaller than the smallest positive number the computer can store, so it is rounded to zero, an event called [underflow](https://en.wikipedia.org/wiki/Arithmetic_underflow).

The log likelihood, which is a sum rather than a product, has no such problem.

```{code-cell} ipython3
np.sum(u.logpdf(price))
```

For this reason, numerical routines for maximum likelihood work with the log likelihood.

### Fitting the gamma distribution

Most numerical routines minimize rather than maximize, so we minimize the *negative* of the log likelihood, using SciPy's `minimize` function.

Both parameters must be positive, so we search over their logarithms, which can take any value.

```{code-cell} ipython3
def gamma_neg_log_likelihood(params, x):
    a, s = np.exp(params)
    return -np.sum(scipy.stats.gamma.logpdf(x, a, scale=s))
```

The routine needs a starting point, and the method of moments estimates from {prf:ref}`obs_mom_gamma` are a natural choice.

```{code-cell} ipython3
m, v = price.mean(), price.var()
a_mom, s_mom = m**2 / v, v / m

result = minimize(gamma_neg_log_likelihood, x0=np.log([a_mom, s_mom]),
                  args=(price,))
a_mle, s_mle = np.exp(result.x)
a_mle, s_mle
```

SciPy's distributions also have a `fit` method that does the same computation, which we can use as a check.

By default, `fit` also estimates a third parameter, called the location, which shifts the whole distribution to the left or right.

Setting `floc=0` fixes the location at zero, so that the distribution starts at zero, as in {doc}`prob_dist`.

```{code-cell} ipython3
a_fit, _, s_fit = scipy.stats.gamma.fit(price, floc=0)
a_fit, s_fit
```

The answers agree.

Here are the two fitted densities.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Gamma densities fitted to house prices
    name: fig:mle-gamma-prices
---
u_mom = scipy.stats.gamma(a_mom, scale=s_mom)
u_mle = scipy.stats.gamma(a_mle, scale=s_mle)

x_grid = np.linspace(0, price.max(), 400)
fig, ax = plt.subplots()
ax.hist(price, bins=50, density=True, alpha=0.25)
ax.plot(x_grid, u_mom.pdf(x_grid), lw=2, label='method of moments')
ax.plot(x_grid, u_mle.pdf(x_grid), lw=2, label='maximum likelihood')
ax.set_xlabel('sale price (thousands of US$)')
ax.set_ylabel('density')
ax.legend()
plt.show()
```

The two fits are similar here, but the two methods can differ much more when the tails are heavy, as we see {ref}`below <mle_vs_mom>`.

(mle_misspecification)=
### Misspecification

Maximum likelihood finds the best member of the family we choose, but it cannot tell us whether we chose a good family.

If the family does not contain the distribution that generated the data, the model is said to be [misspecified](https://en.wikipedia.org/wiki/Statistical_model_specification).

In that case the fitted distribution differs from the data in some systematic way, whatever parameters we choose.

The house price data give an example.

The gamma fit above is the best that the gamma family can do, so any remaining mismatch with the data comes from the family, not from how we chose the parameters.

To see the mismatch, we compare this fit with a lognormal distribution, also fitted by maximum likelihood, using the Q-Q plot function from {doc}`observed_distributions`.

```{code-cell} ipython3
def qq_plot(sample, u, ax, **kwargs):
    "Plot sample quantiles against the quantiles of the distribution u."
    x_sorted = np.sort(sample)
    n = len(x_sorted)
    τ = (np.arange(1, n+1) - 0.5) / n
    ax.plot(u.ppf(τ), x_sorted, '.', ms=3, alpha=0.6, **kwargs)
    lo, hi = u.ppf(τ[0]), u.ppf(τ[-1])
    ax.plot([lo, hi], [lo, hi], 'k--', lw=2)
    ax.set_xlabel('fitted quantiles')
    ax.set_ylabel('sample quantiles')
```

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: House prices against two families fitted by maximum likelihood
    name: fig:mle-qq-families
---
σ_fit, _, scale_fit = scipy.stats.lognorm.fit(price, floc=0)
u_lognormal = scipy.stats.lognorm(σ_fit, scale=scale_fit)

fig, axes = plt.subplots(1, 2, sharey=True, figsize=(10, 4))
for ax, u, title in zip(axes, (u_mle, u_lognormal), ('gamma', 'lognormal')):
    qq_plot(price, u, ax)
    ax.set_title(title)
axes[1].set_ylabel('')
plt.tight_layout()
plt.show()
```

In the gamma panel the points rise well above the line at the top, so even the best gamma distribution has a right tail that is too short for house prices.

The lognormal fits better, although the very largest prices still lie above its line.

This is the same ranking that we found in {doc}`observed_distributions` using the method of moments, which confirms that the problem lies with the family rather than the estimation method.

In practice, no simple family describes real data exactly, so some misspecification is unavoidable.

The question is whether it matters for the purpose at hand: here it matters if we care about the most expensive houses, and much less if we care about typical ones.


(mle_vs_mom)=
## Maximum likelihood and the method of moments

For the loan and normal examples, the two methods give the same answer.

In this section we look at a case where they differ, and where the difference matters: estimating the tail index of the {doc}`Pareto distribution <prob_dist>`.

Heavy tails are common in economic data, for example in the distributions of wealth and firm size, which we study in {doc}`heavy_tails`.

### Two estimates of the tail index

We assume that the minimum value of the Pareto distribution is known and equal to one, so that the only unknown parameter is the tail index $\alpha$.

Maximizing the log likelihood gives

$$
\hat \alpha_{\text{MLE}} = \frac{n}{\sum_{i=1}^n \ln x_i}
$$ (eq:pareto_mle)

```{admonition} Derivation: the Pareto tail index
:class: dropdown

With minimum value one, the Pareto density is $f(x; \alpha) = \alpha / x^{\alpha + 1}$ for $x \geq 1$, so

$$
\ell(\alpha) = n \ln \alpha - (\alpha + 1) \sum_{i=1}^n \ln x_i
$$

Setting the derivative to zero gives

$$
\frac{n}{\alpha} - \sum_{i=1}^n \ln x_i = 0
$$

and solving for $\alpha$ gives {eq}`eq:pareto_mle`.
```

For the method of moments, recall from {doc}`prob_dist` that the mean of this distribution is $\alpha / (\alpha - 1)$, provided that $\alpha > 1$.

Setting this equal to the sample mean $\bar x$ and solving for $\alpha$ gives

$$
\hat \alpha_{\text{MM}} = \frac{\bar x}{\bar x - 1}
$$

Here are both estimates as functions.

```{code-cell} ipython3
def pareto_mle(x):
    return len(x) / np.sum(np.log(x))

def pareto_mom(x):
    m = np.mean(x)
    return m / (m - 1)
```

### Why the method of moments struggles

The method of moments has two problems with heavy tails.

First, when $\alpha \leq 1$ the mean is infinite, so there is no population mean for the sample mean to match.

But the formula can still be computed, since every sample has a finite mean.

Moreover, every observation is at least one, so the sample mean is above one and the formula always returns a value above one.

So if the true $\alpha$ is at or below one, the method of moments silently gives an answer on the wrong side of one, suggesting that the mean is finite when it is not.

Second, even when $1 < \alpha \leq 2$, so that the mean is finite, the variance is infinite.

As we saw in {doc}`observed_distributions`, the sample mean then settles down only slowly, because it is dominated by a few very large draws, and the method of moments inherits this instability.

Maximum likelihood has neither problem.

To see this, rewrite {eq}`eq:pareto_mle` as

$$
\hat \alpha_{\text{MLE}} = \frac{1}{m}
\quad \text{where} \quad
m = \frac{1}{n} \sum_{i=1}^n \ln x_i
$$

So the estimate depends on the data only through $m$, the average of the logs of the observations, and taking logs tames the very large draws.

In fact, by the relationship between the Pareto and exponential distributions noted in {doc}`prob_dist`, each $\ln x_i$ is exponentially distributed.

The exponential distribution is light-tailed, so $m$ settles down quickly as the sample grows.

### A simulation

Let's check this by simulation, drawing 2,000 samples of size 1,000 for each of two values of $\alpha$.

We take $\alpha = 0.8$, where the mean is infinite, and $\alpha = 1.2$, where the mean is finite but the variance is infinite.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Maximum likelihood and method of moments estimates
    name: fig:mle-vs-mom
---
rng = np.random.default_rng(1234)
num_reps, n = 2_000, 1_000

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, α, lims in zip(axes, (0.8, 1.2), ((0.65, 1.2), (1.0, 1.5))):
    pareto = scipy.stats.pareto(α)
    α_mle, α_mom = [], []
    for r in range(num_reps):
        x = pareto.rvs(n, random_state=rng)
        α_mle.append(pareto_mle(x))
        α_mom.append(pareto_mom(x))
    bins = np.linspace(*lims, 80)
    ax.hist(α_mle, bins=bins, density=True, alpha=0.5,
            label='maximum likelihood')
    ax.hist(α_mom, bins=bins, density=True, alpha=0.5,
            label='method of moments')
    ax.axvline(α, color='k', ls=':', lw=2, label=r'true $\alpha$')
    ax.set_title(rf'$\alpha = {α}$')
    ax.set_xlabel(r'estimate of $\alpha$')
    ax.set_ylabel('density')
    ax.legend()
plt.tight_layout()
plt.show()
```

When $\alpha = 0.8$ (left), every method of moments estimate lies above one, as predicted, while the maximum likelihood estimates are centered on the truth.

When $\alpha = 1.2$ (right), the maximum likelihood estimates are again centered on the truth, while the method of moments estimates are too high in most samples and much more dispersed.

They are too high because most samples miss the rare, very large draws that make up much of the population mean, so the sample mean is usually too small, and a small sample mean gives a large estimate.

The long left tail comes from the opposite case, samples in which a few very large draws inflate the mean.

Here are the standard deviations of the two sets of estimates for $\alpha = 1.2$.

```{code-cell} ipython3
np.std(α_mle), np.std(α_mom)
```

The method of moments estimates are almost twice as dispersed.

(mle_consistency)=
### Consistency

An estimator is called [consistent](https://en.wikipedia.org/wiki/Consistent_estimator) if it converges to the true value of the parameter as the sample size grows.

We saw a first sign of this for the loan example in {numref}`fig:mle-bernoulli-lik-n`.

Let's check it for the Pareto estimate {eq}`eq:pareto_mle`, with $\alpha = 1.5$ and samples of increasing size, using a {doc}`kernel density estimate <observed_distributions>` of the 2,000 estimates at each size.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Maximum likelihood estimates of the tail index
    name: fig:mle-consistency
---
α_true = 1.5
pareto = scipy.stats.pareto(α_true)
α_grid = np.linspace(0.9, 2.3, 400)

fig, ax = plt.subplots()
for n in (50, 200, 1_000, 5_000):
    α_hats = []
    for r in range(num_reps):
        x = pareto.rvs(n, random_state=rng)
        α_hats.append(pareto_mle(x))
    kde = scipy.stats.gaussian_kde(α_hats)
    ax.plot(α_grid, kde(α_grid), lw=2, label=f'$n = {n}$')
ax.axvline(α_true, color='k', ls=':', lw=1, label=r'true $\alpha$')
ax.set_xlabel(r'estimate of $\alpha$')
ax.set_ylabel('density')
ax.legend()
plt.show()
```

As $n$ grows, the estimates concentrate on the true value.

Maximum likelihood estimates are consistent under conditions that hold in most applications, which is one reason why maximum likelihood is the standard method of estimation.

These conditions include that the model is not {ref}`misspecified <mle_misspecification>`, meaning that the family we choose contains the distribution that generated the data.

The advantage of maximum likelihood in our simulation came from fitting the right family: if we fitted a normal distribution to Pareto data, maximum likelihood would simply return the sample mean and standard deviation, which suffer from the same heavy tails.

## Exercises

```{exercise-start}
:label: mle_ex1
```

You ask an AI tool to fit a gamma distribution to the house prices by maximum likelihood, and it returns the code below.

```{code-block} python3
params = scipy.stats.gamma.fit(price)
u = scipy.stats.gamma(*params)
```

The code runs without an error or a warning.

Run it, compare the parameters with those found in the lecture, and compare the mean of the fitted distribution with the mean of the data.

What has gone wrong?

```{exercise-end}
```

```{solution-start} mle_ex1
:class: dropdown
```

```{code-cell} ipython3
params = scipy.stats.gamma.fit(price)
u = scipy.stats.gamma(*params)
params
```

Without `floc=0`, `fit` also estimates the location, which is the starting point of the distribution.

The fitted location sits at the smallest price in the data, and the shape is far below the value found in the lecture.

```{code-cell} ipython3
price.min(), u.mean(), price.mean()
```

The fitted distribution has a mean of about 14 thousand dollars, while the mean price is about 181 thousand.

Its log likelihood is also far below that of the lecture's fit, so this is not a better fit obtained with an extra parameter.

```{code-cell} ipython3
u.logpdf(price).sum(), u_mle.logpdf(price).sum()
```

When the shape is below one, the gamma density rises without bound at its starting point.

So, with the location free, the log likelihood has a spike when the starting point is just below the smallest price, and the search has been drawn there and stopped at a poor solution.

Nothing in the output warns us, which is why it pays to check a fitted distribution against the data, for example by comparing means or plotting the fitted density.

```{solution-end}
```

```{exercise-start}
:label: mle_ex2
```

The [Poisson distribution](https://en.wikipedia.org/wiki/Poisson_distribution) with parameter $\lambda$ has probability mass function $f(x; \lambda) = e^{-\lambda} \lambda^x / x!$.

1. Show that the maximum likelihood estimate of $\lambda$ from IID observations $x_1, \ldots, x_n$ is the sample mean.
2. Check this on the Premier League goals data from {ref}`obs_ex4`, by plotting the log likelihood on a grid of values for $\lambda$.

(SciPy's `scipy.stats.poisson.logpmf` computes $\ln f(x; \lambda)$.)

```{code-block} python3
url = ('https://github.com/QuantEcon/data-lectures/raw/main/'
       'lectures/epl_match_goals.csv')
matches = pd.read_csv(url)
goals = matches['home_goals'] + matches['away_goals']
```

```{exercise-end}
```

```{solution-start} mle_ex2
:class: dropdown
```

The log likelihood is

$$
\ell(\lambda) = -n \lambda + \ln \lambda \sum_{i=1}^n x_i - \sum_{i=1}^n \ln x_i!
$$

Setting the derivative to zero gives

$$
-n + \frac{1}{\lambda} \sum_{i=1}^n x_i = 0
\quad \implies \quad
\hat \lambda = \frac{1}{n} \sum_{i=1}^n x_i
$$

So, as for the loan and normal examples, maximum likelihood agrees with the method of moments.

Here is the check.

```{code-cell} ipython3
url = ('https://github.com/QuantEcon/data-lectures/raw/main/'
       'lectures/epl_match_goals.csv')
matches = pd.read_csv(url)
goals = matches['home_goals'] + matches['away_goals']

λ_grid = np.linspace(2, 3.5, 200)
ℓ_values = [scipy.stats.poisson.logpmf(goals, λ).sum() for λ in λ_grid]

fig, ax = plt.subplots()
ax.plot(λ_grid, ℓ_values, lw=2)
ax.axvline(goals.mean(), color='k', ls='--', lw=1, label='sample mean')
ax.set_xlabel(r'$\lambda$')
ax.set_ylabel('log likelihood')
ax.legend()
plt.show()
```

The maximum is at the sample mean.

```{solution-end}
```

```{exercise-start}
:label: mle_ex3
```

The maximum likelihood estimate of the variance of a normal distribution is $\hat \sigma^2 = \frac{1}{n} \sum_i (x_i - \bar x)^2$, which divides by $n$.

As noted in {doc}`observed_distributions`, pandas divides by $n - 1$ by default.

Draw 100,000 samples of size $n = 5$ from the standard normal distribution, and compute the average of $\hat \sigma^2$ across the samples.

Compare it with the true value $\sigma^2 = 1$, and then repeat with the divisor $n - 1$.

What do you conclude?

```{exercise-end}
```

```{solution-start} mle_ex3
:class: dropdown
```

```{code-cell} ipython3
rng = np.random.default_rng(4)
x = rng.standard_normal((100_000, 5))

x.var(axis=1).mean(), x.var(axis=1, ddof=1).mean()
```

With divisor $n$, the average is close to $(n - 1)/n = 0.8$, so the maximum likelihood estimate of $\sigma^2$ is too small on average, which is called [bias](https://en.wikipedia.org/wiki/Bias_of_an_estimator).

With divisor $n - 1$, the average is close to one, so this estimate is unbiased.

The bias disappears as $n$ grows, so it matters only for small samples.

```{solution-end}
```

```{exercise-start}
:label: mle_ex4
```

Suppose that $x_1, \ldots, x_n$ are IID draws from the uniform distribution on $[0, \theta]$, where $\theta > 0$ is unknown.

1. Show that the likelihood is $\theta^{-n}$ when $\theta \geq \max_i x_i$ and zero otherwise.
2. Conclude that the maximum likelihood estimate is $\hat \theta = \max_i x_i$.
3. Plot the likelihood for a sample of size 10 with $\theta = 1$.

Notice that the maximizer is not found by setting a derivative to zero.

```{exercise-end}
```

```{solution-start} mle_ex4
:class: dropdown
```

The density of each observation is $1/\theta$ on $[0, \theta]$ and zero elsewhere.

If any observation exceeds $\theta$, then that observation has zero density, so the likelihood is zero.

Otherwise the likelihood is $(1/\theta)^n = \theta^{-n}$.

Since $\theta^{-n}$ is decreasing in $\theta$, the likelihood is maximized by the smallest $\theta$ that is compatible with the data, which is $\max_i x_i$.

```{code-cell} ipython3
rng = np.random.default_rng(3)
x = rng.uniform(0, 1, size=10)

θ_grid = np.linspace(0.5, 1.5, 500)
L = np.where(θ_grid >= x.max(), θ_grid**(-len(x)), 0.0)

fig, ax = plt.subplots()
ax.plot(θ_grid, L, lw=2)
ax.axvline(x.max(), color='k', ls='--', lw=1, label=r'$\max_i x_i$')
ax.set_xlabel(r'$\theta$')
ax.set_ylabel('likelihood')
ax.legend()
plt.show()
```

The likelihood jumps up at $\max_i x_i$ and declines afterwards, so the maximum is at the jump.

```{solution-end}
```

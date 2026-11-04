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

# Introduction to the Course

```{note}
This lecture is under construction.
```


## What is Python?

[Python](https://www.python.org) is a general-purpose programming language.

It is free and [open source](https://en.wikipedia.org/wiki/Open_source), so anyone can use, read and modify it at no cost.

It is also **high level**, which means that its code is written for people to read rather than for machines to run directly.

The core language is small.

Most of what we do in this course comes from **libraries**, which are packages of code that extend the core language.

### A short history

Python began as a holiday project.

Guido van Rossum [started it](https://docs.python.org/3/faq/general.html#why-was-python-created-in-the-first-place) over the 1989 Christmas break, while working at [CWI](https://www.cwi.nl/en/), the Dutch national research institute for mathematics and computer science.

His aim was an easier way to do system administration, not data analysis.

He took the name from *Monty Python's Flying Circus*, and announced the language publicly in February 1991.

Data science came later, as people built the tools they needed for their own work.

| Year | Project | Origin |
|---|---|---|
| 1995 | Numeric | Jim Hugunin extends a matrix package to multidimensional arrays |
| 2001 | SciPy | Travis Oliphant, Eric Jones and Pearu Peterson merge their scientific modules |
| 2001 | IPython | Fernando Pérez, a graduate student, starts an improved interactive Python |
| c. 2003 | [Matplotlib](https://matplotlib.org/stable/project/history.html) | John Hunter, working on EEG data, wants MATLAB-style plots in Python |
| 2006 | NumPy 1.0 | Oliphant combines the competing array libraries, including Numeric |
| 2008 | pandas | Wes McKinney starts it at AQR Capital Management, an investment firm {cite}`mckinney2022python` |
| 2010 | [scikit-learn](https://scikit-learn.org/stable/about.html) | First public release, led by researchers at INRIA in France |
| 2014 | Project Jupyter | The notebook grows out of IPython and becomes language neutral |
| 2015 | [TensorFlow](https://research.google/blog/tensorflow-googles-latest-machine-learning-system-open-sourced-for-everyone/) | Google releases its machine learning system as open source |
| 2017 | [PyTorch](https://soumith.ch/posts/2023/12/pytorch-design-origins/) | Public release of a Python successor to Torch, a library used through the Lua language |
| 2022 | ChatGPT | LLMs bring code generation to people who do not program |


We meet many of these libraries in {doc}`toolkit`.


### Why did Python become so important?

No one can say for certain, so what follows is our speculation rather than established fact.

Here are the reasons we find most plausible.

*It connects easily to fast code.*

Python itself is slow at numerical work, but it can easily call routines written in fast languages like C and Fortran {cite}`mckinney2022python`.

Users write readable code, and the heavy calculation happens elsewhere.

*It does more than statistics.*

One language can collect data from the web, clean it, fit a model and run the result in production.

Teams that prototype in a specialized language often have to rewrite their work for production.

*It is free.*

MATLAB, SAS and Stata all require paid licenses, which matters to students, to organizations with small budgets and to anyone running code on many machines.

*It is relatively easy to learn.*

Many people meet Python as a first programming language, so it was already familiar when they needed tools for data.

*Deep learning brought money and attention.*

Google and Meta built their main deep learning libraries for Python, and invested heavily in them.

*Success reinforces itself.*

More users mean more libraries and tutorials, which attract more users.

LLMs may be strengthening this loop, since a study of eight models found that Python is their strong default, even for tasks where it is not the best fit {cite}`twist2025study`.

None of this means that Python's success was inevitable.

PyTorch's predecessor was used through Lua, and data science could have settled on a different language.


### Python and R

```{epigraph}
"Python has gotten sufficiently weapons grade that we don't descend into R
anymore. Sorry, R people. I used to be one of you but we no longer descend
into R." -- Chris Wiggins
```


R grew out of [S](https://en.wikipedia.org/wiki/S_(programming_language)), a language for statistics developed at Bell Labs from the mid-1970s by John Chambers and colleagues.

Ross Ihaka and Robert Gentleman wrote R in the statistics department at the University of Auckland in the early 1990s.

It became free software in 1995, and [version 1.0.0](https://cran.r-project.org/src/base/README) appeared on 29 February 2000.

Hence the main difference: R was written by statisticians for data analysis, while Python is a general-purpose language that acquired data tools later.

| | Python | R |
|---|---|---|
| Origin | General-purpose language | Language for statistics |
| Data tables | Added by a library (pandas) | Built into the language |
| Statistical methods | Broad coverage | Very deep coverage, and new methods often appear here first |
| Graphics | Matplotlib, seaborn | ggplot2, which is widely admired |
| Machine learning and AI | The dominant language | A much smaller role |
| Running in production | Common | Less common |

The two also borrow from each other.

The pandas DataFrame is named after R's `data.frame` {cite}`mckinney2022python`, and the statistics library statsmodels accepts [R-style formulas](https://www.statsmodels.org/stable/example_formulas.html).

To see how close they are, let's run the same small analysis in both, using a made-up table of schooling and income.

You don't need to understand the details yet, since we start reading code in {doc}`python_ai`.

Here is the Python version.

```{code-cell} ipython3
import pandas as pd
import statsmodels.formula.api as smf

df = pd.DataFrame({
    'region': ['north', 'north', 'north', 'south', 'south', 'south'],
    'schooling': [8, 10, 12, 6, 9, 11],
    'income': [21, 27, 33, 15, 23, 30]
})

df.groupby('region')['income'].mean()  # Mean income by region
```

```{code-cell} ipython3
smf.ols('income ~ schooling', data=df).fit().params  # Fit a regression
```

Here is the same analysis in R.

```{code-block} r
df <- data.frame(
  region = c("north", "north", "north", "south", "south", "south"),
  schooling = c(8, 10, 12, 6, 9, 11),
  income = c(21, 27, 33, 15, 23, 30)
)

aggregate(income ~ region, data = df, FUN = mean)  # Mean income by region
coef(lm(income ~ schooling, data = df))  # Fit a regression
```

R needs no imports here, because data tables and regression are built in.

Otherwise the two versions are very similar.

### Stata and Excel

Many of you will have used Stata or Excel, which are also common in policy work.

Stata remains widely used in applied economics, and Excel is useful for small tables that need to be shared.

However, Stata plays a small role in machine learning and AI, and work done by hand in Excel is hard to automate and reproduce.

### Why Python for this course?

This course uses Python because 

* it has the widest reach across data science and AI,
* it is free, and 
* AI tools work extremely well with it.

Fortunately, AI makes translating between languages much cheaper, so the choice
of language is less of a concern.


## What is data science?

Data science is not easy to define.

To understand what it covers, let's start with some history.

### From statistics to data science

The foundations of classical statistics have their roots in mathematics and the natural sciences.

Classical statistics emphasizes 

1. estimating small parametric models that are pinned down by theory, and
2. testing hypotheses --- in particular, the validity of models.

But not everyone who wants to use statistics fits this mold.

Examples include

* people in the private sector looking to gain insights from data,
* computer scientists focused on pragmatic knowledge creation, without testing formal hypotheses,
* policy professionals who need evidence to support their decisions, and
* social scientists who have no formal model to test.

Even in economics, which has many formal models, most of them are easily
rejected along at least one dimension, so hypotheses testing is not necessarily
the right paradigm.

todo: make this a note:

==

Some statisticians were making similar points.

In 1962, John Tukey called for a science of data analysis that went beyond mathematical statistics {cite}`tukey1962future`.

In 1976, George Box observed that "all models are wrong" {cite}`box1976science`.

In 2001, Leo Breiman contrasted two cultures: one models how the data were generated, while the other uses algorithms and judges them by how well they predict {cite}`breiman2001statistical`.

==

### Supply and demand

The supply side also changed:

* *Data:* the web, phones and digital records began to generate data on a huge scale.
* *Computing:* cheap computing made it possible to process that data and fit flexible models.
* *Methods:* machine learning, largely from computer science, supplied tools that can predict well without a model pinned down by theory.

The job title *data scientist* came into use at technology firms such as Facebook and LinkedIn around 2008.

In 2012, the Harvard Business Review called it the sexiest job of the 21st century {cite}`davenport2012data`.

Python's data libraries, including pandas and scikit-learn, appeared in those same years.

### A working definition

In this course, **data science** means a collection of techniques and methods for generating knowledge, insights and forecasts from data, and for supporting decision making.

The table below compares it with classical statistics, although the differences are ones of emphasis.

| | Classical statistics | Data science |
|---|---|---|
| Main aim | Estimate models and test hypotheses | Generate insights and forecasts, support decisions |
| Role of theory | Pins down a small parametric model | Often no formal model |
| Typical data | Carefully constructed and relatively small | Large and messy |
| Measure of success | Evidence for or against a hypothesis | Predictive power, usefulness for decisions |

Note that data science does not replace statistics.

It uses many of the same tools for different ends, which is why we discuss many statistical concepts.


### Some policy examples

Here are three examples of data science applied to policy questions.

*Bail decisions.*

A judge deciding whether a defendant can wait for trial at home is trying to predict what that defendant will do.

{cite:t}`kleinberg2018human` used arrest data from New York City to compare judges with a machine learning rule.

They estimate that such a rule could cut jailing rates by about 42% with no increase in crime.


*Mapping poverty.*

Policy makers need information on living standards in different regions in order to make decisions about services.

Reliable data on living standards are scarce in many developing countries, because household surveys are infrequent and costly.

{cite:t}`jean2016combining` combined survey data with satellite images from Nigeria, Tanzania, Uganda, Malawi and Rwanda.

Their model identified image features that explain up to 75% of the variation in local economic outcomes.


*Detecting illegal fishing.*

Most fishing vessels do not broadcast their positions, which makes illegal fishing hard to monitor.

{cite:t}`park2020illuminating` studied the waters between the Koreas, Japan and Russia, using a neural network to identify fishing vessels in satellite images.

They find that more than 900 vessels of Chinese origin in 2017, and more than 700 in 2018, fished illegally in North Korean waters.

Over those two years, the estimated catch was more than 164,000 tonnes of squid, worth more than $440 million.

That is about as much as Japan and South Korea caught combined.


## Jupyter notebooks, Colab and local installs

Topics:

* Jupyter notebooks
* Google Colab and local installation

## The role of AI

Topics:

* The role of AI, and what we still need to learn
* Confidential data and web-hosted AI

## Further reading

There is no agreed definition of data science.

{cite:t}`donoho201750` reviews the debate and the field's history.

{cite:t}`breiman2001statistical` makes the case for prediction-focused methods.

{cite:t}`kleinberg2015prediction` separates policy questions that need prediction from those that need causal evidence.



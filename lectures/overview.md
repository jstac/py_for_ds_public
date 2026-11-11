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

This is the first lecture in **Python for Data Science** (DSC1030E / DSC6031E) at the National Graduate Institute for Policy Studies, Winter 2027.

In this lecture we first introduce Python and data science, and then discuss our aims
and learning objectives.

## What is Python?

[Python](https://www.python.org) is a general-purpose programming language that can be used for almost any programming task.

It is free and [open source](https://en.wikipedia.org/wiki/Open_source), so anyone can use, read and modify it at no cost.

It is also [high level](https://en.wikipedia.org/wiki/High-level_programming_language), which means it was designed for readability (but not speed, which came later).

The core language is small.

Most of what we do in this course comes from [libraries](https://en.wikipedia.org/wiki/Library_(computing)):

* Packages of code that extend the core language
* Some are for statistics / data science / AI
* Some allow Python programs to run fast

### A short history

Python began as a holiday project.

[Guido van Rossum](https://en.wikipedia.org/wiki/Guido_van_Rossum) [started it](https://docs.python.org/3/faq/general.html#why-was-python-created-in-the-first-place) over the 1989 Christmas break, while working at [CWI](https://www.cwi.nl/en/), the Dutch national research institute for mathematics and computer science.

His aim was an easier way to do [system administration](https://en.wikipedia.org/wiki/System_administrator).

He took the name from [*Monty Python's Flying Circus*](https://en.wikipedia.org/wiki/Monty_Python), and announced the language publicly in February 1991.

Over time, people who liked the language added scientific and numerical
capabilities through libraries (code packages).

| Year | Project | Used for | Origin |
|---|---|---|---|
| 1995 | Numeric | Fast calculation with arrays (grids) of numbers | [Jim Hugunin](https://en.wikipedia.org/wiki/Jim_Hugunin) builds a package for multidimensional arrays |
| 2001 | [SciPy](https://scipy.org/) | Scientific and statistical routines | [Travis Oliphant](https://github.com/teoliphant), [Eric Jones](https://www.enthought.com/team/) and [Pearu Peterson](https://github.com/pearu) merge their scientific modules |
| 2001 | [IPython](https://ipython.org/) | Running Python interactively | [Fernando Pérez](https://github.com/fperez), a graduate student, starts an improved interactive Python |
| c. 2003 | [Matplotlib](https://matplotlib.org/stable/project/history.html) | Figures and plots | [John Hunter](https://en.wikipedia.org/wiki/John_D._Hunter), working on EEG (brain activity) data, wants MATLAB-style plots in Python |
| 2006 | [NumPy](https://numpy.org/) 1.0 | Fast calculation with arrays of numbers | Oliphant combines the competing array libraries, including Numeric |
| 2008 | [pandas](https://pandas.pydata.org/) | Tables of data | [Wes McKinney](https://github.com/wesm) starts it at AQR Capital Management, an investment firm {cite}`mckinney2022python` |
| 2010 | [scikit-learn](https://scikit-learn.org/stable/about.html) | Machine learning | First public release, led by researchers at INRIA in France |
| 2014 | [Jupyter](https://jupyter.org/) | Notebooks that mix code and text | The notebook grows out of IPython and comes to support many languages |
| 2015 | [TensorFlow](https://research.google/blog/tensorflow-googles-latest-machine-learning-system-open-sourced-for-everyone/) | Deep learning | Google releases its machine learning system as open source |
| 2017 | [PyTorch](https://soumith.ch/posts/2023/12/pytorch-design-origins/) | Deep learning | Public release of a Python successor to Torch, a library used through the [Lua](https://www.lua.org/) language |
| 2022 | [ChatGPT](https://en.wikipedia.org/wiki/ChatGPT) | Writing text and code on request | [Large language models](https://en.wikipedia.org/wiki/Large_language_model) (LLMs) bring code generation to people who do not program |


We meet many of these libraries in {doc}`toolkit`.


### Why did Python become so important for science / data?

Here are some reasons:

*It is well designed and relatively easy to learn.*

Many people meet Python as a first programming language, so it was already familiar when they needed tools for data.

Many people who started working in Python enjoyed using the language.


*It connects easily to fast code.*

Python itself is slow at numerical work, but it can easily call routines written in fast languages like C and Fortran {cite}`mckinney2022python`.

Scientific libraries use these routines from a Python environment, so we don't need to know the details.

Users write readable code, and the heavy calculation happens elsewhere.

*It does more than statistics.*

One language can collect data from the web, clean it, fit a model and run the result in [production](https://en.wikipedia.org/wiki/Deployment_environment), meaning in a live system that people rely on.

Teams that build a first version in a specialized language often have to rewrite their work for production.

*It is free.*

Commercial alternatives such as [MATLAB](https://en.wikipedia.org/wiki/MATLAB), [SAS](https://www.sas.com/) and [Stata](https://www.stata.com/) require paid licenses, which matters to students, to organizations with small budgets and to anyone running code on many machines.

*Deep learning brought money and attention.*

[Machine learning](https://en.wikipedia.org/wiki/Machine_learning) means algorithms that learn patterns from data, and [deep learning](https://en.wikipedia.org/wiki/Deep_learning) is a form of it that uses large [neural networks](https://en.wikipedia.org/wiki/Neural_network_(machine_learning)).

Google and Meta built their main deep learning libraries for Python, and invested heavily in them.

*Success reinforces success.*

More users mean more libraries and tutorials, which attract more users.

LLMs, the AI models behind tools like ChatGPT, may be strengthening this loop, since a study of eight models found that Python is their strong default, even for tasks where it is not the best fit {cite}`twist2025study`.



### Python and R

```{epigraph}
"Python has gotten sufficiently weapons grade that we don't descend into R
anymore. Sorry, R people. I used to be one of you but we no longer descend
into R." -- Chris Wiggins
```


[R](https://www.r-project.org/) grew out of [S](https://en.wikipedia.org/wiki/S_(programming_language)), a language for statistics developed at [Bell Labs](https://en.wikipedia.org/wiki/Bell_Labs) from the mid-1970s by John Chambers and colleagues.

Ross Ihaka and Robert Gentleman wrote R in the statistics department at the University of Auckland in the early 1990s.

It became free software in 1995, and [version 1.0.0](https://cran.r-project.org/src/base/README) appeared on 29 February 2000.

R was written by statisticians for data analysis, while Python is a general-purpose language that acquired data tools later.

| | Python | R |
|---|---|---|
| Origin | General-purpose language | Language for statistics |
| Data tables | Needs a library | Built into the language |
| Statistical methods | Broad coverage | Deep coverage of statistical methods |
| Graphics | Matplotlib, [seaborn](https://seaborn.pydata.org/) | [ggplot2](https://ggplot2.tidyverse.org/) |
| Machine learning and AI | The dominant language | A much smaller role |
| Running in production | Common | Less common |

The two also borrow from each other.

The pandas DataFrame is named after R's `data.frame` {cite}`mckinney2022python`, and the statistics library statsmodels accepts [R-style formulas](https://www.statsmodels.org/stable/example_formulas.html).

To see how close they are, let's run the same small analysis in both, using a made-up table of schooling and income.

The analysis computes mean income in each region, and then fits a [regression](https://en.wikipedia.org/wiki/Regression_analysis) of income on years of schooling.

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

Notice that R does not need any import statements because data tables and regression are built in.

Otherwise the two versions are very similar.

We discuss regression in {doc}`bivariate_dist`.

### Stata and Excel

Many of you will have used Stata or [Excel](https://en.wikipedia.org/wiki/Microsoft_Excel), which are also common in policy work.

Stata remains widely used in applied economics, and Excel is useful for small tables that need to be shared.

However, Stata

* is too limited for many problems in data science, machine learning and AI, and
* does not pair very well with AI (relative to Python).

Also, work done by hand in Excel is hard to automate and reproduce.

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

1. estimating small [parametric models](https://en.wikipedia.org/wiki/Parametric_model) (models where the basic laws and relationships are known from theory, so we only need to estimate a few unknown values), and
2. [testing hypotheses](https://en.wikipedia.org/wiki/Statistical_hypothesis_test) --- in particular, testing the validity of models.

For example, when the asteroid [Ceres](https://en.wikipedia.org/wiki/Ceres_(dwarf_planet)) was lost behind the Sun in 1801, [Carl Friedrich Gauss](https://en.wikipedia.org/wiki/Carl_Friedrich_Gauss) used gravitational theory and a few observations to estimate the six parameters of its orbit, and astronomers found it again where he predicted {cite}`forbes1971gauss`.

Similarly, during the [1919 solar eclipse](https://en.wikipedia.org/wiki/Eddington_experiment), [Arthur Eddington](https://en.wikipedia.org/wiki/Arthur_Eddington) and colleagues tested Einstein's theory of gravity, which predicted that the Sun would bend starlight by twice the Newtonian amount, and their results favored Einstein {cite}`dyson1920determination`.

But not everyone who wants to use statistics fits this mold.

Examples include

* people in the private sector looking to gain insights from data,
* computer scientists focused on pragmatic knowledge creation, without testing formal hypotheses,
* policy professionals who need evidence to support their decisions, and
* social scientists who have no formal model to test.

Even in economics, which has many formal models, most of them are easily
rejected along at least one dimension, so hypothesis testing is not necessarily
the right paradigm.

```{note}
Some statisticians were making similar points.

In 1962, [John Tukey](https://en.wikipedia.org/wiki/John_Tukey) called for a science of data analysis that went beyond mathematical statistics {cite}`tukey1962future`.

In 1976, [George Box](https://en.wikipedia.org/wiki/George_E._P._Box) observed that "all models are wrong" {cite}`box1976science`.

In 2001, [Leo Breiman](https://en.wikipedia.org/wiki/Leo_Breiman) contrasted two cultures: one models how the data were generated, while the other uses algorithms and judges them by how well they predict {cite}`breiman2001statistical`.
```

### Supply and demand

The groups listed above created a demand for new ways to learn from data.

The supply side also changed:

* *Data:* the web, phones and digital records began to generate data on a huge scale.
* *Computing:* cheap computing made it possible to process that data and fit flexible models.
* *Methods:* machine learning, largely from computer science, supplied tools that can predict well without a model pinned down by theory.

The job title [data scientist](https://en.wikipedia.org/wiki/Data_scientist) came into use at technology firms such as Facebook and LinkedIn around 2008.

In 2012, the Harvard Business Review called it the sexiest job of the 21st century {cite}`davenport2012data`.

Python's data libraries, including pandas and scikit-learn, appeared in those same years.

### A working definition

In this course, [data science](https://en.wikipedia.org/wiki/Data_science) means a collection of techniques and methods for generating knowledge, insights and forecasts from data, and for supporting decision making.

The table below compares it with classical statistics, although the differences are ones of emphasis.

| | Classical statistics | Data science |
|---|---|---|
| Main aim | Estimate models and test hypotheses | Generate insights and forecasts, support decisions |
| Role of theory | Pins down a parametric model | Often no formal model |
| Typical data sets | Carefully constructed, often small | Messy, often large |
| Measure of success | Evidence for or against a hypothesis | Predictive power, usefulness for decisions |


Note that data science uses many statistical tools, albeit for different ends.

Hence we still need to master core statistical concepts.


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


## Jupyter notebooks

A [Jupyter notebook](https://en.wikipedia.org/wiki/Project_Jupyter) is a document that combines code, the results of running that code, and formatted text.

Tables and figures appear directly below the code that produced them.

This suits data analysis, where we try something, look at the result and decide what to do next.

Each lecture in this book can be downloaded as a notebook.

### Background

Notebooks grew out of [IPython](https://ipython.org/), and became a separate project, [Jupyter](https://jupyter.org/), in 2014.

The name comes from three languages: [Julia](https://julialang.org/), Python and R {cite}`perkel2018why`.

(Julia is a language for scientific computing that is sometimes used for data science.)

Like Python, Jupyter is free and [open source](https://jupyter.org/about).

Jupyter is an extremely popular platform for data science.

* GitHub [reported](https://github.blog/news-insights/octoverse/octoverse-2024/) a 92% jump in notebook use in 2024, and linked this to data science, AI and academic work.
* In the [2025 survey](https://survey.stackoverflow.co/2025/technology) by Stack Overflow, a question-and-answer site for programmers, 14.1% of respondents used Jupyter and 7% used Google Colab.

Notebooks are also used for data analysis at leading technology firms, including Meta, Google, LinkedIn, Anthropic, Amazon and DeepMind.


### Two ways to run notebooks

You can install Python and Jupyter on your own computer, which is called a local installation, or use [Google Colab](https://colab.research.google.com/) in a web browser.

Colab is the easiest way to start, since there is nothing to install.

A local installation is better if you plan to use Python regularly, or if you work with data that should not leave your computer.


(overview:colab)=
### Google Colab

[Google Colab](https://colab.research.google.com/) is a [hosted Jupyter notebook service](https://research.google.com/colaboratory/faq.html) that runs in your browser.

[Hosted](https://en.wikipedia.org/wiki/Software_as_a_service) means that Google runs the software for you, on its own computers (called [servers](https://en.wikipedia.org/wiki/Server_(computing))).

It is free to use, but you need a Google account.

To open a lecture from this book, click the download icon at the top of the lecture page, go to Colab and choose File, then Upload notebook.

When you run a notebook, Colab lends you a temporary computer on its servers, and this period of use is called a session.

Colab uses the same notebook file format as Jupyter, so a notebook can move between the two.

However, the two are not the same.

| | Local Jupyter | Google Colab |
|---|---|---|
| Where code runs | Your computer | Google's servers |
| Setup | Install once | None |
| Libraries | Those you have installed | Common ones are preinstalled, and others must be installed each session |
| Where notebooks are saved | Your computer | Google Drive |
| Other files | Stay on your computer | Files on the server can disappear when the session ends |
| Time limits | None | Sessions stop when idle, and after at most 12 hours on the free plan |
| Computing power | Your computer's | Varies, and is not guaranteed on the free plan |
| Sharing | Send the file | Share a link, as with Google Docs |
| Interface | JupyterLab | Google's own design, with different menus and shortcuts |

Two practical points follow.

First, keep your own copy of anything you upload to or create on the server, since those files do not last.

Second, using Colab means sending your code and data to Google, so think before uploading anything confidential.

We return to that point {ref}`below <overview:confidentiality>`, when we discuss the role of AI.


### Local installation

The simplest local option is the [Anaconda distribution](https://www.anaconda.com/download), which bundles Python, Jupyter and the main scientific libraries.

To install it,

1. download the installer for your operating system (you can skip the registration step), and
2. run the installer, accepting the default options.

The download is large, so use a good internet connection.

To start Jupyter, open Anaconda Navigator from your applications and click on JupyterLab.

Alternatively, you can type `jupyter lab` into a [terminal](https://en.wikipedia.org/wiki/Terminal_emulator), which is a window
for typing commands.

(On a Mac, the terminal is an application called Terminal. On Windows you should use the Anaconda Prompt.)

Your web browser will then open [JupyterLab](https://jupyterlab.readthedocs.io/), where you can create or open notebooks.

The browser is only the interface: your code runs on your own computer and your files stay there.

JupyterLab shows the files in the folder where it started, so save any notebooks you download somewhere you can reach from there.


```{note}
Anaconda's [terms](https://www.anaconda.com/pricing/terms-of-service-faqs) allow free use for personal, non-commercial purposes and for eligible academic and non-profit organizations.

If you plan to use Anaconda at work, check your organization's position.

[Miniforge](https://github.com/conda-forge/miniforge) is an open source alternative that installs [conda](https://docs.conda.io/), a tool for installing libraries, set up to use the community-run conda-forge collection, although you then install Jupyter and the libraries yourself.
```

### Using notebooks

A notebook is a sequence of cells.

Code cells contain Python, and text cells contain formatted text written in [Markdown](https://www.markdownguide.org/basic-syntax/), a simple way to add headings, lists and links.

To run a cell, click on it and press Shift+Enter.

Here is a code cell, followed by its output.

The cell sets `x` to 10 and then computes `x` times 2.

```{code-cell} ipython3
x = 10
x * 2
```

The table below lists a few common actions.

| Action | JupyterLab | Colab |
|---|---|---|
| Run the current cell | Shift+Enter | Shift+Enter |
| Add a cell below | Esc, then B | Ctrl+M, then B |
| Change a cell to text | Esc, then M | Ctrl+M, then M |

Both also offer buttons and menus for these actions, so you do not need to remember the shortcuts.

To see the documentation for a [function](https://en.wikipedia.org/wiki/Function_(computer_programming)) (a named piece of code that does a task), type its name followed by a question mark, as in `abs?`, and run the cell.

If you need a library that is not installed, run `!pip install` followed by its name in a cell.

If a cell gives an error, don't worry, since errors are a normal part of working with code.

The last line of an error message usually says what went wrong, and an AI tool can help you interpret the rest.

### Hidden state

Notebooks have one trap that is worth knowing about from the start.

The program that runs your code is called the [kernel](https://en.wikipedia.org/wiki/Project_Jupyter), and it remembers everything you have run so far.

It remembers the order in which you ran the cells, not the order in which they appear on the page.

Try running this pair of cells, and then running the second one several times.

```{code-cell} ipython3
a = 1
```

```{code-cell} ipython3
a = a + 1
a
```

The answer changes each time, even though the code on the page does not.

So a notebook can look correct but give results that depend on steps that are no longer visible.

{cite:t}`pimentel2019large` tried to rerun about 864,000 notebooks from GitHub.

Only 24% ran without errors, and only 4% reproduced the saved results.

The remedy is simple: before you trust a notebook's results, restart the kernel and run all of its cells from top to bottom.

In JupyterLab, the menu option is *Restart Kernel and Run All Cells*, and Colab's Runtime menu has a similar option.

This matters even more when an AI has written or changed some of the cells.

## The role of AI

Programming and data science are now experiencing enormous changes due to the rise of cheap and powerful AI.

### AI for policy professionals

For policy work, implementation of data science procedures (coding, basic analysis) will soon be done almost entirely by AI.

This is because most of the code written for policy work will be relatively routine and familiar.

Such coding is straightforward for LLMs.

As a result, basic Python skills are *far less important* than they were a few
years ago, and they will be even less important in the future.

In this course we aim to read basic Python code, but not to write it from scratch.

At the same time, *conceptual understanding* of data science and the foundations
of statistics is essential for

* determining what is possible, given the questions and the data,
* deciding what is valuable, and
* interacting with LLMs to complete analytical tasks.

If you use AIs on data science tasks, you will soon conclude that you cannot manage them properly unless you
understand the underlying concepts.



```{note}
None of the above should be interpreted as claiming that humans will be irrelevant for policy work.

In fact the opposite is true: policy is one of the areas where human judgment will be absolutely essential.

* AIs will write the code and convert high-level explanations into mathematical and statistical models.
* Humans will direct and supervise the AIs to improve understanding and decision making.
```


(overview:confidentiality)=
### Confidentiality

Most AI tools run on a company's servers, so anything you type or upload leaves your computer.

This includes most desktop apps, and tools that can read files on your computer may send whatever they open.

What happens next depends on the provider, your plan and your settings.

* OpenAI [uses conversations from personal ChatGPT accounts](https://help.openai.com/en/articles/5722486-how-do-i-opt-out-of-my-content-being-used-to-improve-model-performance) to train its models by default, although you can opt out.
* Anthropic [uses chats from personal Claude accounts](https://privacy.claude.com/en/articles/10023580-is-my-data-used-for-model-training) for training if the user allows it in their settings.
* Google [warns users of Gemini](https://support.google.com/gemini/answer/13594961) that some chats are read by human reviewers, and advises: "Please don't enter confidential information that you wouldn't want a reviewer to see."

Colab's AI features [carry a similar warning](https://research.google.com/colaboratory/faq.html) about human reviewers.

Business and API plans often offer more protection: for example, [OpenAI](https://help.openai.com/en/articles/8983130-what-is-the-chatgpt-enterprise-and-team-data-policy) and [Anthropic](https://privacy.claude.com/en/articles/7996868-is-my-data-used-for-model-training) do not use that data for training by default.

Even so, your data has still left your computer, and these terms change, so check them yourself.

The risk is real: in 2023, [Samsung restricted staff use of generative AI](https://www.bangkokpost.com/life/tech/2562014/samsung-bans-staff-use-of-ai-over-data-leak) after engineers reportedly pasted internal source code into ChatGPT.

*What should not be uploaded.*

Unless your organization has approved the tool for this purpose, do not upload

* [personal data](https://en.wikipedia.org/wiki/Personal_data), such as survey responses or health and tax records,
* unpublished statistics, draft policy documents, and anything else your organization treats as confidential,
* data supplied under an agreement or license that restricts sharing, or
* passwords and access keys.

Removing names is not enough to protect people, since records can often be [traced back to individuals](https://en.wikipedia.org/wiki/Data_re-identification).

For example, {cite:t}`sweeney2000simple` estimated that 87% of the US population could be uniquely identified from just their ZIP code, sex and date of birth.

*How to use AI with confidential data.*

An AI usually needs to know the structure of your data, not the data itself.

So you can work as follows.

1. Describe the columns in words, or write a few made-up rows.
2. Ask the AI to write the code.
3. Read the code to check that it does what you asked.
4. Run the code on the real data, using a local installation so that the data stays on your computer.

Take care with what you paste back to the AI, since error messages and output can contain rows of real data.

The same applies to sharing, because a notebook file saves its output along with the code.

Finally, your organization's rules come first, and many governments and firms now have their own.

GRIPS also has guidelines on the use of generative AI, which you should read.


## Further reading on data science

{cite:t}`donoho201750` reviews the field's history.

{cite:t}`breiman2001statistical` makes the case for prediction-focused methods.

{cite:t}`kleinberg2015prediction` separates policy questions that need prediction from those that need causal evidence.



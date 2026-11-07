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

# Reading Python and Directing AI

In this lecture we learn to read simple Python code, and to direct an AI tool that writes code for us.

We will not write programs from scratch.

Instead, the aim is to look at short pieces of code, work out what they do, and notice when they do something other than what we asked for.

Here is an example that uses a made-up table of household incomes in four regions.

```{code-cell} ipython3
incomes = {
    'north': [21, 27, 33, 30, 24],
    'south': [15, 23],
    'east': [12, 14, 19, 16, 10, 13],
    'west': [80, 90]
}

threshold = 25


def mean(values):
    return sum(values) / len(values)


below = []

for region, values in incomes.items():
    m = mean(values)
    print(region, m)
    if m < threshold:
        below.append(region)

print('Regions below the threshold:', below)
```

Don't worry if this looks unfamiliar.

By the end of the lecture you should be able to read every line.

## Data types

In this section we look at the basic kinds of data that Python works with: numbers, strings, lists and dictionaries.

### Numbers

A [variable](https://en.wikipedia.org/wiki/Variable_(high-level_programming)) is a name for a value, and we create one using `=`.

```{code-cell} ipython3
x = 10    # This is a comment, which Python ignores
y = 2.5
x * y
```

Text after `#` is a [comment](https://en.wikipedia.org/wiki/Comment_(computer_programming)), written for human readers.

AI tools add many comments, and these are often the best place to start reading.

Arithmetic mostly works as you would expect, with `*` for multiplication and `/` for division.

Powers use `**`, so that `x**2` means $x^2$.

```{code-cell} ipython3
x**2
```

### Strings

A [string](https://en.wikipedia.org/wiki/String_(computer_science)) is a piece of text, written inside quotes.

```{code-cell} ipython3
region = 'north'
region
```

Text that looks like a number is still text.

```{code-cell} ipython3
'10' + '10'
```

Here `+` joins the two strings instead of adding them.

This is a common problem when data are loaded from a file, and no error message warns you about it.

You will also often see strings starting with `f`, which insert the value of a variable.

```{code-cell} ipython3
f'The region is {region}'
```

### Lists

A [list](https://en.wikipedia.org/wiki/List_(abstract_data_type)) holds several values in order.

```{code-cell} ipython3
values = [21, 27, 33, 30, 24]
len(values)    # The number of elements
```

Elements are selected using square brackets.

Python [counts from zero](https://en.wikipedia.org/wiki/Zero-based_numbering), so the first element has index 0.

```{code-cell} ipython3
values[0]
```

Negative numbers count from the end.

```{code-cell} ipython3
values[-1]
```

We can also select a range of elements, where the last number in the range is not included.

```{code-cell} ipython3
values[0:2]    # Elements 0 and 1, but not 2
```

To add an element to the end of a list, we use `append`.

```{code-cell} ipython3
values.append(40)
values
```

### Dictionaries

A [dictionary](https://en.wikipedia.org/wiki/Associative_array) stores values under names, which are called keys.

```{code-cell} ipython3
means = {'north': 27.0, 'south': 19.0}
means['south']
```

The same notation adds a new entry, or replaces an existing one.

```{code-cell} ipython3
means['east'] = 14.0
means
```

The table in our opening example is a dictionary in which each value is a list.

```{exercise-start}
:label: pai_ex1
```
Without running this code, work out what it prints.

```{code-block} python3
numbers = [4, 8, 15, 16]
numbers.append(23)
print(numbers[0], numbers[-1], len(numbers))

ages = {'Ana': 34, 'Ben': 29}
ages['Ben'] = 30
print(ages['Ben'])
```

```{exercise-end}
```

```{solution-start} pai_ex1
:class: dropdown
```

The first `print` shows the first element, the last element and the length of the list, which now has five elements.

The second `print` shows the replaced value.

```{code-cell} ipython3
numbers = [4, 8, 15, 16]
numbers.append(23)
print(numbers[0], numbers[-1], len(numbers))

ages = {'Ana': 34, 'Ben': 29}
ages['Ben'] = 30
print(ages['Ben'])
```

```{solution-end}
```

## Conditions

A comparison gives one of two values, `True` or `False`, known as [Booleans](https://en.wikipedia.org/wiki/Boolean_data_type).

```{code-cell} ipython3
x = 10
x > 5
```

```{code-cell} ipython3
x == 3
```

Notice the difference between the two kinds of equals sign.

A single `=` gives a value to a variable, while `==` asks whether two values are equal.

Comparisons can be combined using `and` and `or`.

```{code-cell} ipython3
x > 5 and x < 8
```

We can use a comparison to [decide what to do next](https://en.wikipedia.org/wiki/Conditional_(computer_programming)).

```{code-cell} ipython3
m = 19.0

if m < 10:
    print('low')
elif m < 25:
    print('medium')
else:
    print('high')
```

Python checks each condition in turn, and runs the first block whose condition is true.

Here `elif` is short for "else if".

Notice that the blocks are [indented](https://en.wikipedia.org/wiki/Off-side_rule).

In Python, indentation is part of the meaning of the code: the indented lines are the ones that belong to the `if`.

```{exercise-start}
:label: pai_ex2
```
Without running this code, work out what it prints.

How would the output change if `z` were 12?

```{code-block} python3
z = 7

if z > 10:
    print('large')
    print('checked')
print('done')
```

```{exercise-end}
```

```{solution-start} pai_ex2
:class: dropdown
```

Only `done` is printed, because the two indented lines belong to the `if`, and `z > 10` is false.

The last line is not indented, so it always runs.

```{code-cell} ipython3
z = 7

if z > 10:
    print('large')
    print('checked')
print('done')
```

If `z` were 12, all three lines would be printed.

```{solution-end}
```

## Loops

A [for loop](https://en.wikipedia.org/wiki/For_loop) runs a block of code once for each element in a collection.

```{code-cell} ipython3
for v in [21, 27, 33]:
    print(v * 2)
```

Here `v` takes each value in turn.

A loop is often used to build up a result.

```{code-cell} ipython3
total = 0

for v in [21, 27, 33]:
    total = total + v

total
```

The function `range` is often used to run a loop a set number of times.

Like list indices, it starts from zero, so `range(3)` gives 0, 1 and 2.

```{code-cell} ipython3
for i in range(3):
    print(i)
```

To loop over a dictionary, we use `items`, which gives us each key and value together.

```{code-cell} ipython3
means = {'north': 27.0, 'south': 19.0, 'east': 14.0}

for region, m in means.items():
    print(region, m)
```

AI tools often replace a simple loop with a one-line [list comprehension](https://en.wikipedia.org/wiki/List_comprehension).

```{code-cell} ipython3
[v * 2 for v in [21, 27, 33]]
```

Read this as "make a list of `v * 2`, for each `v` in the list".

A comprehension can also include a condition, which keeps only some of the elements.

```{code-cell} ipython3
[v for v in [21, 27, 33] if v > 25]
```

```{exercise-start}
:label: pai_ex3
```
Without running this code, work out what it prints.

```{code-block} python3
total = 0

for i in range(4):
    total = total + i

print(total)
print([k * 2 for k in [1, 2, 3, 4] if k > 2])
```

```{exercise-end}
```

```{solution-start} pai_ex3
:class: dropdown
```

The loop adds 0, 1, 2 and 3, giving 6.

The comprehension keeps 3 and 4, and then doubles them.

```{code-cell} ipython3
total = 0

for i in range(4):
    total = total + i

print(total)
print([k * 2 for k in [1, 2, 3, 4] if k > 2])
```

```{solution-end}
```

## Functions

A [function](https://en.wikipedia.org/wiki/Function_(computer_programming)) is a named piece of code that takes some inputs and usually returns a result.

We have already used `len`, `sum` and `print`, which are built into Python.

Inputs are placed in parentheses, and some are given by name.

```{code-cell} ipython3
sorted([3, 1, 2], reverse=True)    # Sort from largest to smallest
```

Other functions come from libraries, which we load using `import`.

```{code-cell} ipython3
import math

math.sqrt(16)
```

You will also see forms like `import numpy as np`, which gives the library a shorter name, so that its functions are written `np.something`.

### Defining functions

We can define our own functions using `def`.

```{code-cell} ipython3
def mean(values):
    return sum(values) / len(values)

mean([1, 2, 3])
```

The indented lines are the body of the function, and they run only when the function is called.

The line starting with `return` gives back the result.

An input can have a default value, which is used if we do not supply one.

```{code-cell} ipython3
def scale(v, factor=2):
    return v * factor

print(scale(10))
print(scale(10, factor=3))
```

### Lambda functions

A [lambda function](https://en.wikipedia.org/wiki/Anonymous_function) is a short function written on one line.

This code defines the same function in two ways.

```{code-cell} ipython3
def square(v):
    return v**2

square = lambda v: v**2

square(3)
```

You mainly need to recognize these when one function is given to another, as in this code that sorts regions by their mean income.

```{code-cell} ipython3
means = {'north': 27.0, 'south': 19.0, 'east': 14.0}

sorted(means, key=lambda r: means[r])
```

```{exercise-start}
:label: pai_ex4
```
Without running this code, work out what it prints.

```{code-block} python3
def f(a, b=2):
    return a * b

print(f(3))
print(f(3, b=5))
```

```{exercise-end}
```

```{solution-start} pai_ex4
:class: dropdown
```

The first call uses the default value of `b`, giving 6, and the second gives 15.

```{code-cell} ipython3
def f(a, b=2):
    return a * b

print(f(3))
print(f(3, b=5))
```

```{solution-end}
```

## Reading the opening example

We can now return to the code at the start of the lecture.

```{code-cell} ipython3
incomes = {
    'north': [21, 27, 33, 30, 24],
    'south': [15, 23],
    'east': [12, 14, 19, 16, 10, 13],
    'west': [80, 90]
}

threshold = 25


def mean(values):
    return sum(values) / len(values)


below = []

for region, values in incomes.items():
    m = mean(values)
    print(region, m)
    if m < threshold:
        below.append(region)

print('Regions below the threshold:', below)
```

Here is what each part does.

1. `incomes` is a dictionary, where each key is a region and each value is a list of household incomes.
2. `threshold` is a number.
3. `mean` is a function that computes the mean of a list.
4. `below` starts as an empty list.
5. The `for` loop visits each region, computes its mean income and prints it.
6. The `if` adds the region to `below` when its mean is under the threshold.
7. The last line prints the result.

An AI might write the same steps more compactly, using a list comprehension.

```{code-cell} ipython3
[region for region, values in incomes.items() if mean(values) < threshold]
```

It might also use a lambda function, for example to sort the regions from lowest to highest mean income.

```{code-cell} ipython3
sorted(incomes, key=lambda r: mean(incomes[r]))
```

## Directing an AI

In this section we look at the main types of AI coding tool, a basic workflow for using them, and one way that their code can go wrong without any warning.

### Types of tool

AI coding tools come in three main forms.

* *Assistants inside the notebook*, such as the AI in Google Colab and [GitHub Copilot](https://en.wikipedia.org/wiki/GitHub_Copilot), suggest and explain code as you work.
* *Chat apps*, such as ChatGPT, [Claude](https://en.wikipedia.org/wiki/Claude_(language_model)) and Gemini, answer questions and write code that you then copy and run.
* *End-to-end tools* can write code, run it and fix errors with little supervision.

You can use any of these, and the free tier of a standard AI tool is enough for this course.

If you do not have one, the course's free fallback is the AI built into Colab.

In class we will briefly demonstrate Claude and the AI in Colab.

Before using any of them, read the discussion of confidentiality in {doc}`overview`.

### A basic workflow

Whichever tool you use, the same steps apply.

1. *State the question.* Describe the data and say exactly what you want.
2. *Read what comes back.* Use what you learned above, and ask the AI to explain any line you cannot follow.
3. *Run it.* Start with a small case where you already know the answer.
4. *Check the result.* Ask whether it answers the question you actually care about.

The last step is where your own judgment matters most.

### Code that runs but answers the wrong question

Suppose that we ask an AI for the average income in our table.

```{code-cell} ipython3
incomes = {
    'north': [21, 27, 33, 30, 24],
    'south': [15, 23],
    'east': [12, 14, 19, 16, 10, 13],
    'west': [80, 90]
}
```

The AI returns this code.

```{code-cell} ipython3
region_means = [mean(values) for values in incomes.values()]
sum(region_means) / len(region_means)
```

The code runs without error and looks reasonable.

But it computes the mean of the four regional means, which gives the two households in the west the same total weight as the six households in the east.

If we want the average income across all households, we need something different.

```{code-cell} ipython3
total = 0
count = 0

for values in incomes.values():
    total = total + sum(values)
    count = count + len(values)

total / count
```

The two numbers differ because they answer different questions.

No error message can tell you which one you wanted, since that depends on the purpose of the analysis.

This is why you need to state your question precisely, and to understand what the code computes.

We return to this kind of silent failure in {doc}`working_with_data`.

```{exercise-start}
:label: pai_ex5
```
You ask an AI to count the regions with mean income of at least 25, and it returns the code below.

Without running it, find the mistake.

```{code-block} python3
def mean(values):
    return sum(values) / len(values)


table = {'a': [25, 25], 'b': [10, 20], 'c': [30, 40]}

count = 0

for region, values in table.items():
    if mean(values) > 25:
        count = count + 1

print(count)
```

```{exercise-end}
```

```{solution-start} pai_ex5
:class: dropdown
```

"At least 25" includes regions with a mean of exactly 25, so the comparison should be `>=` rather than `>`.

The code above misses region `a` and prints 1, while the correct code prints 2.

```{code-cell} ipython3
def mean(values):
    return sum(values) / len(values)


table = {'a': [25, 25], 'b': [10, 20], 'c': [30, 40]}

count = 0

for region, values in table.items():
    if mean(values) >= 25:
        count = count + 1

print(count)
```

```{solution-end}
```

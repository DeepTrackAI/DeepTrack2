Getting started
===============


DeepTrack is a comprehensive deep learning framework for digital microscopy. 
We provide tools to create physical simulations of customizable optical systems, to generate and train neural network models, and to analyze experimental data.

Installation
------------

DeepTrack 2.0 requires at least python 3.6

To install DeepTrack 2.0, open a terminal or command prompt and run:

   pip install deeptrack


Basics
======

This section will explain core concepts about programming with DeepTrack 2.0. The section `Tutorials` introduces some notebooks which more directly solve problems with DeepTrack 2. Feel free to skip directly to those if you are not interested in customizing the examples.

For the following subsections we assume the following imports:

.. code-block:: python
   import deeptrack as dt
   import numpy as np


Features
--------

Features are the core computing blocks of DeepTrack 2, and operate mainly on numpy arrays. They have two special methods with which you can interact with them: `resolve` and `update`. For now, we only need to focus on `resolve`, which executes the feature. The first argument is the input of the feature, which does not always have to be specified. For example:
.. code-block:: python
    add_one = dt.Add(1)
    add_one.resolve(10)
    >>> 11

    
Here we create a feature that adds 1 to the input. We resolve that feature on the input 10, which returns 11. Simple!

Chaining features
-----------------

Features can be chained using the + operator. If two features are chained, the output of the first is passed as the input of the second.

.. code-block:: python
    add_one = dt.Add(1)
    multiply_two = dt.Multiply(2)
    add_and_multiply = add_one + multiply_two

    add_one.resolve(10)
    >>> 22


Here we have two features, `add_one`, which adds one to the input, and `multiply_two` which multiplies the input with two. We chain these features, which creates a single feature which first adds 1 to the input, and then multiplies it with two. Resolving this feature on the input 10 gives (10 + 2) * 2 = 22.

Executing a feature multiple times
----------------------------------


You will often find that you want to execute a single feature several times in a row. This is achieved using the ** operator. The integer on the right determines how many times to execute the feature on the left.

.. code-block:: python
    add_one = dt.Add(1)
    add_and_multiply = add_one ** 5

    add_one.resolve(10)
    >>> 15

Here, the `add_one` feature is executed five times. Resolving this feature on the input 10 gives 10 + 1 + 1 + 1 + 1 + 1 = 15

Using functions as arguments
----------------------------

The magic of DeepTrack is that any argument used to create feature can be replaced with a function that returns a value to use. The easiet example is a function that always returns the same value:

.. code-block:: python
    add_one = dt.Add(lambda: 1)
    add_one.resolve(10)
    >>> 11


However, this is not very exciting. let's do something more interesting:

.. code-block:: python
    add_zero_or_one = dt.Add(lambda: np.random.randint(2)
    add_zero_or_one.resolve(10)
    >>> 10


Here, the value we add is randomly either 0 or 1. This time, it added zero. We can check this explicitly by running

.. code-block:: python
    add_zero_or_one.value.current_value
    >>> 0


The first input of the Add feature is `value`, which we access and request its current value. Other features may have different names for their arguments, and all arguments are named arguments. 

To request a new value, we call `update`. Until update is called, it will always use the same value.

.. code-block:: python
    add_zero_or_one.update()
    add_zero_or_one.value.current_value
    >>> 1


Dummy arguments
---------------

Here, we will introduce a concept that might seem useless at the time, but will make more sense in the next subsection: dummy arguments, or dummy properties. A dummy property is a argument passed to a feature that is not directly used by the feature. These can have any value and any name (that is not any of the feature's input arguments).

.. code-block:: python
    add_one = dt.Add(
        value=1,
        useless_argument="I do nothing"
    )
    add_one.resolve(10)
    >>> 11


Here `useless_argument` is a dummy property, as it is not directly used by the Add feature

Dependent arguments
-------------------

Now, we will show the use of dummy properties: ordinary arguments can depend on them! This means that arguments that are functions can take them as input:

.. code-block:: python
    add_random_integer = dt.Add(
    min_value=lambda: np.random.randint(100),
    max_value=lambda min_value: np.random.randint(min_value + 1, 101),
    value=lambda min_value, max_value: np.random.randint(min_value, max_value + 1)
    )

    add_random_integer.update().resolve(10)
    >>> 76

Let's break this example down. We define `min_value`, which is a random integer between 0 and 99, which defines the minimum value to add. We also define `max_value` which takes `min_value` as an input, and returns a random integer between `min_value + 1` and 100. Finally, we define `value`, which is the argument used by the Add feature to determine the value to add. It takes `min_value` and `max_value` as inputs, and returns a random integer between `min_value` and `max_value`.

Dependencies between features
-----------------------------

A feature can dependent on the arguments of another feature. The syntax for this is simple:

.. code-block:: python
    add_one_or_two = dt.Add(value=np.random.randint(1, 3))
    undo_add = dt.Subtract(
        value=add_one_or_two.value
    )

    do_nothing = add_one_or_two + undo_add
    do_nothing.update().resolve(10)
    >>> 10


These two arguments will now always be the same. You can of course accept it as a dummy property:

.. code-block:: python
    add_one_or_two = dt.Add(value=np.random.randint(1, 3))
    undo_add = dt.Subtract(
        value_added=add_one_or_two.value,
        value=lambda value_added: value_added
    )


It is also possible to inherit all the arguments of another feature by calling

.. code-block:: python
    add_one_or_two = dt.Add(value=np.random.randint(1, 3))
    undo_add = dt.Subtract(
        **add_one_or_two.properties
    )


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


Here we have two features, `add_one`, which adds one to the input, and `multiply_two` which multiplies the input with two. We chain these features, which creates a single feature which first adds 1 to the input, 
and then multiplies it with two. Resolving this feature on the input 10 gives (10 + 1) * 2 = 22.

Executing a feature multiple times
----------------------------------


You will often find that you want to execute a single feature several times in a row. This is achieved using the ** operator. The integer on the right determines how many times to execute the feature on the left.

.. code-block:: python

    add_one = dt.Add(1)
    add_one_five_times = add_one ** 5

    add_one_five_times.resolve(10)
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

    add_zero_or_one = dt.Add(lambda: np.random.randint(2))
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


Done!
-----

That's it! You're now ready to start playing with DeepTrack 2.0. However, you're encouraged to look at the next section to learn more powerful interactions, such as dependencies between arguments or features!
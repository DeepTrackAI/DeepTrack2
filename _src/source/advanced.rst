
.. role:: python(code)
   :language: python


Advanced Topics
===============

DeepTrack 2.1 allows much more rich interactions than what is exposed on the surface. This section will explain how to define dependent variables, both within and between features. We'll also explain how these can be more explicitly controlled after the entire pipeline has been created.

Dummy arguments
---------------

First, we will introduce a concept that might seem useless at the time, but will make more sense in the next subsection: dummy arguments, or dummy properties. A dummy property is a argument passed to a feature that is not directly used by the feature. These can have any value and any name (that is not any of the feature's input arguments).

.. code-block:: python

    add_one = dt.Add(
        value=1,
        useless_argument="I do nothing"
    )
    add_one(10)
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

    add_random_integer(10)
    >>> 76

Let's break this example down. We define :python:`min_value`, which is a random integer between 0 and 99, which defines the minimum value to add. We also define :python:`max_value` which takes :python:`min_value` as an input, and returns a random integer between `min_value + 1` and 100. Finally, we define `value`, which is the argument used by the Add feature to determine the value to add. It takes `min_value` and `max_value` as inputs, and returns a random integer between `min_value` and `max_value`.

Dependencies between features
-----------------------------

A feature can dependent on the arguments of another feature. The syntax for this is simple:

.. code-block:: python

    add_one_or_two = dt.Add(value=lambda: np.random.randint(1, 3))
    undo_add = dt.Subtract(
        value=add_one_or_two.value
    )

    do_nothing = add_one_or_two >> undo_add
    do_nothing(10)
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

These dependency trees can be any size you want, as long as there are no cyclic dependencies. 

Overriding properties
---------------------

You might find yourself wanting more direct control over the properties in a pipeline. For example, 
you might want to resolve the same image twice but with different levels of noise, 
or maybe you want to use the same pipeline for both training set and the validation set, but with 
slightly different properties. We provide two methods of achieving this, both with their own benefits
and use-cases.

 
Overriding properties during with resolve 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
It is possible override the value of properties when resolving a feature. There are many valuable uses for this, particularly when investigating the behaviour of a pipeline.
This is achieved simply by passing the name of the property as a keyword argument.

.. code-block:: python

    add_one = dt.Add(value=1)
    add_one(10, value=2)
    >>> 12

Note that this will override all properties with the name "value". To get more precise targeting, you can either make use of dummy properties:

.. code-block:: python

    add_one = dt.Add(value=1)
    subtract_one = dt.Add(value=lambda: value_to_subtract, value_to_subtract=1)
    pipeline = add_one >> subtract_one
    pipeline(10, value_to_subtract=0)
    >>> 11

or, using dt.Arguments:

.. code-block:: python

    arguments = dt.Arguments(value_to_add=1, value_to_subtract=1)
    add_one = dt.Add(value=arguments.value_to_add)
    subtract_one = dt.Add(value=arguments.value_to_subtract)
    pipeline = add_one >> subtract_one

    pipeline.add_arugments(arguments)
    add_one(10, value_to_subtract=0)
    >>> 11

In the second case, you also constrain the permitted keyword arguments passed to the feature.
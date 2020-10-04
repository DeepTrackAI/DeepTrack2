
.. role:: python(code)
   :language: python


Advanced Topics
===============

DeepTrack 2.0 allows much more rich interactions than what is exposed on the surface. This section will explain how to define dependent variables, both within and between features. We'll also explain how these can be more explicitly controlled after the entire pipeline has been created.

Dummy arguments
---------------

First, we will introduce a concept that might seem useless at the time, but will make more sense in the next subsection: dummy arguments, or dummy properties. A dummy property is a argument passed to a feature that is not directly used by the feature. These can have any value and any name (that is not any of the feature's input arguments).

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

Let's break this example down. We define :python:`min_value`, which is a random integer between 0 and 99, which defines the minimum value to add. We also define :python:`max_value` which takes :python:`min_value` as an input, and returns a random integer between `min_value + 1` and 100. Finally, we define `value`, which is the argument used by the Add feature to determine the value to add. It takes `min_value` and `max_value` as inputs, and returns a random integer between `min_value` and `max_value`.

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

These dependency trees can be any size you want, as long as there are no cyclic dependencies. 

Overriding properties
---------------------

You might find yourself wanting more direct control over the properties in a pipeline. For example, 
you might want to resolve the same image twice but with different levels of noise, 
or maybe you want to use the same pipeline for both training set and the validation set, but with 
slightly different properties. We provide two methods of achieving this, both with their own benefits
and use-cases.


Overriding with update 
^^^^^^^^^^^^^^^^^^^^^^

When calling :python:`Feature.update()`, you are free to pass keyword arguments. These can be thought of as global
properties that are true for all features, overriding local values. As an example:

.. code-block:: python

    add_one = dt.Add(value=1)
    add_one.update(value=2)
    add_one.resolve(10)
    >>> 12

By passing `value=2` when updating, that value overwrote the internal value of add_one!

.. note::
    If two features share the same name for a property, they will both be overwritten. For example, if 
    we had both :python:`dt.Add(value=1)` and :python:`dt.Subtract(value=1)`, both of them would get 
    :python:`value` from the update call. This can be both an advantage or a disadvantage. For example
    it makes it easy to set the out of plane position of each scatterer to zero, or to remove all 
    averrations. On the other hand, it can lead to confusing behaviour if the name conflict isn't known!

A common use for this is to differentiate between different sets of data:

.. code-block:: python 

    validation_paths = iter(['./validation/file_1', './validation/file_2', './validation/file_3'])
    training_paths = iter(['./training/file_1', './training/file_2', './training/file_3'])

    loader = dt.LoadImage(
        path=lambda is_validation: next(validation_paths) if is_validation \
                              else next(training_paths),
    )

    loader.update().path.current_value
    >>> './training/file_1'

    loader.update(is_validation=True).path.current_value

    >>> './validation/file_1'

Eagle-eyed readers may have noticed that the loader ran, even though :python:`is_validation` should have been undefined. How come it didn't crash?
DeepTrack handles this internally by looking at the specifications of the function. If the function takes an argument for which there is no local nor global
value to pass, DeepTrack first tries not passing the value (which would work for a definition like :python:`def path(is_validation=False)`). If this fails, then
:python:`None` is passed instead.

.. note::
    Since :python:`None` is passed to arguments that do not exist, you may encounter excpetions of the type :python:`+ not defined for NoneType...`. If that's the case
    check if you have misspelled any property!

 
Overriding with resolve 
^^^^^^^^^^^^^^^^^^^^^^^

A very similar approach is possible directly to the method :python:`resolve`, but it behaves slightly differently. Primarly, all other properties, 
even dependent ones, remain unchanged. So if property :python:`Y` is set to be :python:`X+1`, then :python:`.resolve(X=0)` would leave :python:`Y` unchanged. 

This allows you to resolve the exact same image twice, while changing some parameter of the process. For example

.. code-block:: python

    add_one = dt.Add(value=1)
    add_one.resolve(10, value=2)
    >>> 12

A good use-case for this is to create a network label. One could imagine resolving the same image again, but with every particle exactly in focus, or without 
aberrations, or without noise.

.. note:: 
    
    A more targetted solution is provided by the features `ConditionalSetFeature <features.html#conditionalsetfeature>`_ and `ConditionalSetProperty <features.html#conditionalsetproperty>`_
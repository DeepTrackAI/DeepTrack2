backend
=======

.. automodule:: deeptrack.backend

_config
-------

Module classes
<<<<<<<<<<<<<<

Config
^^^^^^

.. autoclass:: deeptrack.backend.Config
   :members:
   :exclude-members: get, add_child, add_dependency, add_feature, invalidate, is_valid, recurse_children, recurse_dependencies, seed, set_value, valid_index, validate, sample, action, resolve, previous, current_value, store

citations
---------

core
----

Module classes
<<<<<<<<<<<<<<

DeepTrackDataDict
^^^^^^^^^^^^^^^^^

.. autoclass:: deeptrack.backend.DeepTrackDataDict
   :members:
   :exclude-members: get, add_child, add_dependency, add_feature, invalidate, is_valid, recurse_children, recurse_dependencies, seed, set_value, valid_index, validate, sample, action, resolve, previous, current_value, store

DeepTrackDataObject
^^^^^^^^^^^^^^^^^^^

.. autoclass:: deeptrack.backend.DeepTrackDataObject
   :members:
   :exclude-members: get, add_child, add_dependency, add_feature, invalidate, is_valid, recurse_children, recurse_dependencies, seed, set_value, valid_index, validate, sample, action, resolve, previous, current_value, store

DeepTrackNode
^^^^^^^^^^^^^

.. autoclass:: deeptrack.backend.DeepTrackNode
   :members:
   :exclude-members: get, add_child, add_dependency, add_feature, invalidate, is_valid, recurse_children, recurse_dependencies, seed, set_value, valid_index, validate, sample, action, resolve, previous, current_value, store

Module functions
<<<<<<<<<<<<<<<<

create_node_with_operator
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: deeptrack.backend.create_node_with_operator

equivalent
^^^^^^^^^^

.. autofunction:: deeptrack.backend.equivalent

mie
---

Module functions
<<<<<<<<<<<<<<<<

mie_coefficients
^^^^^^^^^^^^^^^^

.. autofunction:: deeptrack.backend.mie_coefficients

mie_harmonics
^^^^^^^^^^^^^

.. autofunction:: deeptrack.backend.mie_harmonics

stratified_mie_coefficients
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: deeptrack.backend.stratified_mie_coefficients

pint_definition
---------------

polynomials
-----------

Module functions
<<<<<<<<<<<<<<<<

besselj
^^^^^^^

.. autofunction:: deeptrack.backend.besselj

bessely
^^^^^^^

.. autofunction:: deeptrack.backend.bessely

dbesselj
^^^^^^^^

.. autofunction:: deeptrack.backend.dbesselj

dbessely
^^^^^^^^

.. autofunction:: deeptrack.backend.dbessely

dricbesh
^^^^^^^^

.. autofunction:: deeptrack.backend.dricbesh

dricbesj
^^^^^^^^

.. autofunction:: deeptrack.backend.dricbesj

dricbesy
^^^^^^^^

.. autofunction:: deeptrack.backend.dricbesy

ricbesh
^^^^^^^

.. autofunction:: deeptrack.backend.ricbesh

ricbesj
^^^^^^^

.. autofunction:: deeptrack.backend.ricbesj

ricbesy
^^^^^^^

.. autofunction:: deeptrack.backend.ricbesy

tensorflow_bindings
-------------------

Module functions
<<<<<<<<<<<<<<<<

implements_tf
^^^^^^^^^^^^^

.. autofunction:: deeptrack.backend.implements_tf

units
-----

Module classes
<<<<<<<<<<<<<<

ConversionTable
^^^^^^^^^^^^^^^

.. autoclass:: deeptrack.backend.ConversionTable
   :members:
   :exclude-members: get, add_child, add_dependency, add_feature, invalidate, is_valid, recurse_children, recurse_dependencies, seed, set_value, valid_index, validate, sample, action, resolve, previous, current_value, store


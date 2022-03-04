properties
==========

.. automodule:: deeptrack.properties

Module classes
<<<<<<<<<<<<<<

Property
^^^^^^^^

.. autoclass:: deeptrack.properties.Property
   :members:
   :exclude-members: get, add_child, add_dependency, add_feature, invalidate, is_valid, recurse_children, recurse_dependencies, seed, set_value, valid_index, validate, sample, action, resolve, previous, current_value, store

PropertyDict
^^^^^^^^^^^^

.. autoclass:: deeptrack.properties.PropertyDict
   :members:
   :exclude-members: get, add_child, add_dependency, add_feature, invalidate, is_valid, recurse_children, recurse_dependencies, seed, set_value, valid_index, validate, sample, action, resolve, previous, current_value, store

SequentialProperty
^^^^^^^^^^^^^^^^^^

.. autoclass:: deeptrack.properties.SequentialProperty
   :members:
   :exclude-members: get, add_child, add_dependency, add_feature, invalidate, is_valid, recurse_children, recurse_dependencies, seed, set_value, valid_index, validate, sample, action, resolve, previous, current_value, store

Module functions
<<<<<<<<<<<<<<<<

propagate_data_to_dependencies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: deeptrack.properties.propagate_data_to_dependencies


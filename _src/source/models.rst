models
======

.. automodule:: deeptrack.models

cgan
----

Module functions
<<<<<<<<<<<<<<<<

CGAN
^^^^

.. autofunction:: deeptrack.models.CGAN

cyclegan
--------

dense
-----

Module classes
<<<<<<<<<<<<<<

FullyConnected
^^^^^^^^^^^^^^

.. autoclass:: deeptrack.models.FullyConnected
   :members:
   :exclude-members: get, add_child, add_dependency, add_feature, invalidate, is_valid, recurse_children, recurse_dependencies, seed, set_value, valid_index, validate, sample, action, resolve, previous, current_value, store

embeddings
----------

Module classes
<<<<<<<<<<<<<<

ClassToken
^^^^^^^^^^

.. autoclass:: deeptrack.models.ClassToken
   :members:
   :exclude-members: get, add_child, add_dependency, add_feature, invalidate, is_valid, recurse_children, recurse_dependencies, seed, set_value, valid_index, validate, sample, action, resolve, previous, current_value, store

LearnablePositionEmbs
^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: deeptrack.models.LearnablePositionEmbs
   :members:
   :exclude-members: get, add_child, add_dependency, add_feature, invalidate, is_valid, recurse_children, recurse_dependencies, seed, set_value, valid_index, validate, sample, action, resolve, previous, current_value, store

Module functions
<<<<<<<<<<<<<<<<

ClassTokenLayer
^^^^^^^^^^^^^^^

.. autofunction:: deeptrack.models.ClassTokenLayer

LearnablePositionEmbsLayer
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: deeptrack.models.LearnablePositionEmbsLayer

equivariances
-------------

Module classes
<<<<<<<<<<<<<<

Equivariance
^^^^^^^^^^^^

.. autoclass:: deeptrack.models.Equivariance
   :members:
   :exclude-members: get, add_child, add_dependency, add_feature, invalidate, is_valid, recurse_children, recurse_dependencies, seed, set_value, valid_index, validate, sample, action, resolve, previous, current_value, store

LogScaleEquivariance
^^^^^^^^^^^^^^^^^^^^

.. autoclass:: deeptrack.models.LogScaleEquivariance
   :members:
   :exclude-members: get, add_child, add_dependency, add_feature, invalidate, is_valid, recurse_children, recurse_dependencies, seed, set_value, valid_index, validate, sample, action, resolve, previous, current_value, store

Rotational2DEquivariance
^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: deeptrack.models.Rotational2DEquivariance
   :members:
   :exclude-members: get, add_child, add_dependency, add_feature, invalidate, is_valid, recurse_children, recurse_dependencies, seed, set_value, valid_index, validate, sample, action, resolve, previous, current_value, store

ScaleEquivariance
^^^^^^^^^^^^^^^^^

.. autoclass:: deeptrack.models.ScaleEquivariance
   :members:
   :exclude-members: get, add_child, add_dependency, add_feature, invalidate, is_valid, recurse_children, recurse_dependencies, seed, set_value, valid_index, validate, sample, action, resolve, previous, current_value, store

TranslationalEquivariance
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: deeptrack.models.TranslationalEquivariance
   :members:
   :exclude-members: get, add_child, add_dependency, add_feature, invalidate, is_valid, recurse_children, recurse_dependencies, seed, set_value, valid_index, validate, sample, action, resolve, previous, current_value, store

gan
---

Module functions
<<<<<<<<<<<<<<<<

GAN
^^^

.. autofunction:: deeptrack.models.GAN

gans
----

generators
----------

Module classes
<<<<<<<<<<<<<<

LodeSTARGenerator
^^^^^^^^^^^^^^^^^

.. autoclass:: deeptrack.models.LodeSTARGenerator
   :members:
   :exclude-members: get, add_child, add_dependency, add_feature, invalidate, is_valid, recurse_children, recurse_dependencies, seed, set_value, valid_index, validate, sample, action, resolve, previous, current_value, store

gnns
----

lodestar
--------

models
------

Module classes
<<<<<<<<<<<<<<

LodeSTAR
^^^^^^^^

.. autoclass:: deeptrack.models.LodeSTAR
   :members:
   :exclude-members: get, add_child, add_dependency, add_feature, invalidate, is_valid, recurse_children, recurse_dependencies, seed, set_value, valid_index, validate, sample, action, resolve, previous, current_value, store

LodeSTARBaseModel
^^^^^^^^^^^^^^^^^

.. autoclass:: deeptrack.models.LodeSTARBaseModel
   :members:
   :exclude-members: get, add_child, add_dependency, add_feature, invalidate, is_valid, recurse_children, recurse_dependencies, seed, set_value, valid_index, validate, sample, action, resolve, previous, current_value, store

pcgan
-----

Module functions
<<<<<<<<<<<<<<<<

PCGAN
^^^^^

.. autofunction:: deeptrack.models.PCGAN

recurrent
---------

Module classes
<<<<<<<<<<<<<<

RNN
^^^

.. autoclass:: deeptrack.models.RNN
   :members:
   :exclude-members: get, add_child, add_dependency, add_feature, invalidate, is_valid, recurse_children, recurse_dependencies, seed, set_value, valid_index, validate, sample, action, resolve, previous, current_value, store

rnn
^^^

.. autoclass:: deeptrack.models.rnn
   :members:
   :exclude-members: get, add_child, add_dependency, add_feature, invalidate, is_valid, recurse_children, recurse_dependencies, seed, set_value, valid_index, validate, sample, action, resolve, previous, current_value, store

utils
-----

Module classes
<<<<<<<<<<<<<<

KerasModel
^^^^^^^^^^

.. autoclass:: deeptrack.models.KerasModel
   :members:
   :exclude-members: get, add_child, add_dependency, add_feature, invalidate, is_valid, recurse_children, recurse_dependencies, seed, set_value, valid_index, validate, sample, action, resolve, previous, current_value, store

Model
^^^^^

.. autoclass:: deeptrack.models.Model
   :members:
   :exclude-members: get, add_child, add_dependency, add_feature, invalidate, is_valid, recurse_children, recurse_dependencies, seed, set_value, valid_index, validate, sample, action, resolve, previous, current_value, store

Module functions
<<<<<<<<<<<<<<<<

LoadModel
^^^^^^^^^

.. autofunction:: deeptrack.models.LoadModel

as_KerasModel
^^^^^^^^^^^^^

.. autofunction:: deeptrack.models.as_KerasModel

as_activation
^^^^^^^^^^^^^

.. autofunction:: deeptrack.models.as_activation

as_normalization
^^^^^^^^^^^^^^^^

.. autofunction:: deeptrack.models.as_normalization

compile
^^^^^^^

.. autofunction:: deeptrack.models.compile

load_model
^^^^^^^^^^

.. autofunction:: deeptrack.models.load_model

register_config
^^^^^^^^^^^^^^^

.. autofunction:: deeptrack.models.register_config

single_layer_call
^^^^^^^^^^^^^^^^^

.. autofunction:: deeptrack.models.single_layer_call

with_citation
^^^^^^^^^^^^^

.. autofunction:: deeptrack.models.with_citation


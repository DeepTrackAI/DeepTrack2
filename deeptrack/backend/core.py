import numpy as np
import operator


class DeepTrackDataObject:
    def __init__(self):
        self.data = None
        self.valid = False
        self.propagated = False

    def store(self, data):
        self.propagated = False
        self.valid = True
        self.data = data

    def current_value(self):
        return self.data

    def is_valid(self):
        return self.valid

    def invalidate(self):
        self.valid = False

    def validate(self):
        self.valid = True


class DeepTrackNode:

    __nonelike_default = object()

    def __init__(self, action=__nonelike_default, _static=False, **kwargs):
        self.data = DeepTrackDataObject()
        self.children = []
        self.dependencies = []

        if action is not self.__nonelike_default:

            if callable(action):

                self.action = action
            else:
                self.action = lambda: action

        super(DeepTrackNode, self).__init__(**kwargs)

    def add_child(self, other):
        if other not in self.children:
            self.children.append(other)

    def add_dependency(self, other):
        if other not in self.dependencies:
            self.dependencies.append(other)

    def store(self, data):
        return self.data.store(data)

    def is_valid(self):
        return self.data.is_valid()

    def invalidate(self):
        if self.is_valid():
            self.data.invalidate()
            for child in self.children:
                child.invalidate()

    def validate(self):
        if not self.is_valid():
            self.data.validate()
            for child in self.children:
                child.validate()

    def set_value(self, value):

        # If set to same value, no need to invalidate
        if (
            (value is not self.data.current_value())
            and (id(value) != id(self.data.current_value()))
            and not (np.array_equal(value, self.data.current_value()))
        ):

            self.invalidate()
            self.store(value)

    def update(self):
        self.invalidate()
        for dependency in self.dependencies:
            dependency.update()

    def __call__(self):

        if not self.is_valid():

            new_value = self.action()

            self.store(new_value)

        return self.data.current_value()

    def previous(self):
        return self.data.current_value()

    def __add__(self, other):
        return create_node_with_operator(operator.__add__, self, other)

    def __radd__(self, other):
        return create_node_with_operator(operator.__add__, other, self)

    def __sub__(self, other):
        return create_node_with_operator(operator.__sub__, self, other)

    def __rsub__(self, other):
        return create_node_with_operator(operator.__sub__, other, self)

    def __mul__(self, other):
        return create_node_with_operator(operator.__mul__, self, other)

    def __rmul__(self, other):
        return create_node_with_operator(operator.__mul__, other, self)

    def __truediv__(self, other):
        return create_node_with_operator(operator.__truediv__, self, other)

    def __rtruediv__(self, other):
        return create_node_with_operator(operator.__truediv__, other, self)

    def __floordiv__(self, other):
        return create_node_with_operator(operator.__floordiv__, self, other)

    def __rfloordiv__(self, other):
        return create_node_with_operator(operator.__floordiv__, other, self)

    def __lt__(self, other):
        return create_node_with_operator(operator.__lt__, self, other)

    def __rlt__(self, other):
        return create_node_with_operator(operator.__lt__, other, self)

    def __gt__(self, other):
        return create_node_with_operator(operator.__gt__, self, other)

    def __rgt__(self, other):
        return create_node_with_operator(operator.__gt__, other, self)

    def __le__(self, other):
        return create_node_with_operator(operator.__le__, self, other)

    def __rle__(self, other):
        return create_node_with_operator(operator.__le__, other, self)

    def __ge__(self, other):
        return create_node_with_operator(operator.__ge__, self, other)

    def __rge__(self, other):
        return create_node_with_operator(operator.__ge__, other, self)


def create_node_with_operator(op, a, b):

    if not issubclass(a, DeepTrackNode):
        a = DeepTrackNode(a)

    if not issubclass(b, DeepTrackNode):
        b = DeepTrackNode(b)

    new = DeepTrackNode(lambda: op(a(), b()))
    new.add_dependency(a)
    new.add_dependency(b)
    a.add_child(new)
    b.add_child(new)

    return new

import sys 
import os
import inspect

sys.path.append(os.path.abspath("../"))

PATH_TO_SRC = os.path.abspath("./source")

# MODULE TO CONFIGURE
import deeptrack

module = deeptrack

head_file = open(os.path.join(PATH_TO_SRC, module.__name__ + ".rst"), "w")
head_file.write(".. toctree::\n   :maxdepth: 4\n   \n")
head_file.flush()


# Get all submodules. Non-recursive, but could be made so
for _, submodule in inspect.getmembers(module, lambda x: inspect.ismodule(x) and getattr(deeptrack, x.__name__.split(".")[-1], False)):
    print("Adding package ", submodule.__name__)

    # Add to head
    

    submodule_name = submodule.__name__.split(".")[-1]
    submodule_path = module.__name__ + "." + submodule_name

    head_file.write("   " + submodule_name + "\n")
    head_file.flush()

    submodule_file = open(os.path.join(PATH_TO_SRC, submodule_name + ".rst"), "w")
   

    submodule_file.write(submodule_name + "\n" + "=" * len(submodule_name) + "\n\n" + ".. automodule:: " + submodule.__name__ + "\n\n")
    submodule_file.flush()

    
    submodule_classes = inspect.getmembers(submodule, lambda x: inspect.isclass(x) and x.__module__ == submodule.__name__)

    if submodule_classes:

        submodule_file.write("Module classes\n--------------\n\n")
        submodule_file.flush()

        for name, member in submodule_classes:
            member_name = submodule_path + "." + name
            submodule_file.write(name + "\n" + "^" * len(name) + "\n\n" + ".. autoclass:: " + member_name + "\n   :members:\n\n")
            submodule_file.flush()

    # ADD FUNCTIONS

    submodule_funcs = inspect.getmembers(submodule, lambda x: inspect.isfunction(x) and x.__module__ == submodule.__name__)

    if submodule_funcs:

        submodule_file.write("Module functions\n----------------\n\n")
        submodule_file.flush()

        for name, member in submodule_funcs:
            member_name = submodule_path + "." + name
            submodule_file.write(name + "\n" + "^" * len(name) + "\n\n" + ".. autofunction:: " + member_name + "\n\n")
            submodule_file.flush()
    
    submodule_file.close()

head_file.close()

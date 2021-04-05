import sys
import os
import inspect

sys.path.append(os.path.abspath("../"))

PATH_TO_SRC = os.path.abspath("./source")

# MODULE TO CONFIGURE
import deeptrack
import deeptrack.backend


head_file = open(os.path.join(PATH_TO_SRC, deeptrack.__name__ + ".rst"), "w")
head_file.write("Documentation\n=============\n\n.. toctree::\n   :maxdepth: 1\n   \n")
head_file.flush()

added = []
# Get all submodules. Non-recursive, but could be made so


def get_submodules(module):
    return inspect.getmembers(
        module,
        lambda x: inspect.ismodule(x)
        and getattr(module, x.__name__.split(".")[-1], False)
        and x.__name__.startswith(module.__name__)
        and x.__name__ != module.__name__
        and x.__name__ not in added,
    )


for _, submodule in get_submodules(deeptrack):
    if submodule.__name__ in added:
        continue

    print("Adding package ", submodule.__name__)
    added.append(submodule.__name__)
    # Add to head

    submodule_name = submodule.__name__.split(".")[-1]
    submodule_path = deeptrack.__name__ + "." + submodule_name

    head_file.write("   " + submodule_name + "\n")
    head_file.flush()

    submodule_file = open(os.path.join(PATH_TO_SRC, submodule_name + ".rst"), "w")

    submodule_file.write(
        submodule_name
        + "\n"
        + "=" * len(submodule_name)
        + "\n\n"
        + ".. automodule:: "
        + submodule_path
        + "\n\n"
    )
    submodule_file.flush()

    subsubmodules = get_submodules(submodule) or [("", None)]

    for ss_name, subsubmodule in subsubmodules:
        if not subsubmodule:
            subsubmodule = submodule
        else:
            ss_name = ss_name.split(".")[-1]
            submodule_file.write(ss_name + "\n" + "-" * len(ss_name) + "\n\n")

        submodule_classes = inspect.getmembers(
            subsubmodule,
            lambda x: inspect.isclass(x) and x.__module__ == subsubmodule.__name__,
        )

        if submodule_classes:

            submodule_file.write("Module classes\n<<<<<<<<<<<<<<\n\n")
            submodule_file.flush()

            for name, member in submodule_classes:
                member_name = submodule_path + "." + name
                submodule_file.write(
                    name
                    + "\n"
                    + "^" * len(name)
                    + "\n\n"
                    + ".. autoclass:: "
                    + member_name
                    + "\n   :members:"
                    + "\n   :exclude-members: get\n\n"
                )
                submodule_file.flush()

        # ADD FUNCTIONS

        submodule_funcs = inspect.getmembers(
            subsubmodule,
            lambda x: inspect.isfunction(x)
            and x.__module__ == subsubmodule.__name__
            and x.__name__[0] != "_",
        )

        if submodule_funcs:

            submodule_file.write("Module functions\n<<<<<<<<<<<<<<<<\n\n")
            submodule_file.flush()

            for name, member in submodule_funcs:
                member_name = submodule_path + "." + name
                submodule_file.write(
                    name
                    + "\n"
                    + "^" * len(name)
                    + "\n\n"
                    + ".. autofunction:: "
                    + member_name
                    + "\n\n"
                )
                submodule_file.flush()

    submodule_file.close()

head_file.close()

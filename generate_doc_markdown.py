import os
import sys


def parse_args(args):
    """Parse command line arguments.

    Arguments
    ---------
    - library_name (positional)
    - --force or -f (optional)
    - --exclude or -e (optional, comma-separated list of components to exclude)
    - --output-dir or -o (optional, directory to write output files to; 
                          defaults to 'src')
    
    Returns
    -------
        tuple: (library_name, force, exclude, output_dir)
    
    """

    library_name = None
    force = False
    exclude = []
    output_dir = "src"

    positional = []
    i = 0
    while i < len(args):
        arg = args[i]
        if arg.startswith("--exclude="):
            exclude = arg.split("=", 1)[1].split(",")
            exclude = [x.strip() for x in exclude if x.strip()]
        elif arg == "--exclude" or arg == "-e":
            # Next argument should be the comma-separated modules to exclude.
            if i + 1 < len(args):
                exclude = args[i + 1].split(",")
                exclude = [x.strip() for x in exclude if x.strip()]
                i += 1
            else:
                print(
                    "Error: --exclude requires a comma-separated list, "
                    "e.g. --exclude=mod1,mod2"
                )
                sys.exit(1)
        elif arg == "--force" or arg == "-f":
            force = True
        elif arg.startswith("--output-dir="):
            output_dir = arg.split("=", 1)[1].strip()
        elif arg == "--output-dir" or arg == "-o":
            # Next argument should be the directory.
            if i + 1 < len(args):
                output_dir = args[i + 1].strip()
                i += 1
            else:
                print(
                    "Error: --output-dir requires a directory name, "
                    "e.g. --output-dir=docs"
                )
                sys.exit(1)
        else:
            positional.append(arg)
        i += 1

    if len(positional) < 1:
        print(
            "Usage: python generate_docs.py <library_name> [--force|-f] "
            "[--exclude|-e=mod1,mod2,...] [--output-dir|-o <dir>]"
        )
        sys.exit(1)

    library_name = positional[0]

    return library_name, force, exclude, output_dir


def list_modules_and_packages(directory, exclude_list=None):
    """Extract a list of all modules and packages.

    Extracts a list of all modules (.py files except __init__.py) and packages 
    (subdirectories with __init__.py). It excludes files and directories
    starting with _.

    Parameters
    ----------
        directory (str): The path to the directory to search.
        exclude_list (list of str): Excluded modules and packages.

    Returns
    -------
        tuple: A tuple containing two lists:
            - modules (list of str): Names of all .py files excluding 
                                     __init__.py or starting with _.
            - packages (list of str): Names of all subdirectories containing an 
                                      __init__.py file. It excludes directories
                                      starting with _.
    
    """

    modules = []
    packages = []

    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)

        # Check for Python packages (directories with __init__.py).
        if (os.path.isdir(item_path) 
            and os.path.isfile(os.path.join(item_path, "__init__.py"))
            and not item.startswith("_")):
            packages.append(item)

        # Check for Python modules (.py files excluding __init__.py).
        elif (os.path.isfile(item_path) and item.endswith(".py") 
              and item != "__init__.py" and not item.startswith("_")):
                modules.append(item[:-3])  # Remove the .py extension.

    # Filter out excluded modules and packages.
    if exclude_list:
        packages = [p for p in packages if p not in exclude_list]
        modules = [m for m in modules if m not in exclude_list]

    # Sort the lists.
    modules.sort()
    packages.sort()

    return modules, packages


def get_components(directory, base="", exclude_list=None):
    """

    Parameters
    ----------
        directory (str): The path to the directory to search.
        base (str): Initial part of the name of the component.
        exclude_list (list of str): Excluded modules and packages.

    Returns
    -------
        list of str: A list containing all components to be documented.

    """

    modules, packages = list_modules_and_packages(directory, exclude_list)

    # Recurse on packages.
    components = modules
    for package in packages:
        components += get_components(
            os.path.join(directory, package),
            base=base + package + '.',
            exclude_list=exclude_list,
        )

    return [base + s for s in components]


def main():
    """Gdenerate rst files."""

    library_name, force, exclude_list, output_dir = parse_args(sys.argv[1:])

    base_dir = os.path.join("release-code", library_name)

    if not os.path.isdir(base_dir):
        print(f"Directory {base_dir} does not exist.")
        sys.exit(1)

    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get componets.
    components = get_components(base_dir, base="", exclude_list=exclude_list)

    # Create Documentation.rst.
    doc_filename = os.path.join(output_dir, "Documentation.rst")
    if os.path.exists(doc_filename) and not force:
        print(
            f"{doc_filename} already exists. "
            "Skipping (no --force flag provided)."
        )
    else:
        with open(doc_filename, "w", encoding="utf-8") as doc_file:
            doc_file.write("Documentation\n")
            doc_file.write("=============\n\n")
            doc_file.write(
                f"Here, you will find the documentation for {library_name}.\n"
                "The documentation is organized into the following sections:\n"
                "\n"
                ".. toctree::\n"
                "   :maxdepth: 1\n"
                "   :caption: Contents:\n"
                "\n"
            )
            for component in components:
                doc_file.write(f"   {component}\n")
            doc_file.write("\n")
        print(f"Created {doc_filename}")        

    # Create a .rst file for each component.
    for component in components:
        rst_filename = os.path.join(output_dir, f"{component}.rst")
        if os.path.exists(rst_filename) and not force:
            print(
                f"{rst_filename} already exists. "
                "Skipping (no --force flag provided)."
            )
        else:
            with open(rst_filename, "w", encoding="utf-8") as rst_file:
                rst_file.write(f".. automodapi:: {library_name}.{component}\n")
            print(f"Created {rst_filename}")

    print("Documentation generation process completed.")


if __name__ == "__main__":
    main()

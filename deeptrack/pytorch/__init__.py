try:
    import torch
except ImportError:
    raise ImportError(
        "PyTorch is not installed. "\ 
        "Please install it consulting the documentation at https://pytorch.org/get-started/locally/ " \
        "to access this feature."
    )
try:
    # Check if the code is running in a Jupyter Notebook environment
    shell = get_ipython().__class__.__name__
    if shell == "ZMQInteractiveShell":
        from tqdm.notebook import tqdm as tqdm
    else:
        from tqdm import tqdm as tqdm
except NameError:
    # If the code is not running in a Jupyter Notebook environment
    from tqdm import tqdm as tqdm
"""Helper functions for reading and writing in and out of files."""
import os

def incriment_file(base_filename: str, directory: str) -> None:
    """
    Determine the next incrimented version of the file nmae.

    :param base_filename: look for files with this name, then incriment them with a suffix underscore and number
    :param directory: the relative filepath to look inside of 

    :returns: file name incrimented appropriately
    """
    # Extract the name and extension
    name, ext = os.path.splitext(base_filename)
    path = os.path.join(directory, base_filename)

    # Check if the file already exists
    counter = 0
    while os.path.exists(path):
        # Add or update the suffix
        path = os.path.join(directory, f"{name}_{counter}{ext}")
        counter += 1

    return path
import sys
import os


def tabulate_item(row, col_widths, sep="  "):
    """Convert a string list representing a row into a formatted string.

    Args:
        row (str list): A list of strings representing a row in a table, each 
            item represents a column within a row of the table.
        col_widths (int list): A list of integers representing the column 
            width of the table.
        sep (str, optional): The separation character between rows. Default is
            two spaces.

    Returns:
        str: A string representing a tabulated row.
    """
    str_reps = []
    row_wid = len(row)
    for i in range(0, len(row)):
        wid = col_widths[i]
        str_reps.append(row[i].ljust(wid))
    out = sep.join(str_reps)
    return out


def tabulate(rows, sep="  "):
    """Generate a formatted string representing a table.

    Args:
        rows (str list list): A list in which each element is a list of strings
            representing a row in the table.
        sep (str, optional): The separation character between rows. Default is
            two spaces.

    Returns:
        str: Formatted string representing a table.
    """
    cols = max(map(len, rows))
    col_widths = []
    for i in range(0, cols):
        w = 0
        for c in rows:
            if i < len(c):
                w = max(w, len(c[i]))
        col_widths.append(w)
    res = "\n".join(map(lambda x: tabulate_item(x, col_widths, sep), rows))
    return res


def open_read_file(path, extension):
    """Opens a file to read with some handling.
    
    Args:
        path (str): Path to the file.
        extension (str): Expected extension.
    
    Returns:
        file: The file opened and ready to be read.
    
    Raises:
        ValueError: Raised when the file does not exist or the extension does 
            not match the one expected.
    """
    if (path.split('.')[-1] != extension):
        raise ValueError('File %s is not of extension %s.' % (path, extension))
    if (not os.path.isfile(path)):
        raise ValueError('File %s does not exist.' % path)
    return open(path, 'r')


def open_write_file(path, overwrite_protect=True):
    """Opens a file to write to with some handling.
    
    Args:
        path (str): Path to the file.
        overwrite_protect (bool, optional): When set to True, will give a new 
            file name when the original designated file name has already 
            existed instead of overwriting it.
    
    Returns:
        file: The file opened and ready to be written to.
    """
    if os.path.isfile(path) and overwrite_protect:
        path_split = path.split('.')
        if len(path_split) > 1:
            extension = '.' + path_split[-1]
            path = '.'.join(path_split[0:-1])
        else:
            extension = ''
        counter = 1
        while os.path.isfile(path + '_' + str(counter) + extension):
            counter += 1
        path = path + '_' + str(counter)
        return open(path + extension, 'w')
    else:
        return open(path, 'w')
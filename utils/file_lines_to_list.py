"""
 Convert the lines of a file to a list
"""


def file_lines_to_list(path: str) -> str:
    # open txt file lines to a list
    with open(path) as f:
        content = f.readlines()
    # remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content]
    return content

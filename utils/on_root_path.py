import os


def on_root_path():
    """
    return root directory path as string
    :return: root directory path
    """
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

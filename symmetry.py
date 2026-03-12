import numpy as np


def get_symmetries(board):
    """
    Return 8 symmetries:
    4 rotations + mirrored versions
    """
    b = np.array(board, dtype=int)
    boards = []

    for k in range(4):
        rot = np.rot90(b, k)
        boards.append(rot)
        boards.append(np.fliplr(rot))

    return boards
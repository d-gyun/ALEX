import matplotlib.pyplot as plt
import numpy as np


def minmax(min_val, max_val, val):
    if max(min_val, val) == val and min(val, max_val) == val:
        return val
    elif max(min_val, val) != val and min(val, max_val) == val:
        return min_val
    else:
        return max_val


def hist_plot(err_board, title):
    occurrences = {}
    for i, count in enumerate(err_board):
        occurrences[i] = count
    checks = list(occurrences.keys())
    counts = list(occurrences.values())
    plt.bar(checks, counts, align='center', color='red', edgecolor='black')
    plt.xlabel('Number of errors')
    plt.ylabel('Occurrences')
    plt.title(title)
    plt.show()

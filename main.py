import numpy as np
import matplotlib.pyplot as plt
from LearnedIndex import RMI, LearnedIndexNode, Regression
from DataGen import DataGen, Distribution
from helpers import hist_plot


def main():
    data = DataGen(Distribution.RANDOM, 10000).generate()
    idx = RMI(Regression.LINEAR, data)  # Create a learned index for the samples created
    # res, err = idx.find(10)  # Find the position of the key using the index
    error_board = idx.find_all()
    error_board = error_board.astype(int)
    print(error_board)
    hist_plot(error_board, 'Linear Regression in Random Data N=1000')


if __name__ == '__main__':
    main()








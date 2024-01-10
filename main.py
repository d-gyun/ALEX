import numpy as np
import matplotlib.pyplot as plt
from ALEX import RMI as ALEX_RMI, LearnedIndexNode as ALEX_LearnedIndexNode, Regression as ALEX_Regression
from LearnedIndex import RMI as LI_RMI, LearnedIndexNode as LI_LearnedIndexNode, Regression as LI_Regression
from DataGen import DataGen, Distribution
from helpers import hist_plot


def main():
    data = DataGen(Distribution.RANDOM, 1000).generate()
    ALEX_idx = ALEX_RMI(ALEX_Regression.LINEAR, data)  # Create a learned index for the samples created
    LI_idx = LI_RMI(LI_Regression.LINEAR, data)
    # res, err = idx.find(10)  # Find the position of the key using the index
    ALEX_error_board = ALEX_idx.find_all()
    ALEX_error_board = ALEX_error_board.astype(int)
    LI_error_board = LI_idx.find_all()
    LI_error_board = LI_error_board.astype(int)
    print(ALEX_error_board)
    print(LI_error_board)
    hist_plot(ALEX_error_board, 'ALEX Linear Regression in Random Data N=1000')
    hist_plot(LI_error_board, 'LI Linear Regression in Random Data N=1000')


if __name__ == '__main__':
    main()








import numpy as np
import matplotlib.pyplot as plt
from ALEX import RMI as ALEX_RMI, LearnedIndexNode as ALEX_LearnedIndexNode
from InhancedALEX import RMI as InhancedALEX_RMI, LearnedIndexNode as InhancedALEX_LearnedIndexNode
from LearnedIndex import RMI as LI_RMI, LearnedIndexNode as LI_LearnedIndexNode
from DataGen import DataGen, Distribution
from helpers import hist_plot


def main():
    data = DataGen(Distribution.RANDOM, 1000).generate()
    LI_idx = LI_RMI(data)
    # ALEX_idx = ALEX_RMI(data)  # Create a learned index for the samples created
    # InhancedALEX_idx = InhancedALEX_RMI(data)
    # res, err = LI_idx.find(64895)  # Find the position of the key using the index

    LI_error_board = LI_idx.find_all()
    LI_error_board = LI_error_board.astype(int)

    # ALEX_error_board = ALEX_idx.find_all()
    # ALEX_error_board = ALEX_error_board.astype(int)
    #
    # InhancedALEX_error_board = InhancedALEX_idx.find_all()
    # InhancedALEX_error_board = InhancedALEX_error_board.astype(int)

    print(LI_error_board)
    LearnedIndex_cost = sum(err*(i+1) for i, err in enumerate(LI_error_board))

    # print(ALEX_error_board)
    # ALEX_cost = sum(err*(i+1) for i, err in enumerate(ALEX_error_board))
    #
    # print(InhancedALEX_error_board)
    # InhancedALEX_cost = sum(err*(i+1) for i, err in enumerate(InhancedALEX_error_board))

    print("LearnedIndex_cost:", LearnedIndex_cost)
    # print("ALEX_cost:", ALEX_cost)
    # print("InhancedALEX_cost:", InhancedALEX_cost)
    hist_plot(LI_error_board, 'LI Linear Regression in RANDOM Data N=100000')
    # hist_plot(ALEX_error_board, 'ALEX Linear Regression in RANDOM Data N=100000')
    # hist_plot(InhancedALEX_error_board, 'InhancedALEX Linear Regression in RANDOM Data N=100000')

    # insert_key = DataGen(Distribution.NORMAL, 1).generate()[0]
    # ALEX_idx.insert(50000)


if __name__ == '__main__':
    main()








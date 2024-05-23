import numpy as np
import matplotlib.pyplot as plt
from ALEX import RMI as ALEX_RMI
from EnhancedALEX import RMI as enhancedALEX_RMI
from DataGen import DataGen, Distribution
from helpers import hist_plot

def main():
    data = DataGen(Distribution.RANDOM, 1000, 100000).generate()
    insert_data = DataGen(Distribution.RANDOM, 1, 100000).generate()
    ALEX_idx = ALEX_RMI(data)
    enhancedALEX_idx = enhancedALEX_RMI(data)

    ALEX_error_board = ALEX_idx.find_all(data)
    ALEX_error_board = ALEX_error_board.astype(int)

    enhancedALEX_error_board = enhancedALEX_idx.find_all(data)
    enhancedALEX_error_board = enhancedALEX_error_board.astype(int)

    Before_ALEX_cost = sum(err*(i+1) for i, err in enumerate(ALEX_error_board))
    Before_ALEX_leafnode_cnt = ALEX_idx.leafNodeCnt()

    Before_enhancedALEX_cost = sum(err*(i+1) for i, err in enumerate(enhancedALEX_error_board))
    Before_enhancedALEX_leafnode_cnt = enhancedALEX_idx.leafNodeCnt()

    hist_plot(ALEX_error_board, 'ALEX Linear Regression in RANDOM Data')
    hist_plot(enhancedALEX_error_board, 'enhancedALEX Linear Regression in RANDOM Data')

    ALEX_idx.bulk_load(insert_data)
    enhancedALEX_idx.bulk_load(insert_data)

    ALEX_split_cnt = ALEX_idx.split_cnt
    enhancedALEX_split_cnt = enhancedALEX_idx.split_cnt

    ALEX_error_board = ALEX_idx.find_all(data)
    ALEX_error_board = ALEX_error_board.astype(int)
    enhancedALEX_error_board = enhancedALEX_idx.find_all(data)
    enhancedALEX_error_board = enhancedALEX_error_board.astype(int)

    After_ALEX_cost = sum(err * (i + 1) for i, err in enumerate(ALEX_error_board))
    After_ALEX_leafnode_cnt = ALEX_idx.leafNodeCnt()
    After_enhancedALEX_cost = sum(err * (i + 1) for i, err in enumerate(enhancedALEX_error_board))
    After_enhancedALEX_leafnode_cnt = enhancedALEX_idx.leafNodeCnt()

    print("Before ALEX_cost:", Before_ALEX_cost)
    print("Before enhancedALEX_cost:", Before_enhancedALEX_cost)
    print("Before ALEX_leafnode_cnt:", Before_ALEX_leafnode_cnt)
    print("Before enhancedALEX_leafnode_cnt:", Before_enhancedALEX_leafnode_cnt)

    print("After ALEX_cost:", After_ALEX_cost)
    print("After enhancedALEX_cost:", After_enhancedALEX_cost)
    print("After ALEX_leafnode_cnt:", After_ALEX_leafnode_cnt)
    print("After enhancedALEX_leafnode_cnt:", After_enhancedALEX_leafnode_cnt)

    print("ALEX_split_cnt:", ALEX_split_cnt)
    print("enhancedALEX_split_cnt:", enhancedALEX_split_cnt)

    hist_plot(ALEX_error_board, 'ALEX Linear Regression in RANDOM Data')
    hist_plot(enhancedALEX_error_board, 'enhancedALEX Linear Regression in RANDOM Data')

if __name__ == '__main__':
    main()

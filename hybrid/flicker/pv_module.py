import numpy as np

# pvmismatch standard module description
cell_len = 0.124
cell_rows = 12
cell_cols = 8
cell_num_map = [[11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
                [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
                [35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24],
                [36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47],
                [59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48],
                [60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71],
                [83, 82, 81, 80, 79, 78, 77, 76, 75, 74, 73, 72],
                [84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95]]
cell_num_map_flat = np.array(cell_num_map).flatten()
module_width = cell_len * cell_rows
module_height = cell_len * cell_cols

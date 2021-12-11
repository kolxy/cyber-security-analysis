import os

import numpy as np
import pandas as pd


file_name = input('Enter the name of a file containing a matrix to convert to a csv: ')

if os.path.exists(file_name):
    with open(file_name, 'r') as f:
        f_lines = f.read()
        f_lines = f_lines.split()
        f_lines = map(lambda x: x.lstrip('[').rstrip(']'), f_lines)        
        f_lines = list(filter(lambda x: x != '', f_lines))
        shape = int(np.sqrt(len(f_lines)))
        matrix = np.empty((len(f_lines), 3)).astype(int)

        for index in range(len(f_lines)):
            predicted = index % shape
            target = index // shape
            matrix[index, 0] = target 
            matrix[index, 1] = predicted
            matrix[index, 2] = f_lines[index]

        df = pd.DataFrame(matrix,
                          index=range(1, len(f_lines) + 1),
                          columns=['target', 'prediction', 'n'])

        print(df)
        
        df.to_csv(file_name[:-4] + '.csv', index=False)
        print(f'Operation completed successfully - output at: {file_name[:-4] + ".csv"}')
else:
    raise ValueError(f'Invalid file name: {file_name}')



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


def generateSinWave(save_path: str, length: int = 5000):
    # values for sin waves
    x = np.linspace(0, 2000, 6757)
    y1 = np.sin(x * 0.1 + 2)
    y2 = np.sin(x * 0.1 + 3)
    y3 = np.sin(x * 0.1 + 5)
    y4 = np.sin(x * 0.1 + 12)
    y5 = (np.random.rand(6757) * 40000000 + 100000000).astype(int)

    plt.plot(x, y1)
    plt.show()

    # create timestamps
    timestamps = pd.date_range(start='1/1/2020', periods=length, freq='H')

    # create a dataframe
    data = pd.DataFrame({'timestamp': timestamps, 'Close': y1,
                        'High': y2, 'Low': y3, 'Open': y4, 'Volume': y5})

    # save the dataframe to a CSV file
    data.to_csv('sin_wave.csv', index=False)


save_path = '../dataset/sin_wave.csv'
length = 5000
generateSinWave(save_path, length)

import numpy as np
import pandas as pd 
import sys

class Classifier:
    def __init__(self, data): 
        self.df = None
        self.train(data)

    def train(self, data):
        self.df = pd.read_csv(data, sep=r'\s+', header=None, engine="python")

    def test(self, instance):
        min = sys.float_info.max
        nearestNeighbor = -1
        npDf = self.df.to_numpy()

        temp = npDf[instance][1::]
        for index, row in enumerate(npDf):
            if (index != instance):
                distance = np.linalg.norm(row[1::] - temp)
                if distance < min:
                    min = distance
                    nearestNeighbor = row[0]

        return nearestNeighbor

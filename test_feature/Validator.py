from Classifier import Classifier 
import pandas as pd 
import time

class Validator:
    def __init__(self):
        pass

    def validate(self, features, data):
        initialTime = time.perf_counter()

        if features == []:
            return 0

        classifier = Classifier(data)

        temp = classifier.df[features].copy()

        normalized_df = (temp - temp.mean()) / temp.std()
        classifier.df = pd.concat([classifier.df.iloc[:, 0], normalized_df], axis=1)

        incorrect = 0
        npDf = classifier.df.to_numpy()

        startTime = time.perf_counter()
        for index, row in enumerate(npDf):
            if (classifier.test(index) != row[0]):
                incorrect += 1 
            print("Elapsed time:", time.perf_counter() - startTime)
            startTime = time.perf_counter()
        
        totalTime = time.perf_counter() - initialTime
        accuracy = (len(npDf) - incorrect) / len(npDf)
        return totalTime, accuracy

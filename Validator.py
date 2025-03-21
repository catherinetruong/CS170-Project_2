from Classifier import Classifier
import pandas as pd

class Validator:
    def validate(self, features, data):
        if features == []:
            return 0

        classifier = Classifier(data)

        # copy columns
        temp = classifier.df[features].copy()

        # normalize dataframe by - mean / standard deviation of each column
        normalized_df = (temp - temp.mean()) / temp.std()
        classifier.df = pd.concat([classifier.df.iloc[:, 0], normalized_df], axis=1)

        # find number of incorrect classifications
        incorrect = 0
        npDf = classifier.df.to_numpy()

        # leave one out algo
        for index, row in enumerate(npDf):
            if (classifier.test(index) != row[0]):
                incorrect += 1 
        accuracy = (len(npDf) - incorrect) / len(npDf)

        return accuracy

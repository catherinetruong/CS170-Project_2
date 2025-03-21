import time
import pandas as pd
from Validator import Validator

class Node:
    def __init__(self, features, data):
        self.features = features
        self.children = []
        self.accuracy = -1
        self.data = data
        self.df = pd.read_csv(data, sep=r'\s+', header=None, engine="python")
        self.possibleFeatures = self.df.shape[1] - 1
        self.trace_file = "trace.txt"
        self.features_selected = []
        self.accuracies = []
        
        self.setAccuracy()


    def log(self, message): # for trace
        print(message, flush=True)  # force immediate print -> fix print issue
        with open(self.trace_file, "a") as f:
            f.write(message + "\n")
            f.flush()  # force write to file  -> fix print issue


    def setAccuracy(self):
        valid = Validator()
        self.accuracy = valid.validate(self.features, self.data)


    # forward selection

    def forwardSelection(self):
        start_time = time.time()
        
        self.log(f"\nDataset has {self.possibleFeatures} features with {len(self.df)} instances.")
        self.log("\nBeginning forward selection...")

        best_accuracy, best_features = self.forwardSelectionHelper()
        self.log(f"The best feature set was {best_features} with accuracy {best_accuracy:.3f}")

        elapsed_time = time.time() - start_time
        self.log(f"Forward Selection completed in {elapsed_time:.3f} seconds.")


    def forwardSelectionHelper(self):
        self.addFeature()
        if not self.children:
            return self.accuracy, self.features

        bestAccuracy = self.accuracy
        bestChild = None

        for child in self.children:
            self.log(f"Using feature(s) {child.features}, accuracy is {child.accuracy:.3f}")
            self.features_selected.append(str(child.features))
            self.accuracies.append(child.accuracy * 100)

            if child.accuracy > bestAccuracy:
                bestAccuracy = child.accuracy
                bestChild = child

        if not bestChild:
            return self.accuracy, self.features

        self.log(f"Best feature set to expand is {bestChild.features} with accuracy {bestAccuracy:.3f}")

        tempAccuracy, tempFeatures = bestChild.forwardSelectionHelper()
        return (tempAccuracy, tempFeatures) if tempAccuracy > bestAccuracy else (bestAccuracy, bestChild.features)


    def addFeature(self):
        for i in range(1, self.possibleFeatures + 1):
            if i not in self.features:
                newNode = Node(self.features + [i], self.data)
                self.children.append(newNode)


    # backward elimination

    def backwardElimination(self):
        start_time = time.time()

        self.log(f"\nDataset has {self.possibleFeatures} features with {len(self.df)} instances.")
        self.log("\nBeginning backward elimination...")

        self.features = list(range(1, self.possibleFeatures + 1))
        best_accuracy, best_features = self.backwardEliminationHelper()

        self.log(f"The best feature set was {best_features} with accuracy {best_accuracy:.3f}")

        elapsed_time = time.time() - start_time
        self.log(f"Backward Elimination completed in {elapsed_time:.3f} seconds.")


    def backwardEliminationHelper(self):
        self.removeFeature()
        if not self.children:
            return self.accuracy, self.features

        bestAccuracy = self.accuracy
        bestChild = None

        for child in self.children:
            self.log(f"Using feature(s) {child.features}, accuracy is {child.accuracy:.3f}")

            if child.accuracy > bestAccuracy:
                bestAccuracy = child.accuracy
                bestChild = child

        if not bestChild:
            return self.accuracy, self.features

        self.log(f"Best feature set to expand is {bestChild.features} with accuracy {bestAccuracy:.3f}")

        tempAccuracy, tempFeatures = bestChild.backwardEliminationHelper()
        return (tempAccuracy, tempFeatures) if tempAccuracy > bestAccuracy else (bestAccuracy, bestChild.features)


    def removeFeature(self):
        for i in self.features:
            newFeatures = list(self.features)
            newFeatures.remove(i)
            newNode = Node(newFeatures, self.data)
            self.children.append(newNode)

''' couldn't get to work properly

    def plot_graph(self):
        if not self.features_selected or not self.accuracies:
            print("No data collected for graphing.")
            return

        plt.figure(figsize=(12, 6))
        plt.bar(self.features_selected, self.accuracies, color='skyblue')

        plt.xlabel("Feature Set", fontsize=12)
        plt.ylabel("Accuracy (%)", fontsize=12)
        plt.title("Feature Selection Accuracy", fontsize=14)
        plt.xticks(rotation=45, ha="right")
        plt.yticks(range(0, 110, 10))

        plt.savefig("plot.png", bbox_inches="tight")
        plt.show()
'''
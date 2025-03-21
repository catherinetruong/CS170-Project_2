from Validator import Validator

data = None

option = input("Welcome to Catherine Truong'smFeature Selection Problem!\nPlease choose a dataset:\n1. Small dataset (26)\n2. Large dataset (116)\n3. [TEST] Large dataset (12)\n4. [TEST] Large dataset (17)\nPlease enter a number:\n")
if option == "1":
    data = "../data/CS170_Small_Data__26.txt"
elif option == "2":
    data = "../data/CS170_Large_Data__116.txt"
elif option == "3":
    data = "../data/CS170_Large_Data__12.txt"
elif option == "4":
    data = "../data/CS170_Large_Data__17.txt"
else:
    print("Invalid input. Exiting...")
    quit()

features = []

n = input("Please enter the number of features to test: \n")
print("Please input the features one at a time:")
for i in range(int(n)):
    curr = input()
    features = features + [int(curr)]

print("Running k-fold cross validation...")

validator = Validator()
time, accuracy = validator.validate(features, data)

localPathSubstringLength = 22
print("\nThe data from file \"", data[len(data)-localPathSubstringLength:], "\" when tested with features:", features, "has an accuracy of", accuracy)
print("Total time elapsed:", time)

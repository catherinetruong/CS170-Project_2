from Node import Node

data = None
option = input("Welcome to Catherine Truong's Feature Selection Algorithm!\nPlease choose a dataset:\n1. Small dataset (26)\n2. Large dataset (116)\n3. [TEST] Large dataset (12)\nPlease enter a number:\n")
if option == "1":
    data = "../data/CS170_Small_Data__26.txt"
elif option == "2":
    data = "../data/CS170_Large_Data__116.txt"
elif option == "3":
    data = "../data/CS170_Large_Data__12.txt"
else:
    print("Invalid input. Exiting...")
    quit()
node = Node([], data)

option = input("Please choose a search method:\n1. Forward selection\n2. Backward elimination\nPlease enter a number:\n")
if option == "1":
    node.forwardSelection()
elif option == "2":
    node.backwardElimination()
else:
    print("Invalid input. Exiting...")
    quit()
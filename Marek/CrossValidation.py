import numpy as np

# -------------------------------------------
# Setup data for cross validation then test
#
# input:  X_input, Matrix of data
# input:  k, Number of folds
# output: None
# effect: Conducts k-fold testing
# ------------------------------------------
def crossValidation(X_input, k):

    # Check input values are okay
    if X_input is None:
        print("Invalid X")
        return

    if k <= 1:
        print("Invalid k")
        return

    # Delete data until X is divisble by k
    X = np.copy(X_input)
    while (len(X) / k).is_integer() == False:
        X = np.copy(X[0 : len(X) - 1])

    # Create arrays for test and validation data
    Y = np.split(X, k)
    TestData = np.zeros(shape=(int(len(X) / k) * (k - 1), len(X[0])))
    ValiData = np.zeros(shape=(int(len(X) / k), len(X[0])))

    # Fill test and validation arrays
    for i in range(0, len(Y)):
        count = 0
        for j in range(0, len(Y)):
            if i != j:
                for a in range(0, len(Y[j])):
                    for b in range(0, len(Y[j][a])):
                        TestData[count][b] = Y[j][a][b]
                    count = count + 1
            else:
                for a in range(0, len(Y[j])):
                    for b in range(0, len(Y[j][a])):
                        ValiData[a][b] = Y[j][a][b]
        dummyTest(TestData, ValiData)


# ----------------------------------------------
# Use for pretend test (actually do real tests)
# ----------------------------------------------
def dummyTest(TestData, ValiData):
    print("========================")
    print("Test Data---------------")
    print(TestData)
    print("Validation Data---------")
    print(ValiData)
    print("========================")
    print()


# Example
X = np.array([[1, 3], [2, 4], [3, 1], [2, 2], [2, 1], [4, 4], [7, 8], [6, 3]])
crossValidation(X, 4)

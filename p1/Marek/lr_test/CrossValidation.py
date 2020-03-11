import numpy as np
import LogisticRegression as lr
import ModelEvaluation as meval

# ------------------------------------------
# Compute cross validation on data.
#
# author: Marek Adamowicz
# since : Feb 9 2020
# ------------------------------------------
class LogRegCrossValidation:
    
    # ------------------------------------------
    # Constructor
    #
    # input: X_input, y_input = test data
    # input: k = number of folds
    # ------------------------------------------
    def __init__(self, X_input, y_input, k, lr, eps, iterations):         
        self.X_input = X_input
        self.y_input = y_input
        self.k = k
        self.valiError = [0] * k
        self.lr = lr
        self.eps = eps
        self.iterations = iterations
        
        
    # ------------------------------------------
    # Setup data for cross validation then test
    #
    # input:  X_input, Matrix of data
    # input:  k, Number of folds
    # output: None
    # effect: Conducts k-fold testing
    # ------------------------------------------
    def crossValidation(self):
        
        # Delete data until X is divisble by k
        trim_X = np.copy(self.X_input)
        trim_y = np.copy(self.y_input)
        while (len(trim_X) / self.k).is_integer() == False:
            trim_X = np.copy(trim_X[0:len(trim_X)-1])
            trim_y = np.copy(trim_y[0:len(trim_y)-1])

        # Create arrays for test and validation data
        partitions_X = np.split(trim_X,self.k)
        partitions_y = np.split(trim_y,self.k)
        testdata = np.zeros(shape=(int(len(trim_X)/self.k)*(self.k-1),len(trim_X[0])))
        validata = np.zeros(shape=(int(len(trim_X)/self.k),len(trim_X[0])))
        testtarget = [0] * int(len(trim_X)/self.k*(self.k-1))
        valitarget = [0] * int(len(trim_X)/self.k)
        
      #  print(len(testdata))
      #  print(len(validata))
        
        # Fill test and validation arrays
        for i in range(0, self.k):
        #    print(i)
            count = 0
            for j in range(0, self.k):
                if i != j:
                    for a in range(0,len(partitions_X[0])):
                        for b in range(0,len(partitions_X[0][0])):
                            testdata[count][b] = partitions_X[j][a][b]
                        testtarget[count] = partitions_y[j][a]
                        count = count + 1
                else:
                    for a in range(0,len(partitions_X[0])):
                        for b in range(0,len(partitions_X[0][0])):
                            validata[a][b] = partitions_X[j][a][b]
                        valitarget[a] = partitions_y[j][a]
            self.testPartition(testdata, testtarget, validata, valitarget, i)
                                            
                                            
    # ----------------------------------------------
    # Test a single k-fold iteration
    #
    # input: traindata, traintarget = training fold
    # input: validata, valitarget = validation fold
    # input: index of # of iterations
    # output: None
    # effect: Add a test value to valiError
    # ----------------------------------------------
    def testPartition(self, traindata, traintarget, validata, valitarget, index):
        model = lr.LogisticRegression()
        w = model.fit(traindata, traintarget, lr=self.lr, eps=self.eps, num_iter=self.iterations)
        yp = model.predict(validata, w)
        self.valiError[index] = meval.evaluate_acc(yp,valitarget)
        
    
    # ----------------------------------------------
    # Return the validation error
    #
    # input: None
    # output: valiError
    # ----------------------------------------------
    def getValiError(self):
        return self.valiError
    
    
    # ----------------------------------------------
    # Compute average error 
    #
    # input: None
    # output: Average testing error
    # ----------------------------------------------
    def averageError(self):
        return np.mean(self.valiError)
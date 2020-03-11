import CrossValidation as CV
import matplotlib.pyplot as plt

# ---------------------------------------------------
# Test logisitic regression through cross-validation
# 
# author: Marek Adamowicz
# since : Feb 10 2020
# ---------------------------------------------------
class LogRegTester:
    
    # -------------------------------------------
    # Constructor
    #
    # input: data, target = data to do tests on
    # input: k = # of folds in CV
    # -------------------------------------------
    def __init__(self, data, target, k):
        self.data = data
        self.target = target
        self.k = k
        
        
    # -------------------------------------------
    # Do a number of tests iteratively
    # 
    # input: None
    # output: None
    # effect: Save results in an array
    # --------------------------------------------
    def iterativeTester(self):

        # Example tests on iosphere data
        # ---------------------------------
        
        # lr = 20, eps = 0, tests = 50, iterstep = 5
        # -> Oscialates around best answer
        
        # lr = 1, eps = 0, tests = 50, iterstep = 5
        # -> Reaches answer quickly
        
        # lr = 0.01, eps = 0, tests = 50, iterstep = 5
        # -> Does not have time to reach best answer
        
        
        # Example tests on adult data
        # ---------------------------------
        
        # lr = 1, eps = 0, tests = 10, iterstep = 1000
        # -> Oscialates a lot
        
        # lr = 1, eps = 0, tests = 100, iterstep = 10
        # -> Still lots of oscilating
        
        # lr = 0.001, eps = 0, tests = 100, iterstep = 10
        # -> Still lots of oscilating
        
        
        # Example tests on glass data
        # -----------------------------------
        # lr = 0.02, eps = 0.01, tests = 10, iterstep = 5000
        # -> Good results (over 0.965)
        
        
        # Example tests on auto-mg data
        # -----------------------------------
        # lr = 0.02, eps = 0.01, tests = 10, iterstep = 5000
        # -> Good results (over 0.8 except first run)
        
        
        # Settings
        # --------------
        test_lr = 0.001
        test_eps = 0
        tests = 100000
        iterStep = 10
        self.results = [0] * tests
        
        # Test iteratively
        for i in range(0, tests):
            crossValidator = CV.LogRegCrossValidation(self.data, self.target,
                                    self.k, lr=test_lr, eps=test_eps, 
                                    iterations=(i+1) * iterStep)
            crossValidator.crossValidation()
            self.results[i] = crossValidator.averageError()
        # Plot results
        self.plotResults()
        
    
    # -------------------------------------------
    # Plot the results of the test
    #
    # input: None
    # output: None
    # effect: Plot results of tests
    # -------------------------------------------
    def plotResults(self):
        plt.figure(figsize=(10, 5))
        plt.subplot(1,1,1)
        plt.plot(self.results)
        plt.savefig("glass_samplesize.png")
    
    
            
            
            
            


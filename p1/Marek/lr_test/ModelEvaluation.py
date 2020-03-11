# --------------------------------------------------
# Evaluates the model accuracy. Inside it builds the 
# confusion matrix
#
# input:  true_y   = true labels
# input:  target_y = target labels
# output: Accuracy
# --------------------------------------------------
def evaluate_acc(true_y, target_y):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    
    # Obtain TP, TN, FP, FN
    for i in range(0, len(true_y)):
        if true_y[i] == 1:
            if true_y[i] == target_y[i]:
                TP = TP + 1
            else:
                FN = FN + 1
        else:
            if true_y[i] == target_y[i]:
                TN = TN + 1
            else:
                FP = FP + 1

    # Measure Accuracy
    P = TP + FN
    N = FP + TN
    accuracy = (TP + TN) / (P + N)
    return accuracy

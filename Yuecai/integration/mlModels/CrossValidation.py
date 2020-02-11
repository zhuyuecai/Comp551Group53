import numpy as np

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

class Cross_Validation:

    @staticmethod
    def partition(vector, fold, k):
        size = vector.shape[0]
        start = int(size/k)*fold
        end = int(size/k)*(fold+1)
        validation = vector[start:end]
        training = np.concatenate((vector[:start], vector[end:]))
        return training, validation

    @staticmethod
    def Cross_Validation(learner, k, examples, labels):
        train_folds_score = []
        validation_folds_score = []
        for fold in range(0, k):
            training_set, validation_set = Cross_Validation.partition(examples, fold, k)
            training_labels, validation_labels = Cross_Validation.partition(labels, fold, k)
            learner.fit(training_set, training_labels)
            training_predicted = learner.predict(training_set)
            validation_predicted = learner.predict(validation_set)
            train_folds_score.append(evaluate_acc(training_labels, training_predicted))
            validation_folds_score.append(evaluate_acc(validation_labels, validation_predicted))
        return train_folds_score, validation_folds_score

    @staticmethod
    def Size_Experiment(learner, size, examples, labels, iter=20):
        print("sample size %s"%(size))
        train_folds_score = []
        validation_folds_score = []
        data = np.hstack((examples,labels))
        counter = 0
        while counter < iter:
            try:
                np.random.shuffle(data)
                training_set = data[:size, :-1]
                training_labels = data[:size, -1:][:,0]
                validation_set = data[size:, :-1]
                validation_labels = data[size:, -1:][:,0]
                learner.fit(training_set, training_labels)
                training_predicted = learner.predict(training_set)
                validation_predicted = learner.predict(validation_set)
                train_folds_score.append(evaluate_acc(training_labels, training_predicted))
                validation_folds_score.append(evaluate_acc(validation_labels, validation_predicted))
                counter +=1
            except:
                pass
        return train_folds_score, validation_folds_score

            








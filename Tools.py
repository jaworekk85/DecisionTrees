import math
import numpy as np

class Tools:

    @staticmethod
    def transform_data_vector(data_true_vector, list_of_attributes, classes):
        data = []
        for i in range(len(data_true_vector) - 1):
            data.append(list_of_attributes[i].getIndex(data_true_vector[i]))
        data.append(classes.getIndex(data_true_vector[-1]))
        return data

    @staticmethod
    def two_best_measures(measures):
        max1 = measures[0]
        max2 = measures[1]
        id1 = 0
        id2 = 1
        if max2 > max1:
            max1 = measures[1]
            max2 = measures[0]
            id1 = 1
            id2 = 0
        for i in range(2, len(measures)):
            if measures[i] > max1:
                max2 = max1
                max1 = measures[i]
                id2 = id1
                id1 = i
            elif measures[i] > max2:
                max2 = measures[i]
                id2 = i
        return (id1, max1), (id2, max2)

    @staticmethod
    def entropy(fractions):
        fr = np.where(fractions > 0.0, fractions, 1.0)
        return (-fr * np.log(fr)/math.log(2.0)).sum()

    @staticmethod
    def Gini_index(fractions):
        return 1.0 - (fractions * fractions).sum()

    @staticmethod
    def misclassification_error(fractions):
        return 1.0 - fractions.max()

    @staticmethod
    def get_measure(fractions, measure):
        measures = {'entropy': Tools.entropy,
                    'gini': Tools.Gini_index,
                    'misclassification': Tools.misclassification_error}
        return measures[measure](fractions)

    @staticmethod
    def get_prequential_measure(original_list, lambdaa):
        res = []
        n = int(1.0 / (1.0 - lambdaa))
        avg = sum(original_list[:n]) / n
        for i in range(n):
            res.append(avg)
        for i in range(n, len(original_list)):
            n = n * lambdaa + 1
            res.append(res[i-1] * (n-1) / n + original_list[i] / n)
        return res

    @staticmethod
    def get_W_measure(original_list, W):
        res = []
        avg = sum(original_list[:W]) / W
        for i in range(W):
            res.append(avg)
        for i in range(W, len(original_list)):
            res.append(res[i-1] + original_list[i] / W - original_list[i-W] / W)
        return res






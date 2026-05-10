import numpy as np
from Attribute import Attribute
from Tools import Tools
import time


class Statistics:

    def __init__(self, *args):
        self.d = 0
        self.v = []
        self.k = 0
        self.n = 0
        self.stats = []
        self.attribute_types = []
        self.class_counts = np.zeros(0)
        self.numerical_threshold_values = []
        if isinstance(args[0], list) and isinstance(args[0][0], Attribute) and isinstance(args[1], Attribute):
            list_of_attributes = args[0]
            classes = args[1]
            self.d = len(list_of_attributes)
            self.k = classes.getN()
            self.class_counts = np.zeros(self.k)
            for i in range(self.d):
                self.attribute_types.append(list_of_attributes[i].type)
                self.v.append(list_of_attributes[i].getN())
                if list_of_attributes[i].type == 'numerical':
                    if list_of_attributes[i].range_a is not None and hasattr(list_of_attributes[i], 'dx'):
                        self.numerical_threshold_values.append(
                            [list_of_attributes[i].range_a + (j + 1) * list_of_attributes[i].dx
                             for j in range(list_of_attributes[i].getN() - 1)]
                        )
                    else:
                        self.numerical_threshold_values.append([j + 0.5 for j in range(list_of_attributes[i].getN() - 1)])
                else:
                    self.numerical_threshold_values.append([])
        if isinstance(args[0], Statistics):
            self.d = args[0].d
            self.k = args[0].k
            self.attribute_types = args[0].attribute_types[:]
            self.class_counts = np.zeros(self.k)
            self.numerical_threshold_values = [values[:] for values in args[0].numerical_threshold_values]
            for i in range(self.d):
                self.v.append(args[0].v[i])
        if len(self.v) > 0:
            for i in range(self.d):
                self.stats.append(np.zeros((self.v[i], self.k)))

    def has_numerical_attributes(self):
        return 'numerical' in self.attribute_types

    def rebuild_class_counts(self):
        if self.d == 0:
            self.class_counts = np.zeros(self.k)
        else:
            self.class_counts = self.stats[0].sum(axis=0)

    def update_statistics(self, data_vector):
        clas = data_vector[-1]
        self.class_counts[clas] = self.class_counts[clas] + 1
        for i in range(self.d):
            self.stats[i][data_vector[i]][clas] = self.stats[i][data_vector[i]][clas] + 1
        self.n = self.n + 1

    def get_class_frequencies(self):
        if self.n == 0:
            return np.zeros(self.k)
        else:
            return self.class_counts / self.n

    def get_fractions_for_one_attribute(self, attr_idx):
        if self.n == 0:
            return np.zeros(self.v[attr_idx])
        else:
            return self.stats[attr_idx].sum(axis=1) / self.n

    def get_class_frequencies_for_one_attribute(self, attr_idx):
        denoms = self.stats[attr_idx].sum(axis=1)
        denoms = np.where(denoms > 0, denoms, 1)
        return (self.stats[attr_idx].T / denoms).T

    def get_measure_for_one_attribute(self, attr_idx, measure):
        if self.attribute_types[attr_idx] == 'numerical':
            gain, threshold, threshold_bin, split_measure = self.get_best_numerical_split(attr_idx, measure)
            return split_measure
        fractions = self.get_fractions_for_one_attribute(attr_idx)
        class_frequencies = self.get_class_frequencies_for_one_attribute(attr_idx)
        measure_values = np.zeros(self.v[attr_idx])
        for i in range(self.v[attr_idx]):
            measure_values[i] = Tools.get_measure(class_frequencies[i], measure)
        return (fractions * measure_values).sum()

    def get_best_numerical_split(self, attr_idx, measure):
        if self.v[attr_idx] < 2:
            return 0.0, None, None, Tools.get_measure(self.get_class_frequencies(), measure)
        total_counts = self.stats[attr_idx].sum(axis=0)
        observed_n = total_counts.sum()
        if observed_n == 0:
            return 0.0, None, None, Tools.get_measure(self.get_class_frequencies(), measure)

        base = Tools.get_measure(total_counts / observed_n, measure)
        left_counts = np.zeros(self.k)
        right_counts = total_counts.copy()
        best_gain = 0.0
        best_threshold = None
        best_threshold_bin = None
        best_split_measure = base

        for i in range(self.v[attr_idx] - 1):
            left_counts = left_counts + self.stats[attr_idx][i]
            right_counts = right_counts - self.stats[attr_idx][i]
            left_n = left_counts.sum()
            right_n = right_counts.sum()
            if left_n == 0 or right_n == 0:
                continue
            left_measure = Tools.get_measure(left_counts / left_n, measure)
            right_measure = Tools.get_measure(right_counts / right_n, measure)
            split_measure = (left_n * left_measure + right_n * right_measure) / observed_n
            gain = base - split_measure
            if gain > best_gain:
                best_gain = gain
                best_threshold = self.numerical_threshold_values[attr_idx][i]
                best_threshold_bin = i
                best_split_measure = split_measure

        return best_gain, best_threshold, best_threshold_bin, best_split_measure

    def get_split_candidates(self, measure):
        clas_frequencies = self.get_class_frequencies()
        base = Tools.get_measure(clas_frequencies, measure)
        candidates = []
        for i in range(self.d):
            if self.attribute_types[i] == 'numerical':
                gain, threshold, threshold_bin, split_measure = self.get_best_numerical_split(i, measure)
                groups = None
                if threshold_bin is not None:
                    groups = [list(range(threshold_bin + 1)), list(range(threshold_bin + 1, self.v[i]))]
                candidates.append(dict(attr=i, measure=gain, threshold=threshold, threshold_bin=threshold_bin,
                                       arity=2, split_measure=split_measure, value_groups=groups))
            else:
                split_measure = self.get_measure_for_one_attribute(i, measure)
                candidates.append(dict(attr=i, measure=base - split_measure, threshold=None,
                                       threshold_bin=None, arity=self.v[i], split_measure=split_measure,
                                       value_groups=[[j] for j in range(self.v[i])]))
        return candidates

    def get_measures(self, measure):
        candidates = self.get_split_candidates(measure)
        return np.array([candidate['measure'] for candidate in candidates])

    def majority_class(self):
        return self.get_class_frequencies().argmax()

    def bayes_class(self, data_vector):
        bc = self.get_class_frequencies().copy()
        if bc.sum() == 0:
            return 0
        for atr in range(self.d):
            attr_value = data_vector[atr]
            cs = self.stats[atr][attr_value].copy()
            summ = cs.sum()
            if summ > 0:
                bc = bc * (cs / summ)
        return bc.argmax()

    def classify(self, data_vector, method):
        if method == 0:
            return self.majority_class()
        else:
            return self.bayes_class(data_vector)

    def printall(self, measure):
        ret = 'n: '+str(self.n)+'\n'
        for i in range(self.d):
            ret = ret + 'attr '+str(i)+' : '+str(self.stats[i])+'\n'
        ret = ret + 'class frequencies: \n'
        ret = ret + str(self.get_class_frequencies())+'\n'
        ret = ret + 'fractions for attributes: \n'
        for i in range(self.d):
            ret = ret + 'attr ' + str(i) + ' : ' + str(self.get_fractions_for_one_attribute(i)) + '\n'
        ret = ret + 'class frequencies for attributes: \n'
        for i in range(self.d):
            ret = ret + 'attr ' + str(i) + ' : ' + str(self.get_class_frequencies_for_one_attribute(i)) + '\n'
        ret = ret + 'measures for attributes: \n'
        for i in range(self.d):
            ret = ret + 'attr ' + str(i) + ' : ' + str(self.get_measure_for_one_attribute(i, measure)) + '\n'
        ms = self.get_measures(measure)
        for i in range(self.d):
            ret = ret + 'measure for attr ' + str(i) + ' : ' + str(self.get_measures(measure)) + '\n'
        #time.sleep(120)
        return ret





class Extended_Statistics:

    def __init__(self, statistics):
        self.v = statistics.v[:]
        self.d = statistics.d
        self.k = statistics.k
        self.n = 0
        self.single_stats = []
        for i in range(self.d):
            self.single_stats.append(np.zeros((self.v[i], self.k)))
        self.stats = []
        for i1 in range(self.d - 1):
            self.stats.append([])
            for j1 in range(self.v[i1]):
                self.stats[i1].append([])
                for i2 in range(i1 + 1, self.d):
                    self.stats[i1][j1].append([])
                    for j2 in range(self.v[i2]):
                        self.stats[i1][j1][i2-i1-1].append([])
                        for kk in range(self.k):
                            self.stats[i1][j1][i2-i1-1][j2].append(0)

    def update_statistics(self, data_vector):
        clas = data_vector[-1]
        for i in range(self.d):
            self.single_stats[i][data_vector[i]][clas] = self.single_stats[i][data_vector[i]][clas] + 1
        for i1 in range(self.d - 1):
            for i2 in range(i1 + 1, self.d):
                self.stats[i1][data_vector[i1]][i2-i1-1][data_vector[i2]][clas] = self.stats[i1][data_vector[i1]][i2-i1-1][data_vector[i2]][clas] + 1
        self.n = self.n + 1

    def extract_statistics(self, ii, values):
        if isinstance(values, int):
            group_values = [values]
        else:
            group_values = list(values)
        sts = []
        for i in range(self.d):
            sts.append(np.zeros((self.v[i], self.k)))
        for i in range(ii):
            for j in range(self.v[i]):
                for k in range(self.k):
                    for value in group_values:
                        sts[i][j][k] = sts[i][j][k] + self.stats[i][j][ii-i-1][value][k]
        for i in range(ii + 1, self.d):
            for j in range(self.v[i]):
                for k in range(self.k):
                    for value in group_values:
                        sts[i][j][k] = sts[i][j][k] + self.stats[ii][value][i-ii-1][j][k]
        sum_n = 0
        for value in group_values:
            for k in range(self.k):
                sts[ii][value][k] = self.single_stats[ii][value][k]
                sum_n = sum_n + sts[ii][value][k]
        return sts, sum_n


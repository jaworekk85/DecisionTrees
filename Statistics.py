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
        if isinstance(args[0], list) and isinstance(args[0][0], Attribute) and isinstance(args[1], Attribute):
            list_of_attributes = args[0]
            classes = args[1]
            self.d = len(list_of_attributes)
            self.k = classes.getN()
            for i in range(self.d):
                self.v.append(list_of_attributes[i].getN())
        if isinstance(args[0], Statistics):
            self.d = args[0].d
            self.k = args[0].k
            for i in range(self.d):
                self.v.append(args[0].v[i])
        if len(self.v) > 0:
            for i in range(self.d):
                self.stats.append(np.zeros((self.v[i], self.k)))

    def update_statistics(self, data_vector):
        clas = data_vector[-1]
        for i in range(self.d):
            self.stats[i][data_vector[i]][clas] = self.stats[i][data_vector[i]][clas] + 1
        self.n = self.n + 1

    def get_class_frequencies(self):
        if self.n == 0:
            return np.zeros(self.k)
        else:
            return self.stats[0].sum(axis=0) / self.n

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
        fractions = self.get_fractions_for_one_attribute(attr_idx)
        class_frequencies = self.get_class_frequencies_for_one_attribute(attr_idx)
        measure_values = np.zeros(self.v[attr_idx])
        for i in range(self.v[attr_idx]):
            measure_values[i] = Tools.get_measure(class_frequencies[i], measure)
        return (fractions * measure_values).sum()

    def get_measures(self, measure):
        clas_frequencies = self.get_class_frequencies()
        base = Tools.get_measure(clas_frequencies, measure)
        measure_values = np.zeros(self.d)
        for i in range(self.d):
            measure_values[i] = self.get_measure_for_one_attribute(i, measure)
        return -1 * measure_values + base

    def majority_class(self):
        return self.get_class_frequencies().argmax()

    def bayes_class(self, data_vector):
        cs = np.zeros((self.d, self.k))
        for atr in range(self.d):
            cs[atr] = self.stats[atr][data_vector[atr]]
            summ = cs[atr].sum()
            if summ > 0:
                cs[atr] = cs[atr] / cs[atr].sum()
        bc = cs.prod(axis=0) * self.get_class_frequencies()
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
        for i1 in range(self.d - 1):
            for i2 in range(i1 + 1, self.d):
                self.stats[i1][data_vector[i1]][i2-i1-1][data_vector[i2]][clas] = self.stats[i1][data_vector[i1]][i2-i1-1][data_vector[i2]][clas] + 1
        self.n = self.n + 1

    def extract_statistics(self, ii, jj):
        sts = []
        for i in range(self.d):
            sts.append(np.zeros((self.v[i], self.k)))
        for i in range(ii):
            for j in range(self.v[i]):
                for k in range(self.k):
                    sts[i][j][k] = self.stats[i][j][ii-i-1][jj][k]
        for i in range(ii + 1, self.d):
            for j in range(self.v[i]):
                for k in range(self.k):
                    sts[i][j][k] = self.stats[ii][jj][i-ii-1][j][k]
        idx = 0
        if ii == 0:
            idx = 1
        sum_n = 0
        for k in range(self.k):
            sum_k = 0
            for j in range(self.v[idx]):
                sum_k = sum_k + sts[idx][j][k]
            sts[ii][jj][k] = sum_k
            sum_n = sum_n + sum_k
        return sts, sum_n


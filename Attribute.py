import math

class Attribute:

    def __init__(self, typee, list_of_values=None, range_a=None, range_b=None, nbins=None):
        self.type = typee
        self.list_of_values = list_of_values
        self.range_a = range_a
        self.range_b = range_b
        self.nbins = nbins
        if range_a is not None and range_b is not None and nbins is not None:
            self.dx = (range_b - range_a) / nbins
            self.split_value = (range_b - range_a) / 2.0

    def getN(self):
        if self.type == 'nominal':
            return len(self.list_of_values)
        else:
            return self.nbins

    def getIndexNominal(self, value):
        for i in range(len(self.list_of_values)):
            if self.list_of_values[i] == value:
                return i
        return -1

    def getIndexNumerical(self, value):
        for i in range(self.nbins):
            if (self.range_a + i * self.dx) <= value < (self.range_a + i * self.dx + self.dx):
                return i
        if math.isclose(value, self.range_b):
            return self.nbins - 1
        return -1

    def getIndex(self, value):
        if self.type == 'nominal':
            return self.getIndexNominal(value)
        if self.type == 'numerical':
            return self.getIndexNumerical(value)
        return -1
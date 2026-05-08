import random
from Attribute import Attribute
import numpy as np


class Node_gen:
    def __init__(self, level):
        self.kids = []
        self.splitting_attribute = None
        self.classa = None
        self.level = level

    def split(self, attr):
        self.splitting_attribute = attr
        self.kids.append(Node_gen(self.level + 1))
        self.kids.append(Node_gen(self.level + 1))

    def split_from_scratch(self, list_of_attribute_indexes, min_depth, max_depth, split_probability):
        r = random.random()
        if len(list_of_attribute_indexes) > 0 and (self.level <= min_depth or (r > split_probability and self.level <= max_depth)):
            atr = random.choice(list_of_attribute_indexes)
            new_list1 = []
            new_list2 = []
            for i in list_of_attribute_indexes:
                if i != atr:
                    new_list1.append(i)
                    new_list2.append(i)
            self.split(atr)
            self.kids[0].split_from_scratch(new_list1, min_depth, max_depth, split_probability)
            self.kids[1].split_from_scratch(new_list2, min_depth, max_depth, split_probability)
        else:
            self.classa = random.choice(list([0, 1]))

    def split_from_list(self, split_list):
        if len(split_list) > 0:
            el = split_list.pop(0)
            if len(el) == 1:
                self.split(int(el[0]))
                self.kids[0].split_from_list(split_list)
                self.kids[1].split_from_list(split_list)
            else:
                self.classa = int(el[0])

    def pass_data(self, vector):
        if self.splitting_attribute is not None:
            return self.kids[vector[self.splitting_attribute]].pass_data(vector)
        else:
            return self.classa

    def tree_code(self):
        if self.splitting_attribute is not None:
            return '' + str(self.splitting_attribute) + ',' + str(self.kids[0].tree_code()) \
                + ',' + str(self.kids[1].tree_code())
        else:
            return '' + str(self.classa) + 'K'





class Generator:

    def __init__(self):
        self.root = Node_gen(0)
        self.N = 0

    def create(self, Nofa, min_depth, max_depth, split_prob):
        self.N = Nofa
        list_of_a = [i for i in range(Nofa)]
        self.root.split_from_scratch(list_of_a, min_depth, max_depth, split_prob)

    def load_tree(self, filename):
        lofa = []
        with open(filename, 'r') as f:
            temp = f.read()
            lofa = temp.split(',')
        self.N = int(lofa.pop(0))
        self.root.split_from_list(lofa)

    def save_tree(self, filename):
        code = self.root.tree_code()
        with open(filename, 'w') as f:
            f.write('' + str(self.N) + ',' + code)

    def generate(self):
        data = np.random.choice(2, self.N)
        cls = self.root.pass_data(data)
        data = np.append(data, cls)
        return data

    def get_list_of_attributes(self):
        lofa = []
        for i in range(self.N):
            lofa.append(Attribute('nominal', [0, 1]))
        return lofa, Attribute('nominal', [0, 1])










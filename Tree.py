from Node import Node
from Criterion import Criterion
from Statistics import Statistics, Extended_Statistics
from Tools import Tools
from time import time
from types import FunctionType, ModuleType
import numpy as np
import os
import pickle
import sys

RESULTS_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
MAIN_FOLDER = RESULTS_FOLDER + os.sep
TREE_FOLDER = os.path.join(RESULTS_FOLDER, 'trees')
for folder in [RESULTS_FOLDER, TREE_FOLDER]:
    os.makedirs(folder, exist_ok=True)


class Tree:

    def __init__(self, name, statistics, criteria_list, ex_stats, preq_lambda, W):

        self.root = Node(statistics, 0, None, ex_stats)
        self.criteria_list = criteria_list
        self.lambdaa = preq_lambda
        self.W = W
        self.name = name
        if ex_stats:
            self.name = self.name + '_ES'

        # table of results
        self.holdout_n = []
        self.holdout_m = []
        self.holdout_b = []
        self.sequence_m = []
        self.sequence_b = []
        self.time = [0.0]
        self.leaves = [1]
        self.memory_n = []
        self.memory_bytes = []
        self.memory_mb = []

    def memory_size(self):
        return deep_getsizeof(self.root)

    def measure_memory(self, nn):
        memory_bytes = self.memory_size()
        self.memory_n.append(nn)
        self.memory_bytes.append(memory_bytes)
        self.memory_mb.append(memory_bytes / (1024.0 * 1024.0))

    def pass_data(self, data_vector):
        start = time()
        results, ignored_sa = self.root.pass_data(data_vector, self.criteria_list, [0, 1], False)
        ignored_results, sa = self.root.pass_data(data_vector, self.criteria_list, [0, 1], True)
        dtime = time() - start
        self.sequence_m.append(int(results[0] == data_vector[-1]))
        self.sequence_b.append(int(results[1] == data_vector[-1]))
        self.time.append(self.time[-1] + dtime)
        if sa > -1:
            self.leaves.append(self.leaves[-1] + 1)
        else:
            self.leaves.append(self.leaves[-1])

    def holdout_test(self, holdout_set, nn):
        res = [0, 0]
        for data_vector in holdout_set:
            rr, sa = self.root.pass_data(data_vector, self.criteria_list, [0, 1], False)
            res[0] = res[0] + int(rr[0] == data_vector[-1])
            res[1] = res[1] + int(rr[1] == data_vector[-1])
        res[0] = res[0] / len(holdout_set)
        res[1] = res[1] / len(holdout_set)
        self.holdout_n.append(nn)
        self.holdout_m.append(res[0])
        self.holdout_b.append(res[1])

    def toFile(self):
        dictionary = {
            'time': self.time,
            'leaves': self.leaves,
            'holdout_n': self.holdout_n,
            'holdout_m': self.holdout_m,
            'holdout_b': self.holdout_b,
            'prequential_m': Tools.get_prequential_measure(self.sequence_m, self.lambdaa),
            'prequential_b': Tools.get_prequential_measure(self.sequence_b, self.lambdaa),
            'W_m': Tools.get_W_measure(self.sequence_m, self.W),
            'W_b': Tools.get_W_measure(self.sequence_b, self.W),
            'sequence_m': self.sequence_m,
            'sequence_b': self.sequence_b,
            'memory_n': self.memory_n,
            'memory_bytes': self.memory_bytes,
            'memory_mb': self.memory_mb
        }
        with open(MAIN_FOLDER + self.name + '.pkl', 'wb') as f:
            pickle.dump(dictionary, f)

    def save_tree(self, filename):
        if not os.path.isabs(filename):
            filename = os.path.join(TREE_FOLDER, filename)
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        code = self.root.tree_code()
        with open(filename, 'w') as f:
            f.write('' + code)


def deep_getsizeof(obj, seen=None):
    if seen is None:
        seen = set()

    obj_id = id(obj)
    if obj_id in seen:
        return 0
    seen.add(obj_id)

    if isinstance(obj, (ModuleType, FunctionType, type)):
        return 0
    if isinstance(obj, np.ndarray):
        return sys.getsizeof(obj) + obj.nbytes

    size = sys.getsizeof(obj)

    if isinstance(obj, dict):
        for key, value in obj.items():
            size = size + deep_getsizeof(key, seen)
            size = size + deep_getsizeof(value, seen)
    elif isinstance(obj, (list, tuple, set, frozenset)):
        for item in obj:
            size = size + deep_getsizeof(item, seen)
    elif hasattr(obj, '__dict__'):
        size = size + deep_getsizeof(vars(obj), seen)

    return size

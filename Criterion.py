import math
from scipy.stats import norm


class Criterion:

    def __init__(self, K, delta, variance_type, measure, C=1.0, bias=False, tau=None):
        self.variance_type = variance_type
        self.measure = measure
        self.constant = 0.0
        self.tau = tau
        self.C = C
        ld = math.log(1.0 / delta, math.e)
        criterion_types = dict(hoeffding={'entropy': math.log(K, 2.0) * math.sqrt(ld / 2.0),
                                          'gini': (K - 1) / K * math.sqrt(ld / 2.0),
                                          'misclassification': (K - 1) / K * math.sqrt(ld / 2.0)
                                          },
                               mcdiarmid={'entropy': 0.0,
                                          'gini': math.sqrt(8.0 * ld),
                                          'misclassification': math.sqrt(2.0 * ld)
                                          },
                               gaussian={'entropy': 0.0,
                                         'gini': 0.0,
                                         'misclassification': norm.ppf(
                                             1 - delta) * math.sqrt(1 / 2.0)
                                         })

        self.constant = C * criterion_types[self.variance_type][self.measure]

        bias_const = {'entropy': 0.0, 'gini': 1.0 / math.sqrt(2.0), 'misclassification': 0.0}

        self.bias = bias

        if bias:
            self.constant = self.constant + bias_const[measure]

    def bound(self, n):
        return self.constant / math.sqrt(n)

    def check_split(self, difference, n):
        if n > 100 and (difference > 1.e-8) and (difference > self.bound(n) or (self.tau is not None and self.bound(n) < self.tau)):
            return True
        return False

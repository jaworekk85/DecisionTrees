from Statistics import Statistics, Extended_Statistics
from Tools import Tools
from Criterion import Criterion
import random

class Node:

    def __init__(self, statistics, level, parent=None, extended_statistics=False):
        self.parent = parent
        self.level = level
        self.kids = []
        self.splitting_attribute = None
        self.statistics = Statistics(statistics)
        self.ests = extended_statistics
        if self.ests:
            self.extended_statistics = Extended_Statistics(statistics)

    def check_split(self, criteria_list):
        for criterion in criteria_list:
            ms = self.statistics.get_measures(criterion.measure)
            (id1, max1), (id2, max2) = Tools.two_best_measures(ms)
            if criterion.check_split(max1 - max2, self.statistics.n):
                # print('uwaga dzielimy!')
                # print('attr: ', id1)
                # print('node level: ', self.level)
                # print('id1: ', id1, ', id2: ', id2, ', max1: ', max1, ', max2: ', max2)
                # self.statistics.printall(criterion.measure)
                # BEGINNING OF TEST
                self.max1 = str(max1)
                self.max2 = str(max2)
                self.id2 = str(id2)
                self.nstr = str(self.statistics.n)
                self.boundstr = str(criterion.bound(self.statistics.n))
                # END OF TEST
                return id1
            else:
                continue
        return -1

    def split(self, attr):
        self.splitting_attribute = attr
        for j in range(self.statistics.v[attr]):
            self.kids.append(Node(Statistics(self.statistics), self.level + 1, self, self.ests))
            ###### TEST TEST TEST  part 1 ##########################
            #if not self.ests:
            #    nm = ''.join(random.choice('abcdefghijklmnopqrstuvwxyz') for gg in range(10))
            #    with open('stats_normal_lev'+str(self.level)+'_'+nm+'.txt', 'w') as fnm:
            #        fnm.write('parent statistics: \n')
            #        fnm.write(self.statistics.printall('gini'))
            #        fnm.write('\n')
            #        fnm.write('kid nr ' + str(j) + ' statistics: \n')
            #        fnm.write(self.kids[j].statistics.printall('gini'))
            ####### End of TEST TEST TEST part 1 ######################
            if self.ests:
                self.kids[j].statistics.stats, self.kids[j].statistics.n = self.extended_statistics.extract_statistics(attr, j)
            ####### TEST TES TEST  part 2 ###############################
            #    nm = ''.join(random.choice('abcdefghijklmnopqrstuvwxyz') for gg in range(10))
            #    with open('stats_extended_lev'+str(self.level)+'_'+nm+'.txt', 'w') as fnm:
            #        fnm.write('parent statistics: \n')
            #        fnm.write(self.statistics.printall('gini'))
            #        fnm.write('\n')
            #        fnm.write('kid nr ' + str(j) + ' statistics: \n')
            #        fnm.write(self.kids[j].statistics.printall('gini'))
            ###### End of TEST TEST TEST  part 2 ######################

    def classify(self, data_vector, classification_method):
        return self.statistics.classify(data_vector, classification_method)

    def pass_data(self, data_vector, criteria_list, class_method, training=True):
        if self.splitting_attribute is not None:
            return self.kids[data_vector[self.splitting_attribute]].pass_data(data_vector, criteria_list, class_method,
                                                                              training)
        else:
            split_attr = -1
            if training:
                self.statistics.update_statistics(data_vector)
                if self.ests:
                    self.extended_statistics.update_statistics(data_vector)
                split_attr = self.check_split(criteria_list)
                if split_attr > -1:
                    self.split(split_attr)
            res = []
            for cm in class_method:
                res.append(self.classify(data_vector, cm))
            return res, split_attr

    def tree_code(self):
        if self.splitting_attribute is not None:
            return '' + str(self.splitting_attribute) + ',' + self.tree_code_parameters() + '\n' + self.kids[0].tree_code() + ',' + self.kids[1].tree_code()
        else:
            return '' + 'L'

    # TEST
    def tree_code_parameters(self):
        return 'level: ' + str(self.level) + ', max1: ' + self.max1 + ', id2: ' + self.id2 + ', max2: ' + self.max2 + ', n: ' + self.nstr + ', bound: ' + self.boundstr

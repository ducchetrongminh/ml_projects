# Standard imports
# import itertools
# import unittest
from copy import deepcopy
# from timeit import timeit
# from collections import Counter

# Third-party imports
# from pandas import DataFrame as df
# from pandas import read_csv



# https://www-users.cs.umn.edu/~kumar001/dmbook/ch6.pdf
class AprioriAlgorithm(object):
    def __init__(self, *, dataset = None, minsup = None, minconf = None):
        # Attributes
        self.all_frequent_itemsets = None
        self.all_rules = None
        self.dataset = None
        self.minconf = None
        self.minsup = None
        self.n = None # number of transaction

        # Call methods
        self.set_dataset(dataset) if dataset else None
        self.set_minsup(minsup)
        self.set_minconf(minconf)


    def set_dataset(self, dataset):
        self.dataset = dataset
        self.n = len(self.dataset)
        return self
    
    
    def set_minconf(self, minconf):
        self.minconf = minconf
        return self


    def set_minsup(self, minsup):
        self.minsup = minsup
        return self
        

    def set_transaction_dataset(self, *, transaction_dataset, transaction_id_column, itemset_column):
        transaction_dataset_cloned = transaction_dataset.copy(deep = True)
        transaction_dataset_cloned = transaction_dataset_cloned[{
            transaction_id_column,
            itemset_column
        }] \
        .dropna() \
        .drop_duplicates() \
        .sort_values([transaction_id_column, itemset_column]) \
        .set_index([transaction_id_column])

        self.dataset = []
        for transaction_id in transaction_dataset_cloned.index.unique():
            itemsets = transaction_dataset_cloned.loc[[transaction_id], itemset_column].tolist()
            self.dataset.append(tuple(itemsets))
        self.n = len(self.dataset)
        return self


    def generate_all_frequent_itemsets(self):
        self.all_frequent_itemsets = dict()

        frequent_single_itemsets = previous_frequent_itemsets = self._generate_frequent_single_itemsets()
        self.all_frequent_itemsets.update(frequent_single_itemsets)

        while True:
            previous_itemsets = list(previous_frequent_itemsets.keys())
            candidate_itemsets = self._generate_candidate_itemsets(previous_itemsets)
            frequent_itemsets = self._prune_frequent_itemsets(candidate_itemsets)

            if frequent_itemsets:
                self.all_frequent_itemsets.update(frequent_itemsets)
                previous_frequent_itemsets = frequent_itemsets
            else:
                break
        return self


    def generate_all_rules(self):
        self.all_rules = dict()

        if not self.all_frequent_itemsets:
            self.generate_all_frequent_itemsets()

        for frequent_itemset, support in self.all_frequent_itemsets.items():
            if len(frequent_itemset) <= 1:
                continue
            
            self._generate_rules(itemset=frequent_itemset)
        return self
    
    
    def _calculate_support(self, itemset):
        itemset_cloned = frozenset(itemset)
        support = 0.0
        for event in self.dataset:
            if itemset_cloned.issubset(frozenset(event)):
                support += 1
        return support/self.n

    
    @staticmethod
    def _generate_candidate_itemsets(itemsets):
        candidate_itemsets = list()
        for i, itemset in enumerate(itemsets[:-1]):
            for itemset2 in itemsets[i+1:]:
                itemset_cloned = list(deepcopy(itemset))
                itemset_cloned.sort()
                itemset2_cloned = list(deepcopy(itemset2))
                itemset2_cloned.sort()
                if itemset_cloned[:-1] == itemset2_cloned[:-1]:
                    candidate_itemset = set().union(itemset_cloned, itemset2_cloned)
                    candidate_itemsets.append(frozenset(candidate_itemset))
        return candidate_itemsets


    def _generate_frequent_single_itemsets(self):
        single_itemsets = set()
        for event in self.dataset:
            for item in event:
                single_itemsets.add(frozenset([item]))
        frequent_single_itemsets = self._prune_frequent_itemsets(single_itemsets)
        return frequent_single_itemsets


    def _generate_rules(self, itemset, previous_consequents=None):
        if previous_consequents:
            consequents = self._generate_candidate_itemsets(previous_consequents)
        else:
            consequents = list(deepcopy(itemset))

        if consequents and len(itemset) == len(consequents[0]):
            return

        for consequent in deepcopy(consequents):
            antecedent =  itemset.difference(frozenset([consequent]))
            confident = self.all_frequent_itemsets[frozenset(itemset)] / self.all_frequent_itemsets[frozenset(antecedent)]
            if confident >= self.minconf:
                lift = confident / self.all_frequent_itemsets[frozenset([consequent])]
                self.all_rules[(frozenset(antecedent), frozenset([consequent]))] = {
                    'support': self.all_frequent_itemsets[frozenset(itemset)],
                    'confident': round(confident, 7),
                    'lift': round(lift, 7)
                }
            else:
                consequents.remove(consequent)
        
        if consequents and len(itemset) > len(consequents[0]) + 1:
            self._generate_rules(itemset, previous_consequents = consequents)
        

    def _prune_frequent_itemsets(self, itemsets):
        frequent_itemsets = dict()
        for itemset in itemsets:
            itemset_cloned = frozenset(itemset)
            support = self._calculate_support(itemset_cloned)
            if support >= self.minsup:
                frequent_itemsets[itemset_cloned] = round(support, 7)
        return frequent_itemsets
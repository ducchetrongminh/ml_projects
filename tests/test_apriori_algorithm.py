# Standard library
# import itertools
import unittest
from copy import deepcopy
# from timeit import timeit
from collections import Counter

# Third-party library
# from pandas import DataFrame as df
# from pandas import read_csv



class TestAprioriAlgorithm(unittest.TestCase):
    maxDiff = None
    dataset =  [
        ["a"], ["a", "b", "c"], ["a", "c"], ["c"],
        ["a"], ["c"], ["b", "c"],
        ["a", "b"], ["d"], ["c"], ["b"], ["c"],
        ["a"], ["c"], ["b"], ["c"]
    ]

    apriori_algorithm_case_1 = AprioriAlgorithm(dataset = dataset, minsup = 2/16, minconf = 0.1)
    apriori_algorithm_case_2 = AprioriAlgorithm(dataset = dataset, minsup = 3/16, minconf = 0.2)
    apriori_algorithm_case_3 = AprioriAlgorithm(dataset = dataset, minsup = 6/16, minconf = 0.3)


    def test__calculate_support(self):
        self.assertEqual(self.apriori_algorithm_case_1._calculate_support(['a']), 6/16)
        self.assertEqual(self.apriori_algorithm_case_1._calculate_support(['b']), 5/16)
        self.assertEqual(self.apriori_algorithm_case_1._calculate_support(['c']), 9/16)
        self.assertEqual(self.apriori_algorithm_case_1._calculate_support(['d']), 1/16)
        self.assertEqual(self.apriori_algorithm_case_1._calculate_support(['a','b']), 2/16)
        self.assertEqual(self.apriori_algorithm_case_1._calculate_support(['a','c']), 2/16)
        self.assertEqual(self.apriori_algorithm_case_1._calculate_support(['a','b','c']), 1/16)
        self.assertEqual(self.apriori_algorithm_case_1._calculate_support(['a','b','c','d']), 0/16)


    def test__prune_frequent_itemsets(self):
        self.assertEqual(
            self.apriori_algorithm_case_1._prune_frequent_itemsets([['a'],['b'],['c'],['d']]), 
            {
                frozenset({'a'}): 0.375,
                frozenset({'b'}): 0.3125,
                frozenset({'c'}): 0.5625
            }
        )
        self.assertEqual(
            self.apriori_algorithm_case_2._prune_frequent_itemsets([['a'],['b'],['c'],['d']]), 
            {
                frozenset({'a'}): 0.375,
                frozenset({'b'}): 0.3125,
                frozenset({'c'}): 0.5625
            }
        )
        self.assertEqual(
            self.apriori_algorithm_case_3._prune_frequent_itemsets([['a'],['b'],['c'],['d']]), 
            {
                frozenset({'a'}): 0.375,
                frozenset({'c'}): 0.5625
            }
        )       


    def test__generate_frequent_single_itemsets(self):
        self.assertEqual(
            self.apriori_algorithm_case_1._generate_frequent_single_itemsets(), 
            {
                frozenset({'a'}): 0.375,
                frozenset({'b'}): 0.3125,
                frozenset({'c'}): 0.5625
            }
        )
        self.assertEqual(
            self.apriori_algorithm_case_2._generate_frequent_single_itemsets(), 
            {
                frozenset({'a'}): 0.375,
                frozenset({'b'}): 0.3125,
                frozenset({'c'}): 0.5625
            }
        )
        self.assertEqual(
            self.apriori_algorithm_case_3._generate_frequent_single_itemsets(), 
            {
                frozenset({'a'}): 0.375,
                frozenset({'c'}): 0.5625
            }
        )


    def test__generate_candidate_itemsets(self):
        candidate_itemsets = self.apriori_algorithm_case_1._generate_candidate_itemsets([frozenset({'a'}), frozenset({'c'}), frozenset({'b'})])
        self.assertEqual(
            Counter(candidate_itemsets), 
            Counter([frozenset({'a', 'b'}), frozenset({'a', 'c'}), frozenset({'c', 'b'})])
        )

        candidate_itemsets = self.apriori_algorithm_case_1._generate_candidate_itemsets(
            [frozenset({'a', 'b'}), frozenset({'a', 'c'}), frozenset({'c', 'b'})]
        )
        self.assertEqual(
            Counter(candidate_itemsets), 
            Counter([frozenset({'c', 'b', 'a'})])
        )

        candidate_itemsets = self.apriori_algorithm_case_1._generate_candidate_itemsets([['a'], ['b'], ['c']])
        self.assertEqual(
            Counter(candidate_itemsets), 
            Counter([frozenset({'a', 'b'}), frozenset({'a', 'c'}), frozenset({'c', 'b'})])
        )

        candidate_itemsets = self.apriori_algorithm_case_1._generate_candidate_itemsets(
            [['a','b'], ['b','c'], ['a','c']]
        )
        self.assertEqual(
            Counter(candidate_itemsets), 
            Counter([frozenset({'c', 'b', 'a'})])
        )

        
    def test_generate_all_frequent_itemsets(self):
        self.apriori_algorithm_case_1.generate_all_frequent_itemsets()
        self.assertEqual(
            self.apriori_algorithm_case_1.all_frequent_itemsets,
            {
                frozenset({'c'}): 0.5625,
                frozenset({'b'}): 0.3125,
                frozenset({'a'}): 0.375,
                frozenset({'b', 'c'}): 0.125,
                frozenset({'a', 'c'}): 0.125,
                frozenset({'a', 'b'}): 0.125
            }
        )

        self.apriori_algorithm_case_2.generate_all_frequent_itemsets()
        self.assertEqual(
            self.apriori_algorithm_case_2.all_frequent_itemsets,
            {
                frozenset({'c'}): 0.5625, 
                frozenset({'b'}): 0.3125, 
                frozenset({'a'}): 0.375
            }
        )

        self.apriori_algorithm_case_3.generate_all_frequent_itemsets()
        self.assertEqual(
            self.apriori_algorithm_case_3.all_frequent_itemsets,
            {
                frozenset({'c'}): 0.5625, 
                frozenset({'a'}): 0.375
            }
        )


    def test_generate_all_rules(self):
        self.apriori_algorithm_case_1.generate_all_rules()
        self.assertEqual(
            self.apriori_algorithm_case_1.all_rules,
            {
                (frozenset({'a'}), frozenset({'c'})): { 
                    'confident': 0.3333333,
                    'lift': 0.5925926,
                    'support': 0.125
                },
                (frozenset({'c'}), frozenset({'a'})): {
                    'confident': 0.2222222,
                    'lift': 0.5925926,
                    'support': 0.125
                },
                (frozenset({'a'}), frozenset({'b'})): {
                    'confident': 0.3333333,
                    'lift': 1.0666667,
                    'support': 0.125
                },
                (frozenset({'b'}), frozenset({'a'})): {
                    'confident': 0.4,
                    'lift': 1.0666667,
                    'support': 0.125
                },
                (frozenset({'c'}), frozenset({'b'})): {
                    'confident': 0.2222222,
                    'lift': 0.7111111,
                    'support': 0.125
                },
                (frozenset({'b'}), frozenset({'c'})): {
                    'confident': 0.4,
                    'lift': 0.7111111,
                    'support': 0.125
                }
            }
        )
        
        self.apriori_algorithm_case_1a = deepcopy(self.apriori_algorithm_case_1)
        self.apriori_algorithm_case_1a.set_minconf(0.3)
        self.apriori_algorithm_case_1a.generate_all_frequent_itemsets().generate_all_rules()
        self.assertEqual(
            self.apriori_algorithm_case_1a.all_rules,
            {
                (frozenset({'a'}), frozenset({'c'})): { 
                    'confident': 0.3333333,
                    'lift': 0.5925926,
                    'support': 0.125
                },
                (frozenset({'a'}), frozenset({'b'})): {
                    'confident': 0.3333333,
                    'lift': 1.0666667,
                    'support': 0.125
                },
                (frozenset({'b'}), frozenset({'a'})): {
                    'confident': 0.4,
                    'lift': 1.0666667,
                    'support': 0.125
                },
                (frozenset({'b'}), frozenset({'c'})): {
                    'confident': 0.4,
                    'lift': 0.7111111,
                    'support': 0.125
                }
            }
        )
        
        self.apriori_algorithm_case_1b = deepcopy(self.apriori_algorithm_case_1)
        self.apriori_algorithm_case_1b.set_minconf(0.4)
        self.apriori_algorithm_case_1b.generate_all_frequent_itemsets().generate_all_rules()
        self.assertEqual(
            self.apriori_algorithm_case_1b.all_rules,
            {
                (frozenset({'b'}), frozenset({'a'})): {
                    'confident': 0.4,
                    'lift': 1.0666667,
                    'support': 0.125
                },
                (frozenset({'b'}), frozenset({'c'})): {
                    'confident': 0.4,
                    'lift': 0.7111111,
                    'support': 0.125
                }
            }
        )


if __name__ == "__main__":
    test_apriori_algorithm = TestAprioriAlgorithm()
    test_apriori_algorithm.test__calculate_support()
    test_apriori_algorithm.test__prune_frequent_itemsets()
    test_apriori_algorithm.test__generate_frequent_single_itemsets()
    test_apriori_algorithm.test__generate_candidate_itemsets()
    test_apriori_algorithm.test_generate_all_frequent_itemsets()
    test_apriori_algorithm.test_generate_all_rules()
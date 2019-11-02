import matplotlib.pyplot as plt
import numpy as np


#===================================
# The purpose of this file is for creating linear regression function. What 
# a linear regression function should do:
# - Iterate until convergence using gradient descent
# - Normalize and scale to boost it up
# - Receive test data and return result
# - Predict on incoming data
#===================================


class UnivariateLinearRegression(object):
    def __init__(self, training_set, input_feature, output_feature):
        # Parameters
        self.input_feature = input_feature
        self.output_feature = output_feature

        # Attributes
        self.cost_list = None
        self.iteration_list = None
        self.learning_rate = 10**-5
        self.m = None
        self.test_set = None
        self.theta_0 = 0
        self.theta_1 = 0
        self.training_set = None

        # Call methods
        self.set_training_set(training_set)


    def calculate_cost(self, data_df):
        m = len(data_df)
        cost = 0
        for i in range(m):
            cost += (self.theta_0 + self.theta_1*data_df[self.input_feature].iloc[i] - data_df[self.output_feature].iloc[i])**2
        cost /= 2*m 
        return cost

    
    def draw_cost_plot(self):
        plt.plot(self.iteration_list, self.cost_list)
        plt.show()


    def draw_model(self):
        plt.plot(self.training_set[self.input_feature], self.training_set[self.output_feature], 'r.')
        plt.xlabel(self.input_feature)
        plt.ylabel(self.output_feature)

        x = np.linspace(0, self.training_set[self.input_feature].max() ,1000)
        y = self.theta_0 + self.theta_1*x
        plt.plot(x, y, '-b', label='model')
        plt.legend(loc='upper left')
        plt.show()


    def predict(self, input_value):
        output_value = self.theta_0 + self.theta_1*input_value
        print(f'Predicted value: {output_value}')
        return output_value
    
    
    def test(self, test_set=None):
        if test_set:
            test_cost = self.calculate_cost(test_set)
        else:
            test_cost = self.calculate_cost(self.test_set)
        print('Cost of test set:', test_cost)
        return test_cost


    def train(self):
        print('Start training.')
        cost = self.calculate_cost(self.training_set)
        print(f'Init theta_0: {self.theta_0}, theta_1: {self.theta_1}, cost: {cost}')

        self.cost_list = [cost]
        self.iteration_list = [0]

        t = 1
        is_continue = True
        while is_continue:
            derive_theta_0 = 0
            derive_theta_1 = 0
            for i in range(self.m):
                derive_theta_0 += self.theta_0 + self.theta_1*self.training_set[self.input_feature].iloc[i] - self.training_set[self.output_feature].iloc[i]
                derive_theta_1 += (self.theta_0 + self.theta_1*self.training_set[self.input_feature].iloc[i] - self.training_set[self.output_feature].iloc[i])*self.training_set[self.input_feature].iloc[i]
            derive_theta_0 /= self.m
            derive_theta_1 /= self.m

            self.theta_0 += - self.learning_rate*derive_theta_0
            self.theta_1 += - self.learning_rate*derive_theta_1

            cost = self.calculate_cost(self.training_set)
            print(f'Iterate: {t}, new cost: {cost}')

            self.cost_list.append(cost)
            self.iteration_list.append(t)
            t += 1

            if self.cost_list[-1] > self.cost_list[-2]:
                print('Cost is increasing. Please set a lower learning rate ') 
                break
            elif (self.cost_list[-1] - self.cost_list[-2])*100/self.cost_list[-2] > -0.0001:
                is_continue = False
                print(f'Finish training. theta_0: {self.theta_0}, theta_1: {self.theta_1}')
                self.draw_model()
                self.draw_cost_plot()
                

    def set_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate


    def set_test_set(self, test_set):
        self.test_set = test_set[[self.input_feature, self.output_feature]].copy()

        count_null_values = len(self.test_set) - len(self.test_set.dropna())
        if count_null_values > 0:
            print(f'Dropped {count_null_values} null rows of test set')
            self.test_set.dropna(inplace = True)


    def set_training_set(self, training_set):
        self.training_set = training_set[[self.input_feature, self.output_feature]].copy()

        count_null_values = len(self.training_set) - len(self.training_set.dropna())
        if count_null_values > 0:
            print(f'Dropped {count_null_values} null rows of training set')
            self.training_set.dropna(inplace = True)
        
        self.m = len(self.training_set)
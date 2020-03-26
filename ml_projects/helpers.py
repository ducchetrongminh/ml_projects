import numpy as np 
import matplotlib.pyplot as plt


def plot_by_feature(df_, output_feature):
    input_features = df_.columns.to_list()
    input_features.remove(output_feature)

    for input_feature in input_features:
        if df_[input_feature].dtype not in [np.int64, np.float64]:
            continue
            
        plt.plot(df_[input_feature], df_[output_feature], 'r.')
        plt.xlabel(input_feature)
        plt.ylabel(output_feature)
        plt.suptitle(input_feature)
        plt.show()
        print('\n\n\n')
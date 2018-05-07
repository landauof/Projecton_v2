# imports
import pandas
from collections import Counter

# vars
input_path = 'MFDCA-DATA/FraudedFeatureOutputs/output'
feature_selection_output_path =\
    'MFDCA-DATA/FraudedFeatureSelectedOutputs/output'
number_of_user = 5
number_of_features = 225

def load_data_of_user():
    """
    Loads data
    :param:
    :return: user data
    """
    df = pandas.read_csv(input_path+str(number_of_user)+'.csv')
    return df

def x_occurrences_num_y_class(data_frame):
    col_num = data_frame.shape[1] - 1

    array = data_frame.values
    X = array[:, 0:col_num]  # number of occurrences of each feature
    Y = array[:, col_num]  # Class

    return col_num, X, Y


def most_common_ngrams_signle_user():
    """
    finds the top featres of a user (based on occurrences) and outputs a file with it
    :return:
    """
    data_frame = load_data_of_user()

    col_num, X, Y = x_occurrences_num_y_class(data_frame)

    # feature selection

    score = [0] * col_num
    for i in X:
        score = score + i

    feature_dictionary = Counter(dict(zip(data_frame.columns, score)))    # map in order to easily find top features
    top_features = feature_dictionary.most_common(number_of_features)
    top_features = dict(top_features)
    top_features = top_features.keys()

    for col in data_frame.columns:
        if col not in top_features:
            data_frame.pop(col)

    data_frame = data_frame.sort_index(axis=1)
    data_frame['Class'] = Y
    data_frame.to_csv(feature_selection_output_path + str(number_of_user)+'.csv')
    print("selection")
    most_common_ngrams_other_users(top_features)


def most_common_ngrams_other_users(top_features_of_user):
    """
    creates files for other users based on the features chosen before (only classified blocks)
    :return:
    """
    mega_df = pandas.DataFrame()
    for other_user in range(0, 40):
        if other_user == number_of_user:
            continue
        data_frame = pandas.read_csv(input_path+str(other_user)+'.csv')
        data_frame = data_frame[:-100]  # drop all unclassified rows
        col_num, X, Y = x_occurrences_num_y_class(data_frame)

        # feature selection

        score = [0] * col_num
        for i in X:
            score = score + i

        for col in data_frame.columns:
            if col not in top_features_of_user:
                data_frame.pop(col)

        for feature in top_features_of_user:
            if feature not in data_frame.columns:
                data_frame[feature] = 0

        data_frame = data_frame.sort_index(axis = 1)
        data_frame['Class'] = 1
        l = [mega_df, data_frame]
        mega_df = pandas.concat(l, axis=0)
        # data_frame.to_csv(feature_selection_output_path + str(other_user)+'.csv')
        print("selection")

    mega_df.to_csv(feature_selection_output_path + "ALL"+'.csv')

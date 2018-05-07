# Imports
from nltk import ngrams
from itertools import chain
from collections import Counter
import csv
# Variables
path = 'MFDCA-DATA/FraudedRawData/User'

path_to_classification = 'MFDCA-DATA/challengeToFill.csv'

path_output = 'MFDCA-DATA/FraudedFeatureOutputs/output'


# Magic
def ngramize(commands, num):
    """
    :param commands:
    :param num:
    :return: num-grams of commands
    """
    return ngrams(commands.split(), num)


def read_file(user_number):
    """
    :param user_number:
    :return: commands from the input file
    """
    file = open(path+user_number, 'r')
    text = file.read()
    file.close()
    return text


def get_all_ngrams():
    """
    Iterate over all command files
    :return: a long list of num-grams
    """
    grams = ()
    for i in range(0, 1):
        text_i = read_file(str(i))
        curr_grams = ngramize(text_i, n)
        grams = chain(grams, curr_grams)
    return grams


def count_occurrences(gen):
    return Counter(gen)


def debug_all_users_at_once():
    all_grams = get_all_ngrams()    # all_grams = n-gram generator
    gram_and_occurrences = count_occurrences(all_grams)    # holds all grams and number of occurrences
    print(gram_and_occurrences)


def pre_process_of_commands(commands_as_string):
    """
    Turns string to list of 100 commands per string
    :param commands_as_string:
    :return: list of strings (blocks)
    """
    ans = []
    text = commands_as_string.split()
    for i in range(int(len(text)/100)):
        new_block = ''
        for j in range(100):
            new_block = new_block + text[i*100 + j] + ' '
        ans.append(new_block)
    return ans


def features_per_block(user):
    """
    :param user: the user number
    :return: list of counters of feature and occurrences
    """
    list_of_features_per_block = []
    block_list = pre_process_of_commands(read_file(str(user)))
    for block in block_list:
        block_grams = ngramize(block, n)
        block_features = count_occurrences([str(item) for item in block_grams])
        list_of_features_per_block.append(block_features)
    return list_of_features_per_block


def per_user_feature_extraction():
    """
    Iterates on Users
    :return: List (users) of lists (blocks) of counters (n-grams : occurrences)
    """
    ans = []    # List of counters (dictionaries)
    for i in range(0, 40):
        gram_and_occurrences = features_per_block(i)
        ans.append(gram_and_occurrences)
    return ans


def export_to_csv_single_user(user_number=0):
    with open(path_output+str(user_number)+'.csv', 'w', newline='') as csvFile:     # newline='' removes the empty lines
        n_grams = features_per_block(user_number)
        field_names = get_coloum_headers(user_number)
        field_names = list(set(field_names))
        field_names.append('Class')
        writer = csv.DictWriter(csvFile, fieldnames=field_names)
        writer.writeheader()
        block_number = 0
        for block in n_grams:
            block_number = block_number + 1
            block_dict = dict([(name, block[name]) for name in field_names])
            block_dict['Class'] = get_block_classification(block_number, user_number)
            writer.writerow(block_dict)


def get_single_user_ngrams(ind):
    text_i = read_file(str(ind))
    curr_grams = ngramize(text_i, n)
    return curr_grams
    # [str(s) for s in curr_grams]


def get_coloum_headers(i):
    blocks = features_per_block(i)
    res = []
    for single_block in blocks:
        for ngram in single_block:
            res.append(ngram)
    return res


def get_block_classification(block_number, user_number=0):
    with open(path_to_classification, 'r') as csvFile:
        temp = (csvFile.read()).split(',')[block_number+150*(user_number+1)]
        return temp.split("""
User""")[0]


def export_to_csv_all_users(num):
    """
    Create output files for all users
    :return: void
    """
    global n
    n = num
    for user_number in range(40):
        export_to_csv_single_user(user_number)
        print("done user {}".format(user_number))


# export_to_csv_all_users(3)

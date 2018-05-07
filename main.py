import FeatureExtraction as fe
import FeatureSelection
import Classifier
from pathlib import Path


# Consts
out_0_path = 'MFDCA-DATA/FraudedFeatureOutputs/output0.csv'
number_to_ngram = 3
is_changing_n = False


def run():
    """
    main function - includes feature extraction, selection and classifier
    :return:
    """
    # array to hold all accuracy values
    all_ans = []

    # Feature Extraction
    out_0_file = Path(out_0_path)
    if is_changing_n:
        fe.export_to_csv_all_users(number_to_ngram)
    if not out_0_file.exists():
        fe.export_to_csv_all_users(number_to_ngram)

    # now all files contains the features and stuff

    # Feature Selection
    for user_number in range(0, 1):
        # Feature selection
        FeatureSelection.most_common_ngrams_signle_user()
        Classifier.classify()

        # now FraudedFeatureOutputs contains a file of features X blocks for each user, and the features were chosen
        # based on a specific user we will classify


run()

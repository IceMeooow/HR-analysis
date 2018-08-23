import pickle
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

from sklearn.model_selection import StratifiedShuffleSplit
import pdb

class OracleTrainer:
    '''
    Class is dedicated for training process.
    '''
    def __init__(self, path):
        '''
        Initial method for initializing instance.
        :param path: path to file in csv format.
        '''
        self.path = path

    def train(self):
        '''
        Method implements training process and saves needed models.
        :return: classifier_dict, dictionary contains classifier (sklearn.ensemble.forest.RandomForestClassifier)
                                and its accuracy score.
        '''
        data_frame = self._read_data()
        data_frame = self._clean_data(data_frame)

        sales_encoder, data_frame.sales = self._encode_data(data_frame.sales)
        salary_encoder, data_frame.salary = self._encode_data(data_frame.salary)

        labels, data = self._select_data_and_labels(data_frame)
        train_data, test_data, train_labels, test_labels = self._split_data_on_train_test_set(data, labels)

        classifier_dict = self._fit_model(train_data, test_data, train_labels, test_labels)

        self._save_file("sales_encoder", sales_encoder)
        self._save_file("salary_encoder", salary_encoder)
        self._save_file("classifier_dict", classifier_dict)
        return classifier_dict

    def _read_data(self):
        '''
        Method performs reading data from csv file and converts it to data frame.
        :return: data frame, in the format of pandas.core.frame.DataFrame.
        '''
        data_frame = pd.read_csv(self.path)
        return data_frame

    def _clean_data(self, data_frame):
        '''
        Method deletes duplicates and resets index in the data frame.
        :param data_frame: data frame (pandas.core.frame.DataFrame), which contains duplicates.
        :return: cleaned data frame, in the format of pandas.core.frame.DataFrame.
        '''
        data_frame.drop_duplicates(inplace=True)
        new_data_frame = data_frame.reset_index(drop=True)
        return new_data_frame

    def _encode_data(self, column):
        '''
        Method converts categorical column into indicator column.
        :param column: column (pandas.core.series.Series), which contains categorical values.
        :return: encode_model, model (sklearn.preprocessing.label.LabelEncoder), model which encodes provided column.
                encoded_column, encoded column in the format of pandas.core.series.Series.
        '''
        encode_model = preprocessing.LabelEncoder()
        encode_model.fit(column)
        encoded_column = encode_model.transform(column)
        return encode_model, encoded_column

    def _select_data_and_labels(self, data_frame):
        '''
        Method selects the right columns for data (contains only features) and labels.
        :param data_frame: data frame (pandas.core.frame.DataFrame).
        :return: labels, in the format of pandas.core.series.Series.
                data, in the format of pandas.core.frame.DataFrame.
        '''
        labels = data_frame["left"]
        data = data_frame.drop(["left"], axis=1)
        return labels, data

    def _split_data_on_train_test_set(self, data, labels):
        '''
        Method splits data, labels in train/test sets.
        :param data: data frame (pandas.core.frame.DataFrame), which contains only features.
        :param labels: labels for data (pandas.core.series.Series).
        :return: train_data, test_data - contains only features, in the format of pandas.core.frame.DataFrame.
                train_labels, test_labels - contains only labels, in the format of pandas.core.series.Series.
        '''
        # train_data, test_data, train_labels, test_labels = train_test_split(data,
        #                                                                     labels,
        #                                                                     test_size=0.2,
        #                                                                     random_state=13)

        sss = StratifiedShuffleSplit(n_splits=3, test_size=0.25, random_state=0)
        for train_index, test_index in sss.split(data, labels):
            train_data, test_data = data.loc[train_index], data.loc[test_index]
            train_labels, test_labels = labels[train_index], labels[test_index]

        return train_data, test_data, train_labels, test_labels

    def _fit_model(self, train_data, test_data, train_labels, test_labels):
        '''
        Method creates a classifier and fits it using training data and labels.
        Then it calculates accuracy score on the given test data and labels.
        :param train_data: training data (pandas.core.frame.DataFrame).
        :param test_data: test data (pandas.core.frame.DataFrame).
        :param train_labels: training labels (pandas.core.series.Series).
        :param test_labels: test labels (pandas.core.series.Series).
        :return: classifier_dict, dictionary contains classifier (sklearn.ensemble.forest.RandomForestClassifier)
                                and its accuracy score.
        '''
        classifier = RandomForestClassifier(n_estimators=33,
                                            criterion="gini",
                                            max_depth=21,
                                            random_state=0,
                                            class_weight="balanced")
        classifier.fit(train_data, train_labels)
        score = classifier.score(test_data, test_labels)
        predicted_test = classifier.predict(test_data)
        f1 = f1_score(test_labels, predicted_test, average='weighted')
        classifier_dict = {"classifier": classifier, "score": score, "f1_score": f1}
        return classifier_dict

    def _save_file(self, file_name, file):
        '''
        Method saves provided file.
        :param file_name: name of file(str).
        :param file: file, which is needed to save.
        :return: -
        '''
        file_path = "saved_models/" + file_name + ".pkl"
        with open(file_path, "wb") as savefile:
            pickle.dump(file, savefile, protocol=pickle.HIGHEST_PROTOCOL)

import pickle
import pandas as pd


class Oracle:
    '''
    Class is dedicated for forecasting process.
    '''
    def __init__(self, answers_dict):
        '''
        Initial method for initializing instance.
        :param answers_dict: dictionary of answers on survey.
                Dictionary should include the follow keys (only in such order):
                                                                        - 'satisfaction_level',
                                                                        - 'last_evaluation',
                                                                        - 'number_project',
                                                                        - 'average_montly_hours',
                                                                        - 'time_spend_company',
                                                                        - 'Work_accident',
                                                                        - 'promotion_last_5years',
                                                                        - 'sales',
                                                                        - 'salary'.
        '''
        self.answers_dict = answers_dict

    def predict(self):
        '''
        Method implements forecasting process.
        It transforms dictionary, encodes a needed  data and predicts that employee will stay in the company or not.
        :return: prediction, predicted value (0 - stay, 1 - leave).
                confidence, confidence of model in its decision (float).
        '''
        data_frame = self._transform_to_data_frame(self.answers_dict)

        sales_encoder = self._open_file("sales_encoder")
        salary_encoder = self._open_file("salary_encoder")
        classifier_dict = self._open_file("classifier_dict")

        data_frame.sales = sales_encoder.transform(data_frame.sales)
        data_frame.salary = salary_encoder.transform(data_frame.salary)

        classifier = classifier_dict["classifier"]
        prediction = classifier.predict(data_frame)
        confidence = max(classifier.predict_proba(data_frame)[0]) * 100
        return prediction, confidence

    def _transform_to_data_frame(self, dictionary):
        '''
        Method transforms dictionary of answers into data frame.
        :param dictionary: dictionary of answers on survey.
        :return: data frame, in the format of pandas.core.frame.DataFrame.
        '''
        data_frame = pd.DataFrame([dictionary], columns=dictionary.keys())
        return data_frame

    def _open_file(self, file_name):
        '''
        Method uploads provided file.
        :param file_name: name of file(str).
        :return: uploaded file.
        '''
        file_path = "saved_models/" + file_name + ".pkl"
        with open(file_path, "rb") as openfile:
            file = pickle.load(openfile)
        return file

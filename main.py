from trainer import OracleTrainer
from forecaster import Oracle

# train the model
obj_train = OracleTrainer("HR_comma_sep.csv")
print(obj_train.train())

# test the model
# - example of answers
answers_dict = {'satisfaction_level': 0.2,
                'last_evaluation': 0,
                'number_project': 2,
                'average_montly_hours': 200,
                'time_spend_company': 5,
                'Work_accident': 0,
                'promotion_last_5years': 0,
                'sales': "IT",
                'salary': "low"}

# - predict the result
obj_pred = Oracle(answers_dict)
prediciton, confidence = obj_pred.predict()

if prediciton == 0:
    print("will stay")
    print("with confidence {} %".format(confidence))
else:
    print("will leave")
    print("with confidence {} %".format(confidence))

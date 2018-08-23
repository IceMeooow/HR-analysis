import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn import preprocessing



data_frame = pd.read_csv("HR_comma_sep.csv")
data_frame.drop_duplicates(inplace=True)
data_frame = data_frame.reset_index(drop=True)

# create a plot that shows the distribution of the target function - the "left"
plt.figure(figsize=(6, 4))
plt.scatter(range(data_frame.shape[0]), np.sort(data_frame.left.values))
plt.xlabel('index', fontsize=12)
plt.ylabel('left', fontsize=12)
plt.title("Target variable is left. It is  its distribution:")
plt.show()

# visualize the correlation between all features
corr = data_frame.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr, annot=True, cmap="YlGnBu")
plt.title("Correlation matrix")
plt.show()

# display the distribution of numeric features
for column in ['satisfaction_level', 'last_evaluation', 'number_project',
               'average_montly_hours', 'time_spend_company']:
    plt.figure(figsize=(8, 6))
    plt.title(column)
    sns.distplot(data_frame[column])
    plt.show()

# display the distribution of categorical features
for column in ['sales', 'salary']:
    plt.figure(figsize=(10, 8))
    ax = sns.countplot(data_frame[column])
    plt.title(column)
    plt.show()

# Show how the features affect the target "left"
# - numeric features
def plot_distribution(df, var, target, yl=4, **kwargs):
    row = kwargs.get('row', None)
    col = kwargs.get('col', None)
    facet = sns.FacetGrid(df, hue=target, aspect=4, row=row, col=col)
    facet.map(sns.kdeplot, var, shade=True)
    facet.set(xlim=(0, df[var].max()), ylim=(0, yl))
    facet.add_legend()
    plt.show()

plot_distribution(data_frame, 'satisfaction_level', 'left')
plot_distribution(data_frame, 'last_evaluation', 'left')
plot_distribution(data_frame, 'number_project', 'left', yl=1.5)
plot_distribution(data_frame, 'average_montly_hours', 'left', yl=0.015)
plot_distribution(data_frame, 'time_spend_company', 'left', yl=1)


# - categorical features
with open("saved_models/sales_encoder.pkl", "rb") as openfile:
    sales_encoder = pickle.load(openfile)

with open("saved_models/salary_encoder.pkl", "rb") as openfile:
    salary_encoder = pickle.load(openfile)

data_frame.sales = sales_encoder.transform(data_frame.sales)
data_frame.salary = salary_encoder.transform(data_frame.salary)

plot_distribution(data_frame, 'sales', 'left', yl=0.4)
plot_distribution(data_frame, 'salary', 'left', yl=3)

# boxplot
for column_name in ["satisfaction_level", "last_evaluation", "number_project",
                    "average_montly_hours", "time_spend_company", "sales", "salary"]:
    plt.figure(figsize=(12, 10))
    sns.factorplot(y=column_name, x="left", data=data_frame, kind="box")
    plt.show()

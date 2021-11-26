import pandas as pd
from constants import DataDir
import util
import json
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

df = pd.DataFrame(pd.read_hdf(DataDir.all_tables,
                  key='df',
                  mode='r'))

# do general cleaning
df = df.dropna()
df = df.drop_duplicates()

x = df.iloc[: , :(len(df.columns)-2)]
y = df.iloc[: , (len(df.columns)-1):]

converted = util.convert_input_column_type(x)
dtypes = converted.dtypes.astype(str).to_dict()
# with open('output/dtypes.json', 'w') as f:
#     json.dump(dtypes, f)
# converted.head(30).to_csv("output/head.csv")

converted = converted.drop(['srcip','dstip'], axis=1)

x_train, x_test, y_train, y_test = train_test_split(converted, y.values.ravel(), test_size=0.25)
clf = LogisticRegression()
clf.fit(x_train, y_train)
score = clf.score(x_test, y_test)
print(f"Training accuracy: {score}")

# try to predict attack category
y = df.iloc[: , -2:-1]
x_train, x_test, y_train, y_test = train_test_split(converted, y.values.ravel(), test_size=0.25)
clf = LogisticRegression()
clf.fit(x_train, y_train)
score = clf.score(x_test, y_test)
print(f"Training accuracy: {score}")
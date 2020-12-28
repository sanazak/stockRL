import pandas as pd 
import numpy as np 
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV



df = pd.read_csv('normalized_dataset.csv', index_col=0)
df = df.dropna()



features = list(df.columns[2:-2])
features = ['marketOpen', 'marketClose', 'marketVolume', 'marketNumberOfTrades',
            # 'prevClose_0', 'prevClose_1' , 'prevClose_2' ,' prevClose_3', 'prevClose_4', 'prevClose_5' , 'prevClose_6', 
            'prevClose_0', # 'prevClose_8', 'prevClose_9',
            '%change']
print("* features:", features, sep="\n")




train_df, test_df = train_test_split(df, test_size=0.2)
# test_sym = 'RUN'
# train_df = df[df['sym']!=test_sym]
# test_df = df[df['sym']==test_sym]#.reset_index()
split_date = pd.to_datetime("2020-07-08 9:30", infer_datetime_format=True)  
df['date'] = pd.to_datetime(df['date'], infer_datetime_format=True)  

# test_df = df[df['date'] < split_date]
# train_df = df[df['date'] >= split_date]
days_to_test =[17, 22, 8, 16, 25]
# test_df = df[df['date'].dt.day.isin(days_to_test)]
# train_df = df[~df['date'].dt.day.isin(days_to_test)]

print('len(test_df)', len(test_df), 'len(train_df)', len(train_df))


y = train_df["target"]
X = train_df[features]
clf = DecisionTreeClassifier(random_state=99)
rfclf = RandomForestClassifier(random_state=99, class_weight='balanced' )
param_grid = {'n_estimators': [4, 10, 20],
              'max_depth': [3, 9, 15],
              'min_samples_split': [2, 5, 10]}
clf = GridSearchCV(rfclf, param_grid, cv=5)


# clf = LogisticRegression(random_state=0, class_weight='balanced' )

clf.fit(X, y)
print(clf.best_estimator_)

y_pred = clf.predict(X)
print(confusion_matrix(y, y_pred))
print(classification_report(y, y_pred))



y_true = test_df["target"]
Xtest = test_df[features]
y_pred = clf.predict(Xtest)
print(confusion_matrix(y_true, y_pred))
print(classification_report(y_true, y_pred))


def dummy_decison(row):
    decision = 'hold'
    # if row['marketHigh'] - row['marketOpen'] > 0.005:
    #     decision = 'buy'
    # elif row['marketHigh'] - row['marketOpen'] < -0.01:
    #     decision = 'sell'
    if row['%change'] < -.1:
        decision = 'sell'
    elif row['%change'] > .1:
        decision = 'buy'
    return decision

print('----- dummy decision:')
y_dum = test_df.apply(lambda row: dummy_decison(row), axis=1)
print(confusion_matrix(y_true, y_dum))
print(classification_report(y_true, y_dum))

def visualize_tree(tree, feature_names):
    """Create tree png using graphviz.

    Args
    ----
    tree -- scikit-learn DecsisionTree.
    feature_names -- list of feature names.
    """
    with open("dt.dot", 'w') as f:
        export_graphviz(tree, out_file=f,
                        feature_names=feature_names)

    command = ["dot", "-Tpng", "dt.dot", "-o", "dt.png"]
    try:
        subprocess.check_call(command)
    except:
        exit("Could not run dot, ie graphviz, to "
             "produce visualization")

# visualize_tree(clf, features)
# dot -Tpng dt.dot -o dt.png
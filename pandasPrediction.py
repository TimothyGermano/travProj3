import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import plot_tree
import pickle
df = pd.read_csv("troop_movements.csv")
# print(df)

teamsData = df['empire_or_resistance'].value_counts()
# print(teamsData)

# teamsData = df['unit_id', 'homeworld'].value_counts()
# print(teamsData)

teamsData = df.groupby('homeworld')['unit_id'].count()
# print(teamsData)

# teamsData = df.groupby('unit_type')['unit_id'].count()
# print(teamsData)

#air_quality["ratio_paris_antwerp"] = (
   # air_quality["station_paris"] / air_quality["station_antwerp"]

data = df['is_resistance'] = (df['empire_or_resistance'] == 'resistance')
# print(df)

# sns.countplot(x='is_resistance', data=df)
# plt.show()

# x = pd.get_dummies(df[['unit_type','homeworld']])
# y = df['is_resistance']
# X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
# clf = DecisionTreeClassifier(random_state=42)
# clf.fit(X_train, y_train)
# y_pred = clf.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# print(f"Accuracy: {accuracy}")
# plt.figure(figsize=(12, 8))
# plot_tree(clf, feature_names=x.columns, class_names=['Not Resistance', 'Resistance'], filled=True)
# plt.show()

# importances = model.feature_importances_

# # feature_importances = pd.DataFrame({'Feature'})

# x_encoded = pd.get_dummies(df[['unit_type', 'homeworld']], prefix=['unit_type', 'homeworld'])
 
# y_encoded = df['is_resistance']
 
# X_train, X_test, y_encoded_train, y_encoded_test = train_test_split(x_encoded, y_encoded, test_size=0.3, random_state=42)
 
# clf = DecisionTreeClassifier(random_state=42)
# clf.fit(X_train, y_encoded_train)
 
# feature_importances = clf.feature_importances_
 
# features_df = pd.DataFrame({
#     'feature': x_encoded.columns,
#     'importance': feature_importances
# })
 
# features_df = features_df.sort_values(by='importance', ascending=False)
 
# plt.figure(figsize=(14, 8))
# plt.bar(features_df['feature'], features_df['importance'])
# plt.title('Feature Importance')
# plt.ylabel('Importance')
# plt.xlabel('Feature')
# plt.xticks(rotation=90)
# plt.tight_layout()
# plt.show()
#with open('decision_tree_model.pkl', 'wb') as file:9    pickle.dump(clf, file)

df = pd.read_csv("troop_movements_1m.csv")
print(df)
df['unit_type'].replace(to_replace='invalid_unit', value='unknown', inplace=True)
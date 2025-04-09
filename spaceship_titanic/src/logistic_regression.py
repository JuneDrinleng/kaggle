import pandas as pd
data_path="data/train.csv"
df=pd.read_csv(data_path)
df_columns = df.columns


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='most_frequent')
df_imputed_array = imputer.fit_transform(df)
df_imputed_sklearn = pd.DataFrame(df_imputed_array, columns=df.columns, index=df.index)
# print(df_imputed_sklearn.isnull().sum()) 
# print(df_imputed_sklearn.dtypes)
original_types = df.dtypes
for col in df_imputed_sklearn.columns:
    try:
        df_imputed_sklearn[col] = df_imputed_sklearn[col].astype(original_types[col])
    except Exception as e:
        print(f"Could not convert column '{col}' back to {original_types[col]}: {e}")


train_data=df_imputed_sklearn.drop(columns=['Name', 'PassengerId', 'Transported'])
split_data = df_imputed_sklearn['Cabin'].str.split('/', expand=True)
split_data.columns = ['Deck', 'Num', 'Side']
split_data['Num'] = split_data['Num'].astype('Int64')
train_data=train_data.drop(columns=['Cabin'])
train_data=pd.concat([train_data, split_data], axis=1)
train_label=df_imputed_sklearn['Transported']
for col in train_data.columns:
    if train_data[col].dtype == 'object':
        col_data=train_data[col]
        onehot_data=pd.get_dummies(col_data, prefix=col,dtype=int)
        train_data=train_data.drop(columns=[col])
        train_data=pd.concat([train_data, onehot_data], axis=1)

train_data.to_csv('processed_data/train_data.csv', index=False)
train_label.to_csv('processed_data/train_label.csv', index=False)
from sklearn.model_selection import train_test_split
train_data, val_data, train_label, val_label = train_test_split(train_data, train_label, test_size=0.2, random_state=42)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline 
from sklearn.preprocessing import StandardScaler
pipe = Pipeline([
    ('scaler', StandardScaler()),              # 第一步：缩放
    ('logistic', LogisticRegression(max_iter=10000000)) # 第二步：逻辑回归 (也可以先试默认 max_iter)
])
# model = LogisticRegression(max_iter=1000000,solver='liblinear')
# model.fit(train_data, train_label)
# predictions = model.predict(val_data)

pipe.fit(train_data, train_label)
predictions = pipe.predict(val_data)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(val_label, predictions)
print(f"Accuracy: {accuracy * 100:.2f}%")


test_data_path="data/test.csv"
df_test=pd.read_csv(test_data_path)
imputer = SimpleImputer(strategy='most_frequent')
df_imputed_array_test = imputer.fit_transform(df_test)
df_imputed_sklearn_test = pd.DataFrame(df_imputed_array_test, columns=df_test.columns, index=df_test.index)
original_types = df_test.dtypes
for col in df_imputed_sklearn_test.columns:
    try:
        df_imputed_sklearn_test[col] = df_imputed_sklearn_test[col].astype(original_types[col])
    except Exception as e:
        print(f"Could not convert column '{col}' back to {original_types[col]}: {e}")
test_data=df_imputed_sklearn_test.drop(columns=['Name', 'PassengerId'])
split_data_test = df_imputed_sklearn_test['Cabin'].str.split('/', expand=True)
split_data_test.columns = ['Deck', 'Num', 'Side']
split_data_test['Num'] = split_data_test['Num'].astype('Int64')
test_data=test_data.drop(columns=['Cabin'])
test_data=pd.concat([test_data, split_data_test], axis=1)
for col in test_data.columns:
    if test_data[col].dtype == 'object':
        col_data=test_data[col]
        onehot_data=pd.get_dummies(col_data, prefix=col,dtype=int)
        test_data=test_data.drop(columns=[col])
        test_data=pd.concat([test_data, onehot_data], axis=1)
# predictions = model.predict(test_data)
predictions = pipe.predict(test_data)
id_array=df_test['PassengerId'].values
predictions_df = pd.DataFrame({'PassengerId': id_array, 'Transported': predictions})
predictions_df['Transported'] = predictions_df['Transported'].astype(bool)
predictions_df.to_csv('predict/submission_logicregression_4.csv', index=False)
test_data.to_csv('processed_data/test_data.csv', index=False)
test_ids = df_test['PassengerId']
test_ids.to_csv('processed_data/test_ids.csv', index=False)
pass
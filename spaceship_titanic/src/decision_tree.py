import pandas as pd
import time
train_data= pd.read_csv('processed_data/train_data.csv')
test_data= pd.read_csv('processed_data/test_data.csv')
train_label= pd.read_csv('processed_data/train_label.csv')
test_ids= pd.read_csv('processed_data/test_ids.csv')
#random forest model
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
# model.fit(train_data, train_label)
model.fit(train_data, train_label.values.squeeze())  
predictions= model.predict(test_data)
df_predict=pd.DataFrame(predictions)
df=pd.concat([test_ids, df_predict], axis=1)
df.columns=['PassengerId', 'Transported']
date=time.strftime("%Y-%m-%d_%H-%M", time.localtime())
output_path = f"predict/submission_{date}.csv"
df.to_csv(output_path, index=False)
pass
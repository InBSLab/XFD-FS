from sklearn.model_selection import train_test_split
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import accuracy_score
from lightgbm import LGBMClassifier
import time
from sklearn.metrics import roc_auc_score, precision_score,recall_score
from sklearn.metrics import classification_report
dataset = pd.read_csv('D:\\DESKTOP\\dataset.csv')
X = dataset.values[:,0:179]
Y = dataset.values[:,179]
seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 1, stratify = Y)
rec_time_sum = 0  # timing 打点计时
rec_start = time.time()  # timing 打点计时
model = LGBMClassifier(max_depth=6)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
rec_end = time.time()  # timing 打点计时
rec_time_sum = rec_end - rec_start
dt_roc_auc = roc_auc_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
print("Accuracy: %.2f%%" %(accuracy*100.0))
print("Precision: %.2f%%" %(precision*100.0))
print("Recall: %.2f%%" %(recall*100.0))
print("lightgbm AUC = %2.4f" % dt_roc_auc)
print(classification_report(y_test, y_pred,digits=4))
print('Predict Time: %s' % '\n',rec_time_sum)

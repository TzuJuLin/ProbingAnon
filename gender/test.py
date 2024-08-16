import numpy as np
import pickle
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import pandas as pd

#load testing data
X_val = np.loadtxt('./data/original/val.csv', delimiter=',')
y_val = np.loadtxt('./data/original/val_y.csv', delimiter=',')

# df = pd.read_csv('/mount/arbeitsdaten/analysis/lintu/CV/test/test.csv', encoding='unicode_escape')


#load trained model
filename = 'gender_mix.sav'
loaded_model = pickle.load(open(filename, 'rb'))

#predict
y_pred = loaded_model.predict(X_val)

acc = format(accuracy_score(y_val, y_pred),".3f")
f1 = format(f1_score(y_val, y_pred), ".3f")
precision = format(precision_score(y_val, y_pred), ".3f")
recall = format(recall_score(y_val, y_pred), ".3f")

print("Accuracy: ", acc)
print("F1 Score: ", f1)
print("Precision: ", precision)
print("Recall: ", recall)

coefficients = loaded_model.coef_
intercept = loaded_model.intercept_
print(coefficients)
print(intercept)

# write predictions into csv file
# df['mix-anno_f0'] = y_pred
# df.to_csv('/mount/arbeitsdaten/analysis/lintu/CV/test/test.csv', index = False)

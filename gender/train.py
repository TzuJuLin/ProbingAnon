import numpy as np
from sklearn.linear_model import LogisticRegression
import pickle



model = LogisticRegression(verbose=True)
X_train = np.loadtxt('/mount/arbeitsdaten/analysis/lintu/thesis/gender/data/original/train.csv', delimiter=',')
y_train = np.loadtxt('/mount/arbeitsdaten/analysis/lintu/thesis/gender/data/original/train_y.csv', delimiter=',')

model.fit(X_train, y_train)

#save model with desired name
filename = 'gender_mix_f0_1.sav'
pickle.dump(model, open(filename, 'wb'))






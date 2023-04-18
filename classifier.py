
import pandas as pd
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import h5py as h5
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from joblib import dump
from sklearn.model_selection import learning_curve, ShuffleSplit

with h5.File('./h5/features.h5', mode='r') as f:
    jump_train = pd.DataFrame(f['/features/train/jumping'])
    walk_train = pd.DataFrame(f['/features/train/walking'])
    jump_test = pd.DataFrame(f['/features/test/jumping'])
    walk_test = pd.DataFrame(f['/features/test/walking'])
    

x = np.vstack((jump_train, walk_train))
y = np.concatenate((np.ones(jump_train.shape[0]), np.zeros(walk_train.shape[0])))
lr = LogisticRegression(max_iter=10000)
clf = make_pipeline(StandardScaler(), lr)
cv = ShuffleSplit(n_splits=10, test_size=0.2)

train_sizes = np.linspace(0.1, 1.0, 10)

# Use LearningCurve to compute training scores and validation scores
# for different sizes of the training set
train_sizes, train_scores, validation_scores = \
    learning_curve(clf, x, y, cv=cv, train_sizes=train_sizes, scoring='accuracy', n_jobs=-1)

# Plot the learning curve
plt.figure(figsize=(8, 6))
plt.plot(train_sizes, np.mean(train_scores, axis=1), label='Training score')
plt.plot(train_sizes, np.mean(validation_scores, axis=1), label='Validation score')
plt.xlabel('Training set size')
plt.ylabel('Accuracy')
plt.title('Learning curve')
plt.legend()
plt.show()


clf.fit(x, y)

dump(clf, './models/walk-jump-model.joblib')

z = np.vstack((jump_test, walk_test))
w = np.concatenate((np.ones(jump_test.shape[0]), np.zeros(walk_test.shape[0])))
z_scaled = clf.named_steps['standardscaler'].transform(z)

accuracy = lr.score(z_scaled, w)
print(f'Testing accuracy: {accuracy:.2f}')


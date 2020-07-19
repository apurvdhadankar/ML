import numpy as np 
from sklearn.ensemble import RandomForestClassifier

# Data: Students score in (math, languaege, creativity) --> study field

X = np.array([[9, 5, 4, "Computer Science"],
              [5, 1, 5, "Computer Science"],
              [8, 8, 8, "Computer Science"],
              [1, 10, 7, "literature"],
              [1, 8, 1, "liturature"],
              [5, 7, 9, "art"],
              [1, 1, 8, "art"]])

Forest = RandomForestClassifier(n_estimators=10).fit(X[:, :-1], X[:,-1])

students = Forest.predict([[8, 6, 5],
                           [3, 7, 9],
                           [2, 2, 1]])

print(students)
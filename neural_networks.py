from sklearn.neural_network import MLPRegressor
import numpy as np 

# Questionarrie Data (WEEK, YEARS, BOOKS, PROJECTS, ERN, RATING)

Q = [[20, 11, 20, 30,4000, 3000],
     [12,4,0,1000,1500],
     [2,0,1,10,0,1400],
     [35,5,10,70,6000,3800],
     [30,1,4,65,0,3900],
     [35,1,0,0,0,100],
     [15,1,2,25,0,3700],
     [40,3,-1,60,1000,2000],
     [40,1,2,95,0,1000],
     [10,0,0,45,0,1400],
     [30,1,0,50,0,1700],
     [1,0,0,45,0,1762]]

X = np.array(Q)

# One-Liner
neural_net = MLPRegressor().fit(X[:, :-1], X[:, -1])

# result
res = neural_net.predict([[0,0,0,0,0]])
print(res)
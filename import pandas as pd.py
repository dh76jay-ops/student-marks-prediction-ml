import pandas as pd
import numpy as np
from sklearn.model_selection\
import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

#Datatset create
data = {
    "Hours_Studied":
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
       "Marks":
    [30, 35, 40, 50, 55, 60, 65, 70, 80, 90]
}

df = pd.DataFrame(data)

X = df[["Hours_Studied"]]
Y = df["Marks"]

#Train test split
X_train, X_test, Y_train, y_test = train_test_split(X,Y,test_size=0.2)

# Model
model = LinearRegression()
model.fit(X_train,Y_train)

#prediction
prediction = model.predict([[7]])

print("Predicted marks for 7 hours study:",prediction)

#Graph
plt.scatter(X,Y)
plt.plot(X,model.predict(X))
plt.xlabel("Hours Studied")
plt.ylabel("Marks")
plt.show()
import numpy as np
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error as mae
from mlxtend.plotting import scatterplotmatrix as smp


df = pd.read_excel("50_Startups.xlsx")

independent_var = df.drop(columns="Profit")
profit = df["Profit"]

x_train,x_test, y_train, y_test = train_test_split(independent_var, profit, test_size = 0.2, random_state = 45)
# find the relation between all the attributes
""" cols = ["R&D Spend", "Administration", "Marketing Spend", "Profit"]
smp(df[cols].values, figsize = (14, 5), names = cols, alpha = 0.5) 
plt.tight_layout()
plt.show()"""

model = LinearRegression()
poly = PolynomialFeatures(degree = 2)
poly_x_train = poly.fit_transform(x_train)
poly_x_test = poly.transform(x_test)
model.fit(poly_x_train, y_train)
#(model.score(poly_x_train, y_train) = 0.9723211372631821

#Using testing data to predict
predicted = (model.predict(poly_x_test))

#Using the Testing data to predict
plt.scatter(predicted, y_test)
plt.plot(y_test, y_test, color='red', label='Line of Perfect Fit')
plt.xlabel("Predicted Values")
plt.ylabel("Actual Values")
plt.legend()
plt.title("Polynomial Regression")
#Check the scatter-plot between actual values and predicted values
#plt.show()

#Calculate evaluation metrics
MAE = mae(y_test, predicted)                                  # MAE = 5811.789454923782
MSE = mse(y_test, predicted)                                  # MSE = 41021483.58572562
RMSE = np.sqrt(MSE)                                           # RMSE = 6404.801603931664
MPE = np.mean((+predicted-y_test)/y_test)*100                 # MPE = 1.4911225575714462
R2 = r2_score(y_test, predicted)                              # R2_SCORE = 0.9785520337552707

#inputs
rnd, adm, mktspnd = list(map(float, input("Enter R&D Spend, Administration, Market Spend (Seperated by a space): ").split()))
input_values = {
    "R&D Spend": rnd,
    "Administration": adm,
    "Marketing Spend":mktspnd 
}
x_test = pd.DataFrame([input_values])
poly_x_test = poly.transform(x_test)
predicted = model.predict(poly_x_test)

print("The predicted profit is:", "\033[31m" + str(predicted[0]) + "\033[0m")

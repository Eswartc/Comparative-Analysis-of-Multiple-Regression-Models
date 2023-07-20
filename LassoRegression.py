import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
from mlxtend.plotting import scatterplotmatrix as smp

df = pd.read_excel("50_Startups.xlsx")
independent_vars = df.drop(columns="Profit")
actual_profits = df["Profit"]

x_train, x_test, y_train, y_test = train_test_split(independent_vars, actual_profits, test_size=0.15, random_state = 45) #Splitting into 85 : 15 (training : testing)
# find the relation between all the attributes
""" cols = ["R&D Spend", "Administration", "Marketing Spend", "Profit"]
smp(df[cols].values, figsize = (14, 5), names = cols, alpha = 0.5) 
plt.tight_layout()
plt.show()"""

model = Lasso(alpha = 1.0)
model.fit(x_train, y_train)
# model.score(x_train, y_train) = 0.9406818442523629

predicted = model.predict(x_test)
plt.scatter(predicted, y_test)
plt.plot(y_test, y_test, color='red', label='Line of Perfect Fit')
plt.xlabel("Predicted Values")
plt.ylabel("Lasso Values")
plt.legend()
#Check the scatter-plot between actual values and predicted values
#plt.show()

#Calculate evaluation metrics
MAE = mae(y_test, predicted)                                     # MAE = 5261.1495280498175
MSE = mse(y_test, predicted)                                     # MSE = 33026776.04701189
RMSE = np.sqrt(MSE)                                              # RMSE = 5746.892729728987
R2 = r2_score(y_test, predicted)                                 # R2_SCORE = 0.9827789020336958
MPE = np.mean((-y_test + predicted)/y_test)*100                  # MPE = 1.9795079893658891

#inputs
rnd, adm, mktspnd = list(map(float, input("Enter R&D Spend, Administration, Market Spend (Seperated by a space): ").split()))
input_values = {
    "R&D Spend": rnd,
    "Administration": adm,
    "Marketing Spend":mktspnd 
}
x_test = pd.DataFrame([input_values])
predicted = model.predict(x_test)

print("The predicted profit is:", "\033[31m" + str(predicted[0]) + "\033[0m") 
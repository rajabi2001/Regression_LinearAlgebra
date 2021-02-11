import numpy as np
import pandas
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from scipy import stats

# l = 10


col_list = ["Open"]
df = pandas.read_csv("C:\\Users\\rajab\\OneDrive\\Desktop\\linear algebra\\GOOGL.csv", usecols=col_list )
depended_variable_full = df.Open.to_list()
depended_variable = depended_variable_full[:-10]

# df = pandas.read_csv('C:\\Users\\rajab\\OneDrive\\Desktop\\linear algebra\\GOOGL.csv')
# size = len(df)
# l = size - 1250
# y = df.head(l)["Open"].to_numpy().reshape(-1,1)
y = depended_variable
l = len(y)
x = np.array([i+1 for i in range(l)]).reshape(-1,1)
z = np.array([1 for i in range(l)]).reshape(-1,1)
A = np.append(z,x,axis=1)
print(A)
# AT = A.T
AT = np.transpose(A)
ATA = np.dot(AT,A)
ATy = np.dot(AT,y)
# a = np.linalg.inv(ATA)
# B = np.dot(a,ATy)
# ATA = AT @ A
# ATy = AT @ y
B = np.linalg.solve(ATA , ATy)
intercept ,slope = B
print(B)

def myfunc(x):
  return slope * x + intercept

mymodel = list(map(myfunc, x))

plt.scatter(x, y)
plt.plot(x, mymodel)
plt.show()







# x2 = np.array([(i+1)**2 for i in range(l)]).reshape(-1,1)

# A2 = np.append(A,x2,axis=1)
# # print(A2)
# AT2 = np.transpose(A2)
# ATA2 = AT2 @ A2
# ATy2 = AT2 @ y
# B2 = np.linalg.solve(ATA2 , ATy2)
# print(B2)
# intercept2, slope1 ,slope2 = B2


# model = LinearRegression().fit(A, y)
# intercept = model.intercept_
# slope = model.coef_[0]
# print(model.intercept_)
# print(model.coef_)

# def myfunc(x):
#   return slope * x + intercept

# mymodel = list(map(myfunc, x))

# plt.scatter(x, y)
# plt.plot(x, mymodel)
# plt.show()



# def myfunc(x):
#   return slope2 * (x**2) + slope1 * x + intercept2

# mymodel = list(map(myfunc, x))

# plt.scatter(x, y)
# plt.plot(x, mymodel)
# plt.show()

# # Building the model
# X_mean = np.mean(x)
# Y_mean = np.mean(y)

# num = 0
# den = 0
# for i in range(len(x)):
#     num += (x[i] - X_mean)*(y[i] - Y_mean)
#     den += (x[i] - X_mean)**2
# m = num / den
# c = Y_mean - m*X_mean

# print (m, c)

# # Making predictions
# Y_pred = m*x + c

# plt.scatter(x, y) # actual
# # plt.scatter(X, Y_pred, color='red')
# plt.plot([min(x), max(x)], [min(Y_pred), max(Y_pred)], color='red') # predicted
# plt.show()



# x_ = PolynomialFeatures(degree=2, include_bias=False).fit_transform(x)
# model = LinearRegression(fit_intercept=False).fit(x_, y)
# intercept, coefficients = model.intercept_, model.coef_
# y_pred = model.predict(x_)


# plt.scatter(x, y)
# plt.plot(x, y_pred)
# plt.show()
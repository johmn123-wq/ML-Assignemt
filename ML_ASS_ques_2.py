import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error

iris_x=np.array([[85],[90],[93],[65],[87],[71],[98],[68],[84],[87]])
iris_y=np.array([[82],[88],[96],[72],[91],[80],[95],[72],[89],[84]])
print(len(iris_x))

iris_x_train=iris_x
iris_x_test=iris_x

iris_y_train=iris_y
iris_y_test=iris_y

model = linear_model.LinearRegression()
model.fit(iris_x_train,iris_y_train)

iris_y_predict = model.predict(iris_x_test)
print('SSE',mean_squared_error(iris_y_test,iris_y_predict))

print('Slope',model.coef_)
print("intercept",model.intercept_)


plt.scatter(iris_x_test,iris_y_test)
plt.plot(iris_x_test,iris_y_predict)
plt.show()

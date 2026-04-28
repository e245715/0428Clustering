import datasets
import regression

X, Y = datasets.load_linear_example1()
print(X)
print(X[0])
print(Y)

model = regression.LinearRegression()
model.fit(X,Y)
print(model.theta)

print(model.predict(X))

print(model.score(X,Y))
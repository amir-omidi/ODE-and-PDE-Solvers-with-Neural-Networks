from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
import numpy

"""
 ---- ANN Structure ----
"""

nn = MLPRegressor(hidden_layer_sizes=(10,10,10), activation='tanh',
                    solver='sgd', batch_size=500, learning_rate='adaptive',
                    learning_rate_init= 0.1, max_iter=10000, shuffle=False,
                    tol=0.00001, verbose=True, momentum=0.90)


"""
 ---- Defining Equation ----
"""

def equation(x):
    return x**3 + x**2 - x - 1

def get_equation_data(start,end,step_size):
    X = numpy.arange(start, end, step_size)
    X.shape = (len(X),1)
    y = numpy.array([equation(X[i]) for i in range(len(X))])
    y.shape = (len(y),1)
    return X,y

"""
 ---- Train neural network ----
"""

X,y = get_equation_data(-2,2,.1)
nn.fit(X,y)

# Predict
predictions = nn.predict(X)

"""
 ---- Plotting the Results ----
"""

plt.plot(predictions, label='approx')
plt.plot(y, label='exact')
plt.legend()
plt.show()

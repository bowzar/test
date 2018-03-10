import numpy
numpy.random.seed(1337)
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
import matplotlib.pyplot as plt

model = Sequential()

model.add(Dense(output_dim=1, input_dim=1))

model.compile(loss='mse',
              optimizer=SGD(lr=0.1))

X = numpy.linspace(-1, 1, 1000)
numpy.random.shuffle(X)
Y = 5 * X + 2 + numpy.random.normal(-.05, .05, (1000, ))

plt.scatter(X, Y)
plt.show()

x_train = X[:800]
y_train = Y[:800]

x_test = X[800:]
y_test = Y[800:]

for i in range(1000):
    model.train_on_batch(x_train, y_train)
    if(i % 50 == 0):
        W, b = model.layers[0].get_weights()
        print('Weights=', W, '\nbiases=', b)


print('\nTesting ------------')
cost = model.evaluate(x_test, y_test, batch_size=40)
print('test cost:', cost)
W, b = model.layers[0].get_weights()
print('Weights=', W, '\nbiases=', b)


y_pred = model.predict(x_test)
plt.scatter(x_test, y_test)
plt.plot(x_test, y_pred)
plt.show()

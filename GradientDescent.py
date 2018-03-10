import numpy
import matplotlib.pyplot as plt

numpy.random.seed(1337)

x_data = numpy.random.normal(-1, 1, (100,))
x2_data = numpy.random.normal(-1, 1, (100,))

y_data = x_data * 43 - x2_data * 6 + 23

b = 4.0
w = 2
w2 = 3
lr = 0.001
cnt = 10000

b_logs = [b]
w_logs = [w]
w2_logs = [w2]

# lr_b = 0
# lr_w = 0

for i in range(cnt):

    b_delta = 0.0
    w_delta = 0.0
    w2_delta = 0.0

    for n in range(len(x_data)):

        y_delta = y_data[n] - (w * x_data[n] + w2 * x2_data[n] + b)
        b_delta += 2.0 * y_delta * -1.0
        w_delta += 2.0 * y_delta * (-1 * x_data[n])
        w2_delta += 2.0 * y_delta * (-1 * x2_data[n])

    # lr_b += b_delta ** 2
    # lr_w += w_delta ** 2

    b -= lr * b_delta
    w -= lr * w_delta
    w2 -= lr * w2_delta

    b_logs.append(b)
    w_logs.append(w)
    w2_logs.append(w2)

print(w)
print(w2)
print(b)

plt.plot(b_logs, w_logs)
plt.show()

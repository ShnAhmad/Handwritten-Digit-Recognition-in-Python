from __future__ import division
import cv2
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

digits_data = load_digits()

digits = digits_data.data
targets = digits_data.target

# print(digits.shape)
# print(targets.shape)

a_digit = np.split(digits[1400], 8)
plt.imshow(a_digit, cmap='gray')

x_train, x_test, y_train, y_test = train_test_split(
    digits, targets, test_size=0.25)
# print(x_train.shape, x_test.shape)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def cost_function(h, y):
    return (-y * np.log(h) - (1 - y) * np.log(1 - h))


# print(y_train)

encode = OneHotEncoder(sparse=False)
Y = encode.fit_transform(y_train.reshape(-1, 1))
# print(Y)

X = x_train
m = len(Y)
epochs = 10
B = np.zeros([10, 64])
alpha = 0.1

for iteration in range(epochs):
    dB = np.zeros(B.shape)
    Loss = 0
    for j in range(X.shape[0]):
        x1 = X[j, :].reshape(64, 1)
        y1 = Y[j, :].reshape(10, 1)

        z1 = np.dot(B, x1)
        h = sigmoid(z1)

        db = (h - y1) * x1.T

        dB += db
        Loss += cost_function(h, y1)

    dB = dB / float(X.shape[0])
    Loss = Loss / float(X.shape[0])
    gradient = alpha * dB
    B = B - gradient

# print( Loss)


Digit_names = ["00.png", "01.png", "02.png", "03.png", "04.png", "05.png", "06.png", "07.png", "08.png",
               "09.png", "10.png", "11.png", "12.png", "13.png", "14.png", "15.png", "16.png", "17.png", "18.png", "19.png"]
temp = 0
for x in Digit_names:
    path = r"C:\\Users\\kHaN\\Desktop\\Digit_Recognition_code\\Digits\\" + x
    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    # print(gray)

    gray = cv2.resize(255 - gray, (8, 8))
    # print(gray)

    import matplotlib.pyplot as plt
    plt.imshow(gray, cmap='gray')

    g1 = gray.reshape(64, 1)
    # print(g1)

    z1 = np.dot(B, g1)
    h = sigmoid(z1)
    np.set_printoptions(suppress=True)
    if temp == 0:
        print("predicted values for 0")
    elif temp == 10:
        print("predicted values for 1")
    print(h.argmax(axis=0))
    temp = temp + 1

    # print(h)
    # print(z1)

import numpy as np
from keras.layers import Dense
from keras.models import Sequential

dataset = np.loadtxt("data.csv", delimiter=",")
X, Y = dataset[:, 0:8], dataset[:, 8]

model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X, Y, batch_size=10, verbose=2)
predictions = model.predict(X)

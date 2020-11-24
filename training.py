#code from https://machinelearningmastery.com/handwritten-digit-recognition-using-convolutional-neural-networks-python-keras/

#IMPORT LIBRARY
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils

# load (downloaded if needed) the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# reshape to be [samples][width][height][channels]
x_train = x_train.reshape((x_train.shape[0], 28, 28, 1)).astype('float32')
x_test = x_test.reshape((x_test.shape[0], 28, 28, 1)).astype('float32')

# normalize inputs from 0-255 to 0-1
# dinormalize agar nilai bias yang nanti dimiliki oleh data tidak terlalu besar range nya
x_train = x_train / 255
x_test = x_test / 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# one hot encode outputs, digunakan agar engine bisa mengetahui class atau ada berapa model yang akan direcognize
# Contoh : if we have labels A, B and C and we want to represent them in one-hot encoding then an A is [1, 0, 0], B is [0, 1, 0] and C is [0, 0, 1].
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
# dalam dataset MNIST akan ada 10 class untuk angka 0-9
# [1, 0, 0, 0, 0, 0, 0, 0, 0, 0] --> 0
# [0, 1, 0, 0, 0, 0, 0, 0, 0, 0] --> 1
# [0, 0, 1, 0, 0, 0, 0, 0, 0, 0] --> 2
# [0, 0, 0, 1, 0, 0, 0, 0, 0, 0] --> 3
# [0, 0, 0, 0, 1, 0, 0, 0, 0, 0] --> 4
# [0, 0, 0, 0, 0, 1, 0, 0, 0, 0] --> 5
# [0, 0, 0, 0, 0, 0, 1, 0, 0, 0] --> 6
# [0, 0, 0, 0, 0, 0, 0, 1, 0, 0] --> 7
# [0, 0, 0, 0, 0, 0, 0, 0, 1, 0] --> 8
# [0, 0, 0, 0, 0, 0, 0, 0, 0, 1] --> 9
num_classes = y_test.shape[1]

# CREATE MODEL
# Urutan Model Sequential dimulai dari yang paling awal diadd
model = Sequential()
# convolutional layer yang disebut Convolution2D
# layer memiliki 32 feature maps (neuron/dot), dengan size filter 5x5
# menerima input dengan [pixels][width][height] = 1, 28, 28
# activation function nya 'relu' --> f(z) bernilai nol ketika z kurang dari nol atau sama dengan nol,
# dan f(z) sama dengan z adalah ketika z lebih dari atau sama dengan nol.
model.add(Conv2D(32, (5, 5), input_shape=(1, 28, 28), activation='relu'))
# pooling layer that takes the max called MaxPooling2D. pool size nya 2×2.
model.add(MaxPooling2D(pool_size=(2, 2)))
# secara random exclude 20% neurons di layer untuk mengurangi overfitting
# Overfitting terjadi saat model menjadi sangat ahli dalam mengklasifikasikan atau memprediksi data yang ada dalam set training,
# tetapi tidak begitu baik dalam mengklasifikasikan data yang tidak ditraining.
model.add(Dropout(0.2))
# Flatten adalah mengubah data dari 2D array menjadi 1 dimensi array untuk dimasukkan ke lapisan berikutnya.
model.add(Flatten())
# fully connected layer with 128 neurons, activation function nya 'relu'
model.add(Dense(128, activation='relu'))
# Output layer memiliki 10 neuron untuk 10 classs dan fungsi aktivasi softmax untuk menghasilkan prediksi seperti probabilitas untuk setiap kelas
# Jadi nilai di output layer untuk setiap class berkisar antara 0-1
model.add(Dense(num_classes, activation='softmax'))
# Compile model using logarithmic loss and the ADAM gradient descent algorithm.
# categorical_crossentropy -> fungsi class yang menunjukan nilai float per fitur
# Adam adalah algoritme pengoptimalan yang dapat digunakan sebagai ganti dari prosedur stochastic gradient descent klasik untuk memperbarui weight network secara iteratif berdasarkan data training.
# matriks yang menjadi evaluasi matriks adalah bagian 'accuracy'
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# training data dengan 10 epochs, batch size nya 200
# batch size adalah jumlah sample yang diproses sebelum model diperbaharui
# epoch adalah jumlah yang menentukan berapa kali algoritma pembelajaran akan bekerja mengolah seluruh dataset training.
hist = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=200)
print("The model has successfully trained")
# save training model
model.save('mnist.h5')
print("Saving the model as mnist.h5")

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
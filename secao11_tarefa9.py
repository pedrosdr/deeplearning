from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, BatchNormalization, Dropout
from keras.regularizers import L2
from keras.initializers import HeNormal

gentrain = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=40,
    horizontal_flip=True,
    shear_range=0.2,
    height_shift_range=0.07,
    zoom_range=0.2
)

gentest = ImageDataGenerator(rescale=1.0/255.0)

base_train = gentrain.flow_from_directory(
    directory='dataset_personagens/training_set',
    target_size=(64, 64),
    batch_size=1,
    class_mode='binary'
)

base_test = gentrain.flow_from_directory(
    directory='dataset_personagens/test_set',
    target_size=(64, 64),
    batch_size=1,
    class_mode='binary'
)

# Estrutura da rede neural
model = Sequential()
model.add(Conv2D(16, (3,3), activation='relu', input_shape=(64, 64, 3), kernel_initializer=HeNormal(), kernel_regularizer=L2()))
model.add(BatchNormalization())

model.add(MaxPool2D())
model.add(Dropout(0.4))

model.add(MaxPool2D())
model.add(Dropout(0.4))

model.add(Flatten())

model.add(Dense(128, activation='relu', kernel_initializer=HeNormal()))
model.add(Dropout(0.2))
# model.add(Dense(128, activation='relu', kernel_initializer=HeNormal()))
# model.add(Dropout(0.2))

model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Treinando a Rede Neural
model.fit(x=base_train, steps_per_epoch=196/1, epochs=40, validation_steps=73/1, validation_data=base_test)

model.evaluate(base_test)

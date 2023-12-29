from keras.layers import Dense, MaxPool2D, Flatten, Conv2D, BatchNormalization, Dropout
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator, load_img
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# # Vizualizando as imagens
# image = Image.open('cats_dogs/training_set/cachorro/dog.6.jpg')
# image = np.array(image).astype('float32')
# image = 0.299 * image[:,:,0] + 0.587 * image[:,:,1] + 0.114 * image[:,:,2]
# plt.imshow(image, cmap='gray')

# Estrutura da rede neural
model = Sequential()
model.add(Conv2D(32, (3,3), activation='relu', input_shape=(64, 64, 3)))
model.add(BatchNormalization())
model.add(MaxPool2D())

model.add(Conv2D(32, (3,3), activation='relu', input_shape=(64, 64, 3)))
model.add(BatchNormalization())
model.add(MaxPool2D())

model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Augmentation
gen_train = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=7,
    horizontal_flip=True,
    shear_range=0.2,
    height_shift_range=0.07,
    zoom_range=0.2
)

gen_test = ImageDataGenerator(rescale=1.0/255.0)

base_train = gen_train.flow_from_directory(
    directory='cats_dogs/training_set', 
    batch_size=32,
    target_size=(64, 64),
    class_mode='binary'
)

base_test = gen_test.flow_from_directory(
    directory='cats_dogs/test_set',
    batch_size=32,
    target_size=(64, 64),
    class_mode='binary'
)

# Treinando a Rede Neural
model.fit(
    x=base_train,
    batch_size=32,
    epochs=15,
    validation_data=base_test,
    validation_batch_size=32
)

# # Salvando o classificador
# open('11_cats_dogs.json.json', 'w').write(model.to_json())
# model.save_weights('11_cats_dogs.h5')

# Classificando apenas uma imagem
imagem = load_img('cats_dogs/cachorro2.jpg', target_size=(64,64))
imagem = np.array(imagem).astype('float32')
imagem = imagem.reshape(1, 64, 64, 3)
imagem = imagem / 255

classe = model.predict(imagem)
print(base_train.class_indices)

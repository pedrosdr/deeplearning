import numpy as np
from keras.preprocessing.image import load_img
from keras.models import model_from_json
import matplotlib.pyplot as plt

# Loading model
model = model_from_json(open('10_mnist.json', 'r').read())
model.load_weights('10_mnist.h5')

# Loading image
img = np.array(load_img('digit.jpeg', target_size=(28, 28)))

# Converting image to grayscale
img = 0.299 * img[:,:,0] + 0.587 * img[:,:,1] + 0.114 * img[:,:,2]

# Reshaping the image
img = img.reshape(1, 28, 28, 1)

# Normalizing the image
img /= 255

# Inverting the colors
img = 1 - img

# Giving contrast
img = np.where(img < 0.3, 0, 1)

# Visualizing image
plt.imshow(img.reshape(28,28))

# Making predictions
y_new = [x.argmax() for x in model.predict(img)][0]

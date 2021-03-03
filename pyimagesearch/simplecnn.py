# import the necessary packages
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense

# our CNN architecture
class SimpleCNN:
	@staticmethod
	def build(width, height, depth, classes):
		# initialize the model along with the input shape
		model = Sequential()
		inputShape = (height, width, depth)
		chanDim = -1
		# first CONV => RELU => BN layer set
		model.add(Conv2D(32, (3, 3), strides=(2, 2), padding="same",
			input_shape=inputShape))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		# second CONV => RELU => BN layer set
		model.add(Conv2D(64, (3, 3), strides=(2, 2), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		# first (and only) set of FC => RELU layers
		model.add(Flatten())
		model.add(Dense(128))
		model.add(Activation("relu"))
		model.add(BatchNormalization())
		model.add(Dropout(0.5))
		# softmax classifier
		model.add(Dense(classes))
		model.add(Activation("softmax"))
		# return the constructed network architecture
		return model
"""    
The build
method of our SimpleCNN
class accepts four parameters:

    width
    : Width of the input images in our dataset
    height
    : Height of the input images in our dataset
    channels
    : Number of channels in the images
    classes
    : Total number of unique classes in the dataset

From there, we define a Sequential
network consisting of:

    A first set of CONV => RELU => BN
    layers. The CONV
    layer learns a total of 32 3×3 filters with 2×2 strided convolution to reduce volume size.
    A second set of CONV =>  RELU => BN
    layers. Same as above, but this time the CONV
    layer learns 64 filters.
    A set of dense/fully-connected layers. The output of which is our softmax classifier used for returning probabilities for each class label. 
"""

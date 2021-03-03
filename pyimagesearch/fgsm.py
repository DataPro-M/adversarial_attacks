# Implementing the Fast Gradient Sign Method with Keras and TensorFlow


# import the necessary packages
# the mean-squared error (MSE) loss function 
# you could also use any other appropriate loss function for the task, including categorical cross-entropy, binary cross-entropy, etc.
from tensorflow.keras.losses import MSE
import tensorflow as tf

"""
Inputs:

The model that we are trying to fool
The input image that we want to misclassify
The ground-truth class label of the input image
A small eps value that weights the gradient update — a small-ish value should be used here such that the gradient update is large enough to cause the input image to be misclassified but not so large that the human eye can tell the image has been manipulated
"""
def generate_image_adversary(model, image, label, eps=2 / 255.0):
	# cast the image
	image = tf.cast(image, tf.float32)
	# record our gradients
	with tf.GradientTape() as tape:
		# explicitly indicate that our image should be tacked for gradient updates
		tape.watch(image)
		# use our model to make predictions on the input image and then compute the loss
		pred = model(image)
		loss = MSE(label, pred)
	# calculate the gradients of loss with respect to the image, then compute the sign of the gradient
	gradient = tape.gradient(loss, image)
	# The output of this line of code is a vector filled with three values — either 1(positive), 0, or -1 (negative).
	signedGrad = tf.sign(gradient)
	# construct the image adversary
	adversary = (image + (signedGrad * eps)).numpy()
	# return the image adversary to the calling function
	return adversary
"""
* Taking the signed gradient and multiplying it by a small epsilon factor. The goal here is to make our gradient update large enough to misclassify the input image but not so large that the human eye can tell the image has been tampered.

* We then add this small delta value to our image, which ever so slightly changes the pixel intensity values in the image.

These pixel updates will be undetectable to the human eye, but according to our CNN, the image will appear vastly different, resulting in misclassification.
"""
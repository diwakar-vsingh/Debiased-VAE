import matplotlib.pyplot as plt
import PIL
import glob
from IPython import display as ipythondisplay
import time
import numpy as np
import os
import cv2
import tensorflow as tf

IM_SHAPE = (64, 64, 3)


def plot_sample(x, y, vae):
	plt.figure(figsize=(2, 1))
	plt.subplot(1, 2, 1)
	
	idx = np.where(y == 1)[0][0]
	plt.imshow(x[idx])
	plt.grid(False)
	
	plt.subplot(1, 2, 2)
	_, _, _, recon = vae(x)
	recon = np.clip(recon, 0, 1)
	plt.imshow(recon[idx])
	plt.grid(False)
	
	plt.show()


def get_test_faces():
	cwd = os.getcwd()
	images = {
		"LF": [],
		"LM": [],
		"DF": [],
		"DM": []
	}
	for key in images.keys():
		files = glob.glob(os.path.join(cwd, "faces", key, "*.png"))
		for file in sorted(files):
			image = cv2.resize(cv2.imread(file), (64, 64))[:, :, ::-1] / 255.
			images[key].append(image)
	
	return images["LF"], images["LM"], images["DF"], images["DM"]


def display_image(image_dir, epoch_no):
	return PIL.Image.open((image_dir + '/image_at_epoch_{:04d}.png').format(epoch_no))


def generate_and_save_images(model, epoch, test_input, image_dir):
	predictions = model.decode(test_input)
	plt.figure(figsize=(6, 6))
	for i in range(predictions.shape[0]):
		plt.subplot(4, 4, i + 1)
		plt.imshow(predictions[i, :, :, 0], cmap='gray')
		plt.axis('off')
	
	# tight_layout minimizes the overlap between 2 sub-plots
	plt.savefig((image_dir + '/image_at_epoch_{:04d}.png').format(epoch))
	plt.show()


class LossHistory:
	def __init__(self, smoothing_factor=0.0):
		self.alpha = smoothing_factor
		self.loss = []
	
	def append(self, value):
		self.loss.append(self.alpha * self.loss[-1] + (1 - self.alpha) * value if len(self.loss) > 0 else value)
	
	def get(self):
		return self.loss


class PeriodicPlotter:
	def __init__(self, sec, xlabel='', ylabel='', scale=None):
		
		self.xlabel = xlabel
		self.ylabel = ylabel
		self.sec = sec
		self.scale = scale
		
		self.tic = time.time()
	
	def plot(self, data):
		if time.time() - self.tic > self.sec:
			plt.cla()
			
			if self.scale is None:
				plt.plot(data)
			elif self.scale == 'semilogx':
				plt.semilogx(data)
			elif self.scale == 'semilogy':
				plt.semilogy(data)
			elif self.scale == 'loglog':
				plt.loglog(data)
			else:
				raise ValueError("unrecognized parameter scale {}".format(self.scale))
			
			plt.xlabel(self.xlabel)
			plt.ylabel(self.ylabel)
			plt.grid(True)
			ipythondisplay.clear_output(wait=True)
			ipythondisplay.display(plt.gcf())
			self.tic = time.time()
			
			
class standard_classifier:
	def __init__(self, dataset, batch_size, n_filters=12, num_epochs=6, learning_rate=1e-3):
		self.dataset = dataset
		self.batch_size = batch_size
		self.n_filters = n_filters
		self.model = self.encoder()
		self.num_epochs = num_epochs
		self.learning_rate = learning_rate
		self.optimizer = tf.keras.optimizers.Adam(learning_rate)  # define our optimizer
		self.loss_history = LossHistory(smoothing_factor=0.99)  # to record loss evolution
		self.plotter = PeriodicPlotter(sec=2, scale='semilogy')
	
	def encoder(self):
		inputs = tf.keras.Input(shape=(64, 64, 3))
		x = tf.keras.layers.Conv2D(filters=1 * self.n_filters, kernel_size=5, strides=2, padding='same', activation='relu')(inputs)
		x = tf.keras.layers.BatchNormalization()(x)
		
		x = tf.keras.layers.Conv2D(filters=2 * self.n_filters, kernel_size=5, strides=2, padding='same', activation='relu')(x)
		x = tf.keras.layers.BatchNormalization()(x)
		
		x = tf.keras.layers.Conv2D(filters=4 * self.n_filters, kernel_size=3, strides=2, padding='same', activation='relu')(x)
		x = tf.keras.layers.BatchNormalization()(x)
		
		x = tf.keras.layers.Conv2D(filters=6 * self.n_filters, kernel_size=3, strides=2, padding='same', activation='relu')(x)
		x = tf.keras.layers.BatchNormalization()(x)
		
		x = tf.keras.layers.Flatten()(x)
		x = tf.keras.layers.Dense(512, activation='relu')(x)
		x = tf.keras.layers.Dense(1, activation='relu')(x)
		
		model = tf.keras.Model(inputs, x)
		return model
	
	@tf.function
	def standard_train_step(self, x, y):
		with tf.GradientTape() as tape:
			# feed the images into the model
			logits = self.model(x)
			
			# Compute the loss
			loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits)
		
		# Backpropagation
		grads = tape.gradient(loss, self.model.trainable_variables)
		self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
		return loss
	
	def training(self):
		# The training loop!
		for epoch in range(self.num_epochs):
			for idx in range(self.dataset.get_train_size() // self.batch_size):
				# Grab a batch of training data and propagate through the network
				x, y = self.dataset.get_batch(self.batch_size)
				loss = self.standard_train_step(x, y)
				
				# Record the loss and plot the evolution of the loss as a function of training
				self.loss_history.append(loss.numpy().mean())
				
	def evaluate(self):
		test_faces = get_test_faces()
		keys = ["Light Female", "Light Male", "Dark Female", "Dark Male"]
		
		standard_classifier_logits = [self.model(np.array(x, dtype=np.float32)) for x in test_faces]
		standard_classifier_probs = tf.squeeze(tf.sigmoid(standard_classifier_logits))
		
		# Plot the prediction accuracies per demographic
		xx = range(len(keys))
		yy = standard_classifier_probs.numpy().mean(1)
		plt.bar(xx, yy)
		plt.xticks(xx, keys)
		plt.title("Standard classifier predictions")
import tensorflow as tf
import numpy as np
import functools


class DVAE(tf.keras.Model):
	def __init__(self, latent_dim, n_filters=12):
		super(DVAE, self).__init__()
		self.latent_dim = latent_dim
		self.n_filters = n_filters
		self.inference_net = self.encoder(2 * self.latent_dim + 1)
		self.generative_net = self.decoder()
		self.inference_net.summary()
		self.generative_net.summary()
	
	def encoder(self, n_output):
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
		x = tf.keras.layers.Dense(n_output, activation='relu')(x)
		
		model = tf.keras.Model(inputs, x)
		return model
	
	def decoder(self):
		inputs = tf.keras.Input(shape=self.latent_dim)
		x = tf.keras.layers.Dense(units=4 * 4 * 6 * self.n_filters, activation=tf.nn.relu)(inputs)  # 4x4 feature maps with 6N occurances
		x = tf.keras.layers.Reshape(target_shape=(4, 4, 6 * self.n_filters))(x)
		
		# Up-scaling convolutions
		x = tf.keras.layers.Conv2DTranspose(filters=4 * self.n_filters, kernel_size=3, strides=2, padding='same', activation='relu')(x)
		x = tf.keras.layers.Conv2DTranspose(filters=2 * self.n_filters, kernel_size=3, strides=2, padding='same', activation='relu')(x)
		x = tf.keras.layers.Conv2DTranspose(filters=1 * self.n_filters, kernel_size=5, strides=2, padding='same', activation='relu')(x)
		x = tf.keras.layers.Conv2DTranspose(filters=3, kernel_size=5, strides=2, padding='same', activation='relu')(x)
		model = tf.keras.Model(inputs, x)
		return model
	
	def encode(self, x):
		"""
		Function to feed images into encoder and encode the latent space
		"""
		# Encoder output
		encoder_output = self.inference_net(x)
		
		# Classification prediction
		y_logit = tf.expand_dims(encoder_output[:, 0], -1)
		# y_logit = encoder_output[:, 0]
		
		# Latent variable distribution parameters
		z_mean = encoder_output[:, 1:self.latent_dim + 1]
		z_log_var = encoder_output[:, self.latent_dim + 1:]
		
		return y_logit, z_mean, z_log_var
	
	@tf.function
	def sample(self):
		"""
		Reparameterization trick by sampling from an isotropic unit Gaussian.
		Arguments: z_mean, z_log_var:
		Returns: z [sampled latent vector]
		"""
		eps = tf.random.normal(shape=(100, self.latent_dim))
		return self.decode(eps, apply_sigmoid=True)
	
	def reparameterize(self, z_mean, z_log_var):
		"""
		VAE reparameterization: given a mean and logsigma, sample latent variables
		"""
		batch, latent_dim = z_mean.shape
		epsilon = tf.random.normal(shape=(batch, latent_dim))
		z = z_mean + tf.exp(0.5 * z_log_var) * epsilon
		return z
	
	def decode(self, z, apply_sigmoid=False):
		"""
		Use the decoder to output the reconstructed image
		"""
		reconstruction = self.generative_net(z)
		if apply_sigmoid:
			probs = tf.sigmoid(reconstruction)
			return probs
		return reconstruction
	
	def get_latent_mu(self, x, batch_size=1024):
		"""
		Function to return the means for an input image batch
		"""
		N = x.shape[0]
		mu = np.zeros((N, self.latent_dim))
		
		for start_idx in range(0, N, batch_size):
			end_idx = min(start_idx + batch_size, N + 1)
			batch = (x[start_idx:end_idx]).astype(np.float32) / 255.0
			_, batch_mu, _ = self.encode(batch)
			mu[start_idx:end_idx] = batch_mu
		return mu
	
	def get_training_sample_probabilities(self, x, bins=10, smoothing_fn=0.001):
		"""
		Function that recomputes the sampling probabilities for images within
		a batch based on how they distribute across the training data
		"""
		print("Recomputing the sampling probabilities")
		
		# Run the input batch and get the latent variable means
		mu = self.get_latent_mu(x)
		
		# Sampling probabilities for the images
		training_sampling_p = np.zeros(mu.shape[0])
		
		# Consider the distribution for each latent variable
		for i in range(self.latent_dim):
			latent_distribution = mu[:, i]
			
			# Generate a histogram of the latent distribution
			hist_density, bin_edges = np.histogram(latent_distribution, density=True, bins=bins)
			
			# Find in which latent bin does every data sample falls in
			bin_edges[0] = -float('inf')
			bin_edges[-1] = float('inf')
			
			# Find which bins in the latent distribution every data sample falls in to
			bin_idx = np.digitize(latent_distribution, bin_edges)
			
			# Smooth the density function
			hist_smoothed_density = hist_density + smoothing_fn
			hist_smoothed_density = hist_smoothed_density / np.sum(hist_smoothed_density)
			
			# Invert the density function
			p = 1.0 / (hist_smoothed_density[bin_idx - 1])
			
			# Normalize all probabilities
			p /= np.sum(p)
			
			# Update sampling probabilities by considering whether the newly computed
			# p is greater than the existing sampling probabilities
			training_sampling_p = np.maximum(p, training_sampling_p)
		
		# Final normalization
		training_sampling_p /= np.sum(training_sampling_p)
		
		return training_sampling_p

	def predict(self, x):
		# Predict face or not face logit for given input x
		y_logit, _, _ = self.encode(x)
		return y_logit
	
	def call(self, x):
		# Encode input to a prediction and latent space
		y_logit, z_mean, z_logsigma = self.encode(x)
		
		# Reparameterization
		z = self.reparameterize(z_mean, z_logsigma)
		
		# Reconstruction
		recon = self.decode(z)
		return y_logit, z_mean, z_logsigma, recon

import tensorflow as tf


class calculate_losses:
	def __init__(self, model, strategy, optimizer, global_batch, kl_weight=0.001):
		self.model = model
		self.strategy = strategy
		self.optimizer = optimizer
		self.kl_weight = kl_weight
		self.batch_size = global_batch
	
	def debiasing_loss_fn(self, x, y):
		# Obtain VAE loss
		vae_loss = self.vae_loss(x)
		
		y_logit, _, _, _ = self.model(x)
		
		# Define the classification loss using sigmoid_cross_entropy
		classification_loss = tf.squeeze(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=y_logit))
		
		# Use the training data labels to create variable face_indicator:
		# indicator that reflects which training data are images of faces
		face_indicator = tf.squeeze(tf.cast(tf.equal(y, 1), tf.float32))
		
		# Total DV-VAE loss
		# total_loss = tf.reduce_mean(classification_loss + face_indicator * vae_loss)
		total_loss = classification_loss + face_indicator * vae_loss
		
		total_loss = tf.nn.compute_average_loss(total_loss, global_batch_size=self.batch_size)
		classification_loss = tf.nn.compute_average_loss(classification_loss, global_batch_size=self.batch_size)
		return total_loss, classification_loss
	
	def vae_loss(self, x):
		"""
		Function to calculate VAE loss given:
		"""
		# Compute z, z_mean and z_log_var
		_, z_mean, z_log_var, x_logit = self.model(x)
		
		# Reshape x and x_logit
		x_shape = tf.shape(x)
		x = tf.reshape(x, [self.batch_size, x_shape[1] * x_shape[2] * x_shape[3]])
		
		x_logit_shape = tf.shape(x_logit)
		x_logit = tf.reshape(x_logit, [self.batch_size, x_logit_shape[1] * x_logit_shape[2] * x_logit_shape[3]])
		
		# KL divergence regularization loss.
		kl_loss = 0.5 * tf.reduce_sum(tf.exp(z_log_var) + tf.square(z_mean) - 1.0 - z_log_var, axis=1)
		
		# Reconstruction loss:
		reconstruction_loss = tf.keras.losses.MSE(x, x_logit)
		
		# Total VAE loss
		vae_loss = self.kl_weight * kl_loss + reconstruction_loss
		return vae_loss
	
	def compute_apply_gradients(self, x, y):
		with tf.GradientTape() as tape:
			# Compute loss
			loss, class_loss = self.debiasing_loss_fn(x, y)
		
		# Compute the gradients
		grads = tape.gradient(loss, self.model.trainable_variables)
		
		# Apply gradients to variables
		self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
		return loss
	
	@tf.function
	def distributed_train_step(self, dataset_inputs, dataset_label):
		# `run` replicates the provided computation and runs it with the distributed input.
		per_replica_losses = self.strategy.run(self.compute_apply_gradients, args=(dataset_inputs, dataset_label))
		return self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

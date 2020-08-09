import datetime

from DVAE import *
from input_fn import *
from loss_fn import *
from utility import *
from tqdm import tqdm

# `tf.distribute.MirroredStrategy` constructor
strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

# Hyper-parameters
learning_rate = 5e-4
latent_dim = 100
epochs = 6
BATCH_SIZE_PER_REPLICA = 512
path_to_training_data = "train_face.h5"

# Run standard classifier model for benchmarking purpose
dataset = TrainingDatasetLoader(data_path=path_to_training_data)
standard_classifier_model = standard_classifier(dataset, batch_size=BATCH_SIZE_PER_REPLICA, num_epochs=epochs, learning_rate=learning_rate)

# Train the model
print("Training Standard Classifier model ...")
standard_classifier_model.training()
standard_classifier_model.plotter.plot(standard_classifier_model.loss_history.get())
print("Standard Classifier model trained")

# Evaluate the standard CNN on the test data
standard_classifier_model.evaluate()

if __name__ == "__main__":
	
	with strategy.scope():
		# Instantiate dataset
		GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync
		
		# Instantiate VAE model
		model = DVAE(latent_dim)
		
		# Define optimizer
		optimizer = tf.keras.optimizers.Adam(learning_rate)
		
		# Create a checkpoint directory to store the checkpoints.
		checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
		manager = tf.train.CheckpointManager(checkpoint, './tf_ckpts', max_to_keep=3)
		checkpoint_dir = './training_checkpoints'
		checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
		
		# Set up summary writers to write the summaries to disk in different logs directory
		current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
		train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
		train_summary_writer = tf.summary.create_file_writer(train_log_dir)
		
		# get all the training faces from data loader
		all_faces = dataset.get_all_train_faces()
		
		# Utily function to plot loss vs iteration
		loss_history = LossHistory(smoothing_factor=0.99)  # to record loss evolution
		plotter = PeriodicPlotter(sec=2, scale='semilogy')
		
		# Instantiate loss function
		loss_fn = calculate_losses(model, strategy, optimizer, GLOBAL_BATCH_SIZE, kl_weight=0.001)
		
		if hasattr(tqdm, '_instances'): tqdm._instances.clear()  # clear if it exists
		for epoch in range(1, epochs + 1):
			
			# IPython.display.clear_output(wait=False)
			print("Starting epoch {}/{}".format(epoch, epochs))
			
			# Recompute data sampling probabilities for data debiasing
			p_faces = model.get_training_sample_probabilities(all_faces)
			
			start_time = time.time()
			# get a batch of training data and compute the training step
			for j in tqdm(range(dataset.get_train_size() // GLOBAL_BATCH_SIZE)):
				# load a batch of data
				(x, y) = dataset.get_batch(GLOBAL_BATCH_SIZE, p_pos=p_faces)
				
				# TRAIN LOOP
				total_loss = 0.0
				num_batches = 0
				total_loss += loss_fn.distributed_train_step(x, y)
				# Record the loss and plot the evolution of the loss as a function of training
				loss_history.append(total_loss)
				num_batches += 1
				
				with train_summary_writer.as_default():
					tf.summary.scalar("loss", total_loss/num_batches, step=(epoch - 1) * (dataset.get_train_size() // GLOBAL_BATCH_SIZE) + j)
				
				if j % 500 == 0:
					plot_sample(x, y, model)
			
			train_loss = total_loss / num_batches
			end_time = time.time()
			
			template = 'Epoch {}, Loss: {}, Time elapsed for current epoch: {}'
			print(template.format(epoch, train_loss, end_time - start_time))
			plotter.plot(loss_history.get())
			
			if epoch % 2 == 0:
				checkpoint.save(checkpoint_prefix)
	
	# Save weights to a TensorFlow Checkpoint file
	model.save_weights('./weights/my_model')
	
	test_faces = get_test_faces()
	keys = ["Light Female", "Light Male", "Dark Female", "Dark Male"]
	dbvae_logits = [model.predict(np.array(x, dtype=np.float32)) for x in test_faces]
	dbvae_probs = tf.squeeze(tf.sigmoid(dbvae_logits))
	
	xx = np.arange(len(keys))
	standard_classifier_logits = [standard_classifier_model.model(np.array(x, dtype=np.float32)) for x in test_faces]
	standard_classifier_probs = tf.squeeze(tf.sigmoid(standard_classifier_logits))
	
	plt.bar(xx, standard_classifier_probs.numpy().mean(1), width=0.2, label="Standard CNN")
	plt.bar(xx + 0.2, dbvae_probs.numpy().mean(1), width=0.2, label="DB-VAE")
	plt.xticks(xx, keys)
	plt.title("Network predictions on test dataset")
	plt.ylabel("Probability")
	plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
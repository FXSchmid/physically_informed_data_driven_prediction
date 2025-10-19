import tensorflow as tf
from tensorflow import keras
import datetime
import numpy as np
from keras.callbacks import ModelCheckpoint, CallbackList

class Config:
    def __init__(self, model):
        self.model = model
        self.start_time = datetime.datetime.now().strftime('%Y.%m.%d - %H:%M:%S')
        self.train_size = 91
        self.vali_size = 113
        self.events = 420
        self.event = 10
        self.batch_size = 4 
        self.epochs = 30
        self.steps = 910 
        self.steps_vali = 210
        self.n_timesteps = 25
        self.learning_rate = 0.0001
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)
        self.loss_fn = keras.losses.MeanSquaredError()
        self.cp_path = r"..."  #path to the loaction for saving the evaluation arrays
        self.cp_path_epochen = r"..."  #path to the loaction
        self.filename_weights = '/weights.{epoch:02d}.hdf5'
        self.checkpoint_filepath = r"..."   + self.filename_weights # path to the loaction for the saved weights
        self.logs = {}
        self.callback_cp = ModelCheckpoint(filepath=self.checkpoint_filepath,save_weights_only=True, monitor='val_loss',mode='min', save_best_only=True)
        self.callbacks = CallbackList(self.callback_cp, model=self.model)
        self.train_acc = keras.metrics.RootMeanSquaredError()
        self.val_acc = keras.metrics.RootMeanSquaredError()
        self.log_training = []
        self.log_validation = []

    def export_settings_after_epoche(self,epoch):
        train_acc_results_np = np.array(self.log_training)
        val_acc_results_np = np.array(self.log_validation)
        np.save(self.cp_path_epochen + f'/train_results_epoche_{epoch}.npy', train_acc_results_np)
        np.save(self.cp_path_epochen + f'/val_results_{epoch}.npy', val_acc_results_np)

        filename = self.cp_path_epochen + '/' + f'training_settings_epoche_{epoch}.txt'
        total_trainable_weights = sum([tf.size(w).numpy() for w in self.model.trainable_weights])
        total_trainable_weights = format(total_trainable_weights, ',')
        settings = {
            "model name": self.model.__class__.__name__,
            "start training time": self.start_time,
            "end training time": datetime.datetime.now().strftime('%Y.%m.%d - %H:%M:%S'),
            "batchsize": self.batch_size,
            "epochs": self.epochs,
            "epoch_now": epoch+1,
            "RMSE Train loss of epoch": train_acc_results_np[-1],
            "RMSE Val loss of epoch": val_acc_results_np[-1],
            "steps": self.steps,
            "optimizer": self.optimizer.__class__.__name__,
            "learning rate": self.learning_rate,
            "loss function": self.loss_fn.name,
            "number trainable weights": total_trainable_weights,
            "best train epoch": self.log_training.index(min(self.log_training)),
            "minimum train error": min(self.log_training),
            "best vali epoch": self.log_validation.index(min(self.log_validation)),
            "minimum validation error": min(self.log_validation)
        }
        with open(filename, 'w') as f:
            f.write("Training Settings after epoch\n")
            for key, value in settings.items():
                f.write(f"{key}: {value}\n")

    def export_settings(self):
        train_acc_results_np = np.array(self.log_training)
        val_acc_results_np = np.array(self.log_validation)
        np.save(self.cp_path + '/train_results.npy', train_acc_results_np)
        np.save(self.cp_path + '/val_results.npy', val_acc_results_np)

        filename = self.cp_path + '/' + 'training_settings.txt'
        dot_img_file = self.cp_path + '/' + 'architecture_pic.png'
        tf.keras.utils.plot_model(self.model.build_graph(), to_file=dot_img_file, show_shapes=True)
        total_trainable_weights = sum([tf.size(w).numpy() for w in self.model.trainable_weights])
        total_trainable_weights = format(total_trainable_weights, ',')
        settings = {
            "model name": self.model.__class__.__name__,
            "start training time": self.start_time,
            "end training time": datetime.datetime.now().strftime('%Y.%m.%d - %H:%M:%S'),
            "batchsize": self.batch_size,
            "epochs": self.epochs,
            "steps": self.steps,
            "optimizer": self.optimizer.__class__.__name__,
            "learning rate": self.learning_rate,
            "loss function": self.loss_fn.name,
            "number trainable weights": total_trainable_weights,
            "best train epoch": self.log_training.index(min(self.log_training)),
            "minimum train error": min(self.log_training),
            "best vali epoch": self.log_validation.index(min(self.log_validation)),
            "minimum validation error": min(self.log_validation)
        }
        with open(filename, 'w') as f:
            f.write("Training Settings\n")
            for key, value in settings.items():
                f.write(f"{key}: {value}\n")
    
    def round_tensor(self, tensor, decimals):
        factor = tf.constant(10.0 ** decimals, dtype=tensor.dtype)
        return tf.round(tensor * factor) / factor
    
    def c_s_get_shapes(self, x_water, x_rain, y_true, GIS): 
        elevation = GIS[:,:,:,1]
        elevation = np.reshape(elevation,(self.batch_size,256,256,1))
        elevation_center = elevation[:,64:192,64:192,:] # center part
        elevation_center_nn = elevation_center * 548.4481

        n = GIS[:,:,:,2]
        n = np.reshape(n,(self.batch_size,256,256,1))
        n_center = n[:,64:192,64:192,:]
        n_center_nn = n_center * 0.1

        y_previous = x_water[:,1,:,:,:]
        y_center = y_previous[:,64:192,64:192,:] #center part
        y_true_sliced = y_true[:, :, :, :-1]
        y_true_previous = np.concatenate([y_center, y_true_sliced], axis = -1)

        #x_rain
        x_rain_ts = x_rain[ :, :, :, ::2] # in mm/hr for each timestep
        x_rain_m = x_rain_ts / (6000*100) # for recalculation in m
        x_rain_m = np.tile(x_rain_m, (1, 8, 8, 1))

        return x_rain_m, y_true_previous, elevation_center_nn, n_center_nn
    
    def custom_sequence_physics(self, x_water, x_rain, y_true, y_pred, GIS):
        dx = 5 #m
        dy = 5 #m
        dt = 600 #s   

        x_rain_m, y_true_previous, elevation_center, n_center = self.c_s_get_shapes(x_water, x_rain, y_true, GIS)

        rain = tf.cast(x_rain_m, tf.float32)
        ht_y_true = tf.cast(y_true_previous, tf.float32)
        ht1_y_pred = tf.cast(y_pred, tf.float32)
        elevation = tf.cast(elevation_center, tf.float32)
        n_center = tf.cast(n_center, tf.float32)
        ht1_y_true = tf.cast(y_true, tf.float32)

        # x
        hx1 = tf.roll(ht_y_true, shift=-1, axis=1)
        hx2 = tf.roll(ht_y_true, shift=1, axis=1)
        #y
        hy1 = tf.roll(ht_y_true, shift=-1, axis=2)
        hy2 = tf.roll(ht_y_true, shift=1, axis=2)

        # slope 
        elevation = tf.where(elevation == 0, 256.6746, elevation)
        n_center = tf.where(n_center == 0, 0.025, n_center)
        s_x = (tf.roll(elevation, shift=-1, axis=1) - tf.roll(elevation, shift=1, axis=1)) / (2 * dx)
        s_y = (tf.roll(elevation, shift=-1, axis=2) - tf.roll(elevation, shift=1, axis=2)) / (2 * dy)

        # q
        dh = (ht1_y_pred - ht_y_true)
        dq_dx = (tf.sqrt(tf.abs(s_x))/(n_center + 1e-10) * (tf.pow(hx1, 5/3) - tf.pow(hx2, 5/3)))/(2*dx)
        dq_dy = (tf.sqrt(tf.abs(s_y))/(n_center + 1e-10) * (tf.pow(hy1, 5/3) - tf.pow(hy2, 5/3)))/(2*dy)
        loss_q = tf.square(dh + dq_dx*dt + dq_dy*dt - rain*dt)

        round_loss_q = self.round_tensor(loss_q,2)
        sum_loss_q = tf.reduce_sum(round_loss_q) 
        return sum_loss_q
    
    def mse_loss(self,ht1_y_true, ht1_y_pred): 
        mse_loss = tf.keras.losses.MeanSquaredError()(ht1_y_true, ht1_y_pred)
        return mse_loss 

    
    def custom_sequence_MSE_physics(self, x_water, x_rain, y_true, y_pred, GIS):
        
        sum_loss_q = self.custom_sequence_physics(x_water, x_rain, y_true, y_pred, GIS)
        ht1_y_true = tf.cast(y_true, tf.float32)
        ht1_y_pred = tf.cast(y_pred, tf.float32)
        mse_loss = self.mse_loss(ht1_y_true, ht1_y_pred)

        overall_loss = mse_loss + sum_loss_q

        return overall_loss

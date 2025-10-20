import tensorflow as tf
import os
from architecture.initial_network import InitialNetwork
import logging
import config
import numpy as np
import h5py
import itertools

class Training_sequ:
    def __init__(self, data_path, channels=None):
        self.setup_gpu()
        logging.disable(logging.WARNING)
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

        if not isinstance(data_path, str):
            raise ValueError("data_path must be a string path to the folder containing processed_data files")

        self.data_path = data_path

        # allow passing channels or keep previous default
        self.channels = channels if channels is not None else [32, 64, 128, 256, 512]
        self.model = InitialNetwork(self.channels)
        self.config_C = config.Config(self.model)
        self.batch_size = self.config_C.batch_size
        self.epochs = self.config_C.epochs
        self.steps = self.config_C.steps
        self.steps_vali = self.config_C.steps_vali
        self.train_size = self.config_C.train_size
        self.vali_size = self.config_C.vali_size
        self.event = self.config_C.event

        # prepare data
        self.e_r_a_train_complete_1, self.e_r_a_vali_complete_1 = self.prepare_data()

        print(f"Training_sequ initialized using data_path={self.data_path}")

    def setup_gpu(self):
        #### tensorflow error ###
        print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        if len(physical_devices) > 0:
            tf.config.experimental.set_visible_devices(physical_devices[0], 'GPU')
            tf.config.experimental.set_memory_growth(physical_devices[0], True)

    def prepare_network(self):
        # Try to build the model with example inputs if available, otherwise skip
        try:
            self.model(self.x_gis, self.x_water, self.x_rain, training=False)
        except Exception:
            pass

        # Load weights only if a valid path attribute was set
        if hasattr(self, 'weights_path') and isinstance(self.weights_path, str) and os.path.isfile(self.weights_path):
            try:
                self.model.load_weights(self.weights_path)
                print('weights loaded!')
            except Exception as e:
                print(f'Could not load weights from {self.weights_path}: {e}')

    def prepare_data(self): # this function you need to repalce probably
        # expected files under processed_data inside the provided data_path
        processed_dir = os.path.join(self.data_path, 'processed_data')
        index_areas_file = os.path.join(processed_dir, 'areas_train_multi.txt')
        event_rotation_file = os.path.join(processed_dir, 'event_rotation.txt')

        if not os.path.isdir(processed_dir):
            raise FileNotFoundError(f"processed_data directory not found at {processed_dir}")

        index_areas_ds = np.genfromtxt(index_areas_file, dtype='int')
        event_rotation = np.genfromtxt(event_rotation_file, dtype='int')

        np.random.shuffle(index_areas_ds)
        index_areas_train_full = index_areas_ds[:self.train_size]
        index_areas_vali_full = index_areas_ds[self.train_size:self.vali_size]
        # with array rotation
        e_r_a_train_complete = [[*row,value] for row, value in itertools.product(event_rotation,index_areas_train_full)]
        e_r_a_vali_complete = [[*row,value] for row, value in itertools.product(event_rotation,index_areas_vali_full)]
        # Shuffle
        np.random.shuffle(e_r_a_train_complete)
        np.random.shuffle(e_r_a_vali_complete)

        return e_r_a_train_complete, e_r_a_vali_complete
    
    def generate_train_data(self, data):
        rotation_area_hdf = np.genfromtxt(os.path.join(self.data_path, 'processed_data', 'rotation_area_hdf.txt'), dtype='int')
                     
        row = []
        index = []
        index_hdf = []
        np.random.shuffle(data)

        if len(data) > self.batch_size:
            data_event_rotation_area = data[:self.batch_size]
            data_rest = data[self.batch_size::]
        else:
            data_event_rotation_area = data
            data_rest = []
        for i in range(len(data_event_rotation_area)):
            matches = (rotation_area_hdf[:, 0] == data_event_rotation_area[i][1]) & (rotation_area_hdf[:, 1] == data_event_rotation_area[i][2])
            row_indices = np.where(matches)[0]
            row.append(row_indices)
            index.append(i)

        indexed_row = list(enumerate(row))
        sorted_indexed_row = sorted(indexed_row, key=lambda x: x[1])
        sorted_row = [x[1] for x in sorted_indexed_row] 
        sorted_indices = [x[0] for x in sorted_indexed_row]

        x_gis = np.zeros((len(index), 256, 256, 8), dtype=float)
        x_rain = np.zeros((len(index), 16, 16, 22), dtype=float)
        y_water = np.zeros((len(index), 128, 128, 11), dtype=float)
        x_water = np.zeros((len(index), 3, 256, 256, 1), dtype=float) 
        x_water_assumed = np.zeros((len(index),256,256,1), dtype=float) 
        hdf_file = os.path.join(self.data_path, 'processed_data', 'Dataset.h5')
        if not os.path.isfile(hdf_file):
            raise FileNotFoundError(f"HDF dataset not found at {hdf_file}")

        with h5py.File(hdf_file,'r') as hdf:
            for value_array,indices in zip(sorted_row,sorted_indices):
                value = value_array
                x_gis[indices] = hdf['x_gis'][value, :, :, :]
                x_rain[indices] = hdf['x_rain'][data_event_rotation_area[indices][0], :, :, :]
                x_water[indices] = hdf['x_water'][data_event_rotation_area[indices][0], value,:, :, :, :]
                y_water[indices] = hdf['y'][data_event_rotation_area[indices][0], value, :, :, :]
                x_water_assumed[indices] = hdf['x_water_assumed'][data_event_rotation_area[indices][0], value, :, :, :] # precalcualed CAM attention
        for j in range(len(data_event_rotation_area)):
            x_water[j, 0, :, :, :] = x_water_assumed[j, :, :, :]
        return x_gis, x_rain, x_water, y_water, data_rest, data_event_rotation_area #event_timestep_hdf, data_event_timestep_area, data_rest, row, row_sorted #x_gis, x_rain, x_water, y_water, data_rest, data_event_area_timestep self.event_timestep_hdf, self.data_event_area_timestep, self.data_rest, self.row, self.row_sorted

    def shuffle_data(self, x_gis, x_rain, x_water, y_water):
        indices = np.arange(len(x_gis))
        np.random.shuffle(indices)
        x_gis_shuffled = x_gis[indices]
        x_water_shuffled = x_water[indices]
        x_rain_shuffled = x_rain[indices]
        y_water_shuffled = y_water[indices]
        return x_gis_shuffled, x_rain_shuffled, x_water_shuffled, y_water_shuffled

    def training_step(self, x_batch, y_batch):
        input_gis = x_batch[0]
        input_water = x_batch[1]
        input_rain = x_batch[2]
        with tf.GradientTape() as tape:
            y_pred = self.model(input_gis, input_water, input_rain, training=True)
            # custom_sequence_MSE_physics signature: (x_water, x_rain, y_true, y_pred, GIS)
            loss = self.config_C.custom_sequence_MSE_physics(input_water, input_rain, y_batch, y_pred, input_gis)
        gradients = tape.gradient(loss, self.model.trainable_weights)
        self.config_C.optimizer.apply_gradients(zip(gradients, self.model.trainable_weights))
        self.config_C.train_acc.update_state(y_batch, y_pred)
        return loss

    def validation_step(self, x_batch_val, y_batch_val):
        input_gis = x_batch_val[0]
        input_water = x_batch_val[1]
        input_rain = x_batch_val[2]
        y_pred_val = self.model(input_gis, input_water, input_rain, training=False)
        self.config_C.val_acc.update_state(y_batch_val, y_pred_val)
        loss_val = self.config_C.custom_sequence_MSE_physics(input_water, input_rain, y_batch_val, y_pred_val, input_gis)
        return loss_val
    
    def train_sequ(self):    
            train_config = self.config_C
            ### train
            for epoch in range( self.epochs):
                print(f"\nStart of training epoch {epoch + 1}")
                train = self.e_r_a_train_complete_1
                vali = self.e_r_a_vali_complete_1

                for i in range(self.steps):#
                    x_gis, x_rain, x_water, y_water, data_rest, data_event_rotation_area = self.generate_train_data(train)
                    x_gis_shuffled, x_rain_shuffled, x_water_shuffled, y_water_shuffled = self.shuffle_data(x_gis, x_rain, x_water, y_water)

                    train_data = [x_gis_shuffled, x_water_shuffled, x_rain_shuffled]
                    loss = self.training_step(train_data, y_water_shuffled)

                    batch_rmse = train_config.train_acc.result().numpy()
                    print(f"Batch {i+1}/{self.steps} - RMSE: {batch_rmse}")
                    train = data_rest

                for k in range(self.steps_vali):

                    x_gis, x_rain, x_water, y_water, data_rest, data_event_rotation_area = self.generate_train_data(vali)
                    x_gis_shuffled, x_rain_shuffled, x_water_shuffled, y_water_shuffled = self.shuffle_data(x_gis, x_rain, x_water, y_water)
                    
                    vali_data = [x_gis_shuffled, x_water_shuffled, x_rain_shuffled]
                    # run validation step to compute batch loss and update val_acc
                    loss_val = self.validation_step(vali_data, y_water_shuffled)

                    batch_val_rmse = train_config.val_acc.result().numpy()
                    print(f"Batch {k + 1}/{self.steps_vali} - Validation RMSE: {batch_val_rmse} - batch_loss: {loss_val}")
                    vali = data_rest

                train_loss = train_config.train_acc.result().numpy()
                print(f"RMSE Loss over epoch {train_loss}")
                train_config.log_training.append(train_loss)
                train_config.train_acc.reset_states()

                val_loss = train_config.val_acc.result().numpy()
                print(f"RMSE Val loss of epoch {val_loss}")
                train_config.log_validation.append(val_loss)


                train_config.logs["val_loss"] = train_config.val_acc.result()
                train_config.callbacks.on_epoch_end(epoch=epoch, logs=train_config.logs)
                train_config.val_acc.reset_states()

                train_config.export_settings_after_epoche(epoch)

            train_config.export_settings()
            self.model.save('Forecast_multi_seq.h5')

            return print('finished training')    


if __name__ == '__main__':


    print('start of training')
    # Example usage: pass the base folder that contains 'processed_data' directory
    # By default try to use a 'data' folder at the repository root if exists
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    default_data_path = os.path.join(repo_root, 'data')
    if os.path.isdir(default_data_path):
        data_path = default_data_path
    else:
        data_path = os.getcwd()

    print(f"Using data_path={data_path}")
    trainer = Training_sequ(data_path=data_path, channels=[32, 64, 128, 256, 512])
    trainer.train_sequ()
    print('finish')

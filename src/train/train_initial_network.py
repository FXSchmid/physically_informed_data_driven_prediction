import tensorflow as tf
import os
from architecture.initial_networks import InitialNetwork
import logging
import config
import numpy as np
import h5py
import itertools

class Trainer:
    def __init__(self, data_path, channels=None):

        self.setup_gpu()
        logging.disable(logging.WARNING)
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

        if not isinstance(data_path, str):
            raise ValueError("data_path must be a string path to the folder containing processed_data")
        self.data_path = data_path

        # allow optional channels param
        self.channels = channels if channels is not None else [32, 64, 128, 256, 512]
        self.model = InitialNetwork(self.channels)
        self.config_C = config.Config(self.model)
        self.batch_size = self.config_C.batch_size
        self.epochs = self.config_C.epochs
        self.steps = self.config_C.steps
        self.steps_vali = self.config_C.steps_vali
        self.train_size = self.config_C.train_size
        self.vali_size = self.config_C.vali_size
        self.events = self.config_C.events
        self.event = self.config_C.event
        self.n_timesteps = self.config_C.n_timesteps

        # prepare data
        self.e_t_a_train_complete, self.e_t_a_vali_complete = self.prepare_data_balanced()
        # generate one initial batch lazily
        self.x_gis, self.x_rain, self.x_water, self.y_water, self.data_rest, self.data_event_timestep_area = self.generate_train_data_balanced(self.e_t_a_train_complete)
        print("Trainer initialized")

    def setup_gpu(self):
        #### tensorflow error ###
        print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        if len(physical_devices) > 0:
            tf.config.experimental.set_visible_devices(physical_devices[0], 'GPU')
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
    
    def training_step(self, x_batch, y_batch):
        input_gis = x_batch[0]
        input_water = x_batch[1]
        input_rain = x_batch[2]
        with tf.GradientTape() as tape:
            y_pred = self.model(input_gis, input_water, input_rain, training=True)
            loss = self.config_C.custom_sequence_MSE_physics(y_batch, y_pred, input_gis, input_water)   #loss_fn(y_batch, y_pred)   y_true, y_pred, GIS, X_water
        gradients = tape.gradient(loss, self.model.trainable_weights)
        self.config_C.optimizer.apply_gradients(zip(gradients, self.model.trainable_weights))
        self.config_C.train_acc.update_state(y_batch, y_pred)
        return loss

    def validation_step(self, x_batch_val, y_batch_val):
        input_gis = x_batch_val[0]
        input_water = x_batch_val[1]
        input_rain = x_batch_val[2]
        y_pred_val = self.model(input_gis, input_water, input_rain, training=False)
        loss_val = self.config_C.custom_sequence_MSE_physics(y_batch_val, y_pred_val, input_gis, input_water)
        self.config_C.val_acc.update_state(y_batch_val, y_pred_val)
        return loss_val
#______________________________________

    def prepare_data_balanced(self):
        processed_dir = os.path.join(self.data_path, 'processed_data')
        if not os.path.isdir(processed_dir):
            raise FileNotFoundError(f"processed_data directory not found at {processed_dir}")

        index_areas_ds = np.genfromtxt(os.path.join(processed_dir, 'areas_train_multi.txt'), dtype='int')
        event_timestep_selected = np.genfromtxt(os.path.join(processed_dir, 'event_timestep_selected.txt'), dtype='int')
 
        np.random.shuffle(index_areas_ds)
        index_areas_train_full = index_areas_ds[:self.train_size]
        index_areas_vali_full = index_areas_ds[self.train_size:self.vali_size]

        e_t_a_train_complete = [[*row,value] for row, value in itertools.product(event_timestep_selected,index_areas_train_full)]
        e_t_a_vali_complete = [[*row,value] for row, value in itertools.product(event_timestep_selected,index_areas_vali_full)]

        np.random.shuffle(e_t_a_train_complete)
        np.random.shuffle(e_t_a_vali_complete)

        return e_t_a_train_complete, e_t_a_vali_complete

    def generate_train_data_balanced(self, data):
        processed_dir = os.path.join(self.data_path, 'processed_data')
        event_timestep_hdf = np.genfromtxt(os.path.join(processed_dir, 'event_timestep_hdf.txt'), dtype='int')
        
        row = []
        index = []
        np.random.shuffle(data)

        if len(data) > self.batch_size:#8
            data_event_timestep_area = data[:self.batch_size]#8
            data_rest = data[self.batch_size::]#8
        else:
            data_event_timestep_area = data
            data_rest = []

        for i in range(len(data_event_timestep_area)):
            matches = (event_timestep_hdf[:, 0] == data_event_timestep_area[i][0]) & (event_timestep_hdf[:, 1] == data_event_timestep_area[i][1])
            row_indices = np.where(matches)[0]
            row.append(row_indices)
            index.append(i)
        row_sorted = sorted(row)

        x_gis = np.zeros((len(index), 256, 256, 13), dtype=float)
        x_rain = np.zeros((len(index), 16, 16, 2), dtype=float)
        y_water = np.zeros((len(index), 128, 128, 1), dtype=float)
        x_water = np.zeros((len(index), 2, 256, 256, 1), dtype=float)

        hdf_file = os.path.join(processed_dir, 'Dataset.h5')
        if not os.path.isfile(hdf_file):
            raise FileNotFoundError(f"HDF dataset not found at {hdf_file}")

        with h5py.File(hdf_file,'r') as hdf:
            k=0
            for value_array in row_sorted:
                if k > len(index):
                    break
                value = value_array[0]
                x_gis[k] = hdf['x_gis'][data_event_timestep_area[k][2], :, :, :]
                x_rain[k] = hdf['x_rain'][value, data_event_timestep_area[k][2], :, :, :]
                x_water[k] = hdf['x_water'][value, data_event_timestep_area[k][2],:, :, :, :]
                y_water[k] = hdf['y'][value, data_event_timestep_area[k][2], :, :, :]
                k += 1

        for j in range(len(data_event_timestep_area)):
                if data_event_timestep_area[j][1]==2:
                    x_water_assumed = np.zeros((1,256,256,1), dtype=float)
                    with h5py.File(hdf_file,'r') as hdf:
                        x_water_assumed[0] = hdf['x_water_assumed'][data_event_timestep_area[j][0], data_event_timestep_area[j][2],:, :, :]
                    x_water[j, 0, :, :, :] = x_water_assumed
                if data_event_timestep_area[j][1]==3:
                    x_water_assumed = np.zeros((1,256,256,1), dtype=float)
                    with h5py.File(hdf_file,'r') as hdf:
                        x_water_assumed[0] = hdf['x_water_assumed'][data_event_timestep_area[j][0], data_event_timestep_area[j][2],:, :, :]
                    x_water[j, 1, :, :, :] = x_water_assumed

        return x_gis, x_rain, x_water, y_water, data_rest, data_event_timestep_area 
    
    def shuffle_data(self, x_gis, x_rain, x_water, y_water):
        indices = np.arange(len(x_gis)) 
        np.random.shuffle(indices)
        x_gis_shuffled = x_gis[indices]
        x_water_shuffled = x_water[indices]
        x_rain_shuffled = x_rain[indices]
        y_water_shuffled = y_water[indices]
        return x_gis_shuffled, x_rain_shuffled, x_water_shuffled, y_water_shuffled
    
    def train_balanced(self):    
       
        train_config = self.config_C

        ### train
        for epoch in range(self.epochs):
            print(f"\nStart of training epoch {epoch + 1}")
            train = self.e_t_a_train_complete
            vali = self.e_t_a_vali_complete 

            for i in range(self.steps):
                x_gis, x_rain, x_water, y_water, data_rest, data_event_area_timestep = self.generate_train_data_balanced(train)

                x_gis_shuffled, x_rain_shuffled, x_water_shuffled, y_water_shuffled = self.shuffle_data(x_gis, x_rain, x_water, y_water)

                train_data = [x_gis_shuffled, x_water_shuffled, x_rain_shuffled]
                loss = self.training_step(train_data, y_water_shuffled)
                print(loss)
                train = data_rest

            for k in range(self.steps_vali):

                x_gis, x_rain, x_water, y_water, data_rest, data_event_timestep_area = self.generate_train_data_balanced(vali)
                x_gis_shuffled, x_rain_shuffled, x_water_shuffled, y_water_shuffled = self.shuffle_data(x_gis, x_rain, x_water, y_water)
                
                vali_data = [x_gis_shuffled, x_water_shuffled, x_rain_shuffled]     
                loss_val = self.validation_step(vali_data, y_water_shuffled)
                print(loss_val)
                vali = data_rest

            train_loss = train_config.train_acc.result().numpy()
            print(f"Loss over epoch {train_loss}")
            train_config.log_training.append(train_loss)
            train_config.train_acc.reset_states()

            val_loss = train_config.val_acc.result().numpy()
            print(f"Val loss of epoch {val_loss}")
            train_config.log_validation.append(val_loss)


            train_config.logs["val_loss"] = train_config.val_acc.result()
            train_config.callbacks.on_epoch_end(epoch=epoch, logs=train_config.logs)
            train_config.val_acc.reset_states()

        train_config.export_settings()
        self.model.save('Forecast_initial.h5')

        return print('finished training')


if __name__ == '__main__':

    print('start of training')
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    default_data_path = os.path.join(repo_root, 'data')
    if os.path.isdir(default_data_path):
        data_path = default_data_path
    else:
        data_path = os.getcwd()

    print(f"Using data_path={data_path}")
    t = Trainer(data_path=data_path, channels=[32, 64, 128, 256, 512])
    t.train_balanced()

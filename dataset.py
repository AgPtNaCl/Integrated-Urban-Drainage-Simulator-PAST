from torch.utils.data import Dataset
from Data_preprocess import *

class Trainingset(Dataset):
    def __init__(self, config, num_nodes):
        self.training_set_path = config['training_params']['file_path']
        self.training_set_size = config['training_params']['num_samples']
        rainfall_train, inflow_train, self.labels = load_data_from_hdf5(self.training_set_path, self.training_set_size, num_nodes)
        self.rainfall = rainfall_train.unsqueeze(2)
        self.inflow = inflow_train.unsqueeze(2)
        self.lateral_inflow, _ = get_lateral_inflow(self.training_set_path, self.training_set_size, num_nodes)
    
    def __len__(self):
        return len(self.rainfall)

    def __getitem__(self, index):
        rainfall = self.rainfall[index]
        inflow = self.inflow[index]
        lateral_inflow = self.lateral_inflow[index]
        label = self.labels[index]
        return rainfall, inflow, lateral_inflow, label

class Validationset(Dataset):
    def __init__(self, config, num_nodes):
        self.validation_set_size = config['validation_params']['valid_num']
        self.validation_set_path = config['validation_params']['validation_set_path']
        rainfall_valid, inflow_valid, self.labels = load_data_from_hdf5(self.validation_set_path, self.validation_set_size, num_nodes)
        self.rainfall = rainfall_valid.unsqueeze(2)
        self.inflow = inflow_valid.unsqueeze(2)
    
    def __len__(self):
        return len(self.rainfall)

    def __getitem__(self, index):
        rainfall = self.rainfall[index]
        inflow = self.inflow[index]
        label = self.labels[index]
        return rainfall, inflow, label

class Trainingset_DR(Dataset):
    def __init__(self, config, num_nodes):
        self.training_set_path = config['training_params']['file_path']
        self.training_set_size = config['training_params']['num_samples']
        rainfall_train, inflow_train, self.labels = load_data_from_hdf5(self.training_set_path, self.training_set_size, num_nodes)
        self.forcing_data = Merge_forcing_data(rainfall_train, inflow_train, num_nodes)
        
    
    def __len__(self):
        return len(self.forcing_data)

    def __getitem__(self, index):
        forcing_data = self.forcing_data[index]
        label = self.labels[index]
        return forcing_data, label

class Validationset_DR(Dataset):
    def __init__(self, config, num_nodes):
        self.validation_set_size = config['validation_params']['valid_num']
        self.validation_set_path = config['validation_params']['validation_set_path']
        rainfall_valid, inflow_valid, self.labels = load_data_from_hdf5(self.validation_set_path, self.validation_set_size, num_nodes)
        self.forcing_data = Merge_forcing_data(rainfall_valid, inflow_valid, num_nodes)
    
    def __len__(self):
        return len(self.forcing_data)

    def __getitem__(self, index):
        forcing_data = self.forcing_data[index]
        label = self.labels[index]
        return forcing_data, label
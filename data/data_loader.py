import torch
from data import NiftiDataset
from data import transformations as custom_tf

def load_datasets(exp_config, batch_size, path_train, path_test, path_val, tf = custom_tf.ToTensor()):
    train_loader, test_loader, val_loader = get_data_loaders_nifti(exp_config, batch_size, path_train, path_test, path_val, TF = tf)

    return train_loader, test_loader, val_loader

def get_data_loaders_nifti(exp_config, batch_size, path_train, path_test, path_val, TF):
    train_loader, test_loader, val_loader = None, None, None

    if path_train != '':
        ds_train = NiftiDataset.NiftiDataset(path_train, TF)
        train_loader = torch.utils.data.DataLoader(ds_train, batch_size = batch_size, shuffle = exp_config.shuffle_train, num_workers = 4)
    
    if path_test != '':
        ds_test = NiftiDataset.NiftiDataset(path_test, custom_tf.ToTensor())
        test_loader = torch.utils.data.DataLoader(ds_test, batch_size = batch_size, shuffle = exp_config.shuffle_test, num_workers = 4)
    
    if path_val != '':
        ds_validation = NiftiDataset.NiftiDataset(path_val, custom_tf.ToTensor())
        val_loader = torch.utils.data.DataLoader(ds_validation, batch_size = batch_size, shuffle = exp_config.shuffle_validation, num_workers = 4)
    
    return train_loader, test_loader, val_loader
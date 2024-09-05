from importlib.machinery import SourceFileLoader
from data import data_loader
import argparse
from utils_train import *
import random
import numpy as np

def main(exp_config):
    
    print(exp_config.experiment_name)

    # =====================
    # Define network architecture
    # =====================    
    model = exp_config.model
    model.cuda()
    
    # =========================
    # Load source dataset
    # =========================
    source_train_loader, source_test_loader, source_val_loader = data_loader.load_datasets(exp_config,\
                                                                            exp_config.batch_size,\
                                                                          exp_config.path_train,\
                                                                          exp_config.path_test,\
                                                                          exp_config.path_validation,\
                                                                          exp_config.tf)
    # =========================
    # Train segmentation network
    # =========================
    train_segmentation_network(exp_config, \
                               model, \
                               source_train_loader, \
                               source_val_loader, \
                               exp_config.path_to_save_pretrained_models)
    
if __name__ == '__main__':
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    
    parser = argparse.ArgumentParser(description="Script for training")
    parser.add_argument("--config_path", type=str, help="Path to experiment config file")
    args = parser.parse_args()

    config_file = args.config_path
    config_module = config_file.split('/')[-1].rstrip('.py')
        
    exp_config = SourceFileLoader(config_module, config_file).load_module() # exp_config stores configurations in the given config file under experiments folder.
    main(exp_config=exp_config)


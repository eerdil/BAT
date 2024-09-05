from itertools import count
import torch
import torch.optim as optim
import os
import torch.nn.functional as F

# ===============
# Function for training segmentation network
# ===============
def train_segmentation_network(exp_config, model, loader_train, loader_val, path_to_save_pretrained_models):

    os.makedirs(path_to_save_pretrained_models, exist_ok=True)
    
    optimizer = optim.Adam(model.parameters(), lr = exp_config.learning_rate)

    criterion_mse = exp_config.criterion_mse

    best_loss_val = 0
    for epoch in range(exp_config.number_of_epoch):
        
        # =========
        # TRAIN
        # =========
        model.train() # Switch on training mode            
        running_loss_train  = 0.0

        counter = 0
        for data, target in loader_train:

            data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            pred_logits, _ = model(data)

            loss_train = criterion_mse(pred_logits, target)
            
            loss_train.backward()
            optimizer.step()

            running_loss_train += loss_train.item()
            
            counter += 1

        running_loss_train = running_loss_train / counter
        
        # =========
        # VAL
        # =========
        model.eval() # Switch on evaluation mode
        
        running_loss_val = 0.0
        counter = 0
        for data, target in loader_val:

            data, target, norm_params = data.cuda(), target.cuda(), norm_params.cuda()
            pred_logits, _ = model(data)

            loss_val = criterion_mse(pred_logits, target)

            running_loss_val += loss_val.item()
            
            counter += 1

        running_loss_val = running_loss_val / counter

        # =========
        # SAVE Model
        # =========
        print('epoch:%d - loss_tr: %.5f loss_val: %.5f  - saved' %
                  (epoch, running_loss_train, running_loss_val))
        
        if epoch == 0 or best_loss_val > running_loss_val:
            best_loss_val = running_loss_val
            torch.save(model.state_dict(), ('%s/model.pth')%(path_to_save_pretrained_models))

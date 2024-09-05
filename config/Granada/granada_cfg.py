from data import transformations as custom_tf
from torchvision import transforms as pytorch_tf
import torch.nn as nn
from models.attention_unet import AttentionUNet
# =======================
# Experiment name / input dataset
# =======================
dataset = 'Granada'
experiment_name = '%s_experiment'%(dataset)

# ==================
# Hyperparameters
# ==================
batch_size = 32
number_of_epoch = 1000
learning_rate = 1e-3
n0 = 16 # number of filters in the first conv layer of AttentionUNet
model = AttentionUNet(n0)
criterion_mse = nn.MSELoss()

# ==============
# Paths - !REPLACE WITH THE CORRECT PATH OF YOUR DATA
# ==============
path_train = '/%s/train'%(dataset)
path_validation = '/%s/val'%(dataset)
path_test = '/%s/test'%(dataset)

shuffle_validation = False
shuffle_train = True
shuffle_test = False

path_to_save_pretrained_models = './pre_trained/%s/%s'%(dataset, experiment_name)

# ======================================
# Transformations for data augmentation
# ======================================
tf = pytorch_tf.Compose([custom_tf.RandomApply(\
                                pytorch_tf.Compose([\
                                    custom_tf.RandomApply(custom_tf.Translate([10, 10]), p=0.5),\
                                    custom_tf.RandomApply(custom_tf.Rotate([-30, 30]), p=0.5),\
                                    custom_tf.RandomApply(pytorch_tf.Compose([\
                                                                custom_tf.RandomCrop([360, 240]),\
                                                                custom_tf.Resize([480, 320])]),\
                                        p=0.5),\
                                    custom_tf.RandomApply(custom_tf.Flip(), p=0.5),\
                                    ]),\
                                p=0.50),
                             custom_tf.ToTensor()])
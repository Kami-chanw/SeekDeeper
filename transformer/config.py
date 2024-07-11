import torch

# GPU device setting
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# model parameter setting
batch_size = 128
max_len = 256
d_model = 512
n_layer = 6
n_head = 8
ffn_hidden = 2048
dropout = 0.1

# optimizer parameter setting
init_lr = 1e-5
epochs = 5
clip = 1.0
weight_decay = 5e-4

# path
checkpoint_dir = "./checkpoints"

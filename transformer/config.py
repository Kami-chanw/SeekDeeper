import torch

# GPU device setting
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# model parameter setting (Transformer base)
max_len = 256
d_model = 512
n_layer = 6
n_head = 8
ffn_hidden = 2048
dropout = 0.1

# training setting
batch_size = 128
update_freq = 1 # in Multi30k with batch_size = 128, it should be 6, see README.md
epochs = 20
eps_ls = 0.1 # eps for label smoothing
warmup_step = 4000
clip = 1

# optimizer parameter setting
lr = 0.0007
betas = (0.9, 0.98)
adam_eps = 1e-6

# path
checkpoint_dir = "./checkpoints"
dataset_dir = "./datasets/Multi30k"

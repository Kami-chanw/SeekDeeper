# model parameter setting (Transformer base)
max_len = 512
d_model = 512
n_layer = 6
n_head = 8
ffn_hidden = 2048
dropout = 0.1

# training setting
batch_size = 32
update_freq = 16
epochs = 20
eps_ls = 0.1  # eps for label smoothing
warmup_step = 4000
clip = 1

# optimizer parameter setting
betas = (0.9, 0.98)
adam_eps = 1e-9

# path
checkpoint_dir = "./checkpoints"
dataset_dir = "./datasets/IWSLT17"

# inference
num_beams = 3
top_k = 30
top_p = 0.7
temperature = 1.0
length_penalty = 0.7
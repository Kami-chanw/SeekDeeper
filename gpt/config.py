from dataclasses import dataclass

# model parameter setting
max_len = 512
d_model = 768
n_layer = 12
n_head = 12
vocab_size = 40478
dropout = 0.1

@dataclass
class PretrainConfig:
    n_epoch = 100
    batch_size = 64
    lr = 2.5e-4
    warmup_step = 2000


@dataclass
class TrainConfig:
    n_epoch = 3
    batch_size = 32
    lr = 6.25e-5
    loss_weight = 0.5
    warmup_ratio = 0.2 / 100

# path
checkpoint_dir = "./checkpoints"
bookcorpus_dir = "./datasets/bookcorpus"


# optimizer args
adamwr_w = 1e-2
clip = 1

# inference
num_beams = 3
top_k = 30
top_p = 0.7
temperature = 1.0
length_penalty = 0.7
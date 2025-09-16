import torch

# model parameters
Input_size = 10001
num_classes = 3
n_ind_pan = 50
n_ind_pan_model = n_ind_pan // num_classes
n_ind_max = n_ind_pan_model * num_classes
n_embd = 36
n_embd_model = n_embd + 2
min_pos_prob = 0.06
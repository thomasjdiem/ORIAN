from shared import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import exp


OHE_values = torch.arange(3**4).reshape(3, 3, 3, 3) # SOI value, ref value, class we are in, labeled class of ref
OHE_values[:, :, 2] = torch.flip(OHE_values[:, :, 0], dims=(2,))
OHE_values[2] = torch.flip(OHE_values[0], dims=(0,))
unique_elements, inverse_indices = torch.unique(OHE_values, return_inverse=True)
OHE_values = torch.arange(len(unique_elements))[inverse_indices]
OHE_values = OHE_values.flatten()
assert n_embd == OHE_values.max() + 1

# how to index:
# labeled class + class we are in * 3 + ref_value * 9 + SOI value * 27
#    var1              var2                  var3           var4
    
# if var3_1 == var3_2, var4_1 == var4_2, var2_1 == 2 - var2_2 != 1, var1_1 == 2 - var1_2
    # then values should be same
# else they should be different

class ParallelAttentionSetModel(nn.Module):
    def __init__(self, input_dim, output_dim, num_models):
        super(ParallelAttentionSetModel, self).__init__()

        self.attention_weights = nn.Parameter(torch.randn(input_dim, num_models))
        self.fc_weight = nn.Parameter(torch.randn(num_models, input_dim, output_dim))
        self.fc_bias = nn.Parameter(torch.randn(num_models, 1, output_dim))

    def forward(self, x):
        attention_scores = torch.matmul(x, self.attention_weights)  # Compute attention scores
        attention_weights = torch.softmax(attention_scores, dim=-3)
        weighted_sum = (x.unsqueeze(-2) * attention_weights.unsqueeze(-1)).sum(dim=-3)

        out = torch.matmul(weighted_sum.unsqueeze(-2), self.fc_weight) + self.fc_bias

        return out.squeeze(-2)

    
class NeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        self.input_size = input_size
        self.hidden0 = n_embd_model

        self.hidden1 = 150
        self.hidden2 = 75

        self.hidden2_1 = 25

        self.hidden3 = 50

        hidden1c = num_classes * n_ind_pan_model
        hidden1d = num_classes * n_ind_pan_model

        self.linear0 = nn.Linear(n_embd_model, self.hidden0, bias=False)
        self.ln0 = nn.LayerNorm(self.hidden0)

        self.linear1 = nn.Linear(input_size * self.hidden0, self.hidden1)

        self.ln1 = nn.LayerNorm(self.hidden1)
        self.linear2 = nn.Linear(self.hidden1, self.hidden2) 
        self.ln2 = nn.LayerNorm(self.hidden2)
        self.linear3 = nn.Linear(self.hidden2, self.hidden2_1)
        self.ln3 = nn.LayerNorm(num_classes * n_ind_pan_model)

        self.linear4 = nn.Linear((num_classes * n_ind_pan_model), self.hidden3)
        self.ln4 = nn.LayerNorm(self.hidden3)
        self.linear5 = nn.Linear(self.hidden3, 1)

        self.linear6 = nn.Linear(self.hidden2_1, 1)

        self.linear1c = nn.Linear((num_classes * n_ind_pan_model),hidden1c)
        self.ln1c = nn.LayerNorm(num_classes * n_ind_pan_model)

        self.linear1d = nn.Linear(hidden1c, hidden1d)
        self.ln1d = nn.LayerNorm(num_classes * n_ind_pan_model)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.parallel_attention_set_model1 = ParallelAttentionSetModel(self.hidden1, self.hidden1, hidden1c)
        self.parallel_attention_set_model2 = ParallelAttentionSetModel(self.hidden2, self.hidden2, hidden1d)
        self.parallel_attention_set_model3 = ParallelAttentionSetModel(self.hidden2_1, self.hidden2_1, self.hidden3)
        self.parallel_attention_set_model4 = ParallelAttentionSetModel(self.hidden2_1, self.hidden2_1, 1)


    def fuse_layers(self, device):
        # linear0_reshaped = torch.zeros((input_size * n_embd_model, input_size * self.hidden0), device=device)
        # for i in range(input_size):
        #     linear0_reshaped[i * n_embd_model: (i + 1) * n_embd_model, i * self.hidden0: (i + 1) * self.hidden0] = self.linear0.weight.t()

        # self.linear0_reshaped = linear0_reshaped

        linear0w = self.linear0.weight.t()
        linear1w = self.linear1.weight.t()
        w = torch.zeros((self.input_size * n_embd_model, self.hidden1), device=device)

        for i in range(self.input_size):
            w[i * n_embd_model: (i + 1) * n_embd_model, :] = (linear0w @ linear1w[i * self.hidden0: (i + 1) * self.hidden0, :])

        self.linear01_fused = nn.Linear(self.input_size * n_embd_model, self.hidden1).to(device)
        with torch.no_grad():
            self.linear01_fused.weight.copy_(w.t())
            self.linear01_fused.bias.copy_(self.linear1.bias)

        self.__delattr__('linear0')
        self.__delattr__('linear1')

            
    def forward(self, ref_sim):

        # ref_sim = self.linear0(ref_sim) # batch, num_classes, n_ind_max, input_size, hidden0
        ref_sim = ref_sim.reshape(*ref_sim.shape[:3], -1) # batch, num_classes, n_ind_max, input_size * self.hidden0
        # ref_sim = self.linear1(ref_sim) # batch, num_classes, n_ind_max, self.hidden1

        ref_sim = self.linear01_fused(ref_sim)

        if ref_sim.shape[-2] < n_ind_max:
            padding = self.linear01_fused.bias.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(*ref_sim.shape[:2], n_ind_max - ref_sim.shape[-2], -1)
            ref_sim = torch.cat((ref_sim, padding), dim=-2)
            
        ref_sim = self.ln1(ref_sim)
        ref_sim = self.relu(ref_sim)

        ref_sim = self.parallel_attention_set_model1(ref_sim) # batch, num_classes, n_ind_max, self.hidden1

        ref_sim = self.linear2(ref_sim) # batch, num_classes, n_ind_max, self.hidden2
        ref_sim = self.ln2(ref_sim)
        ref_sim = self.relu(ref_sim)

        ref_sim = self.parallel_attention_set_model2(ref_sim) # batch, num_classes, n_ind_max, self.hidden2


        ref_sim = self.linear3(ref_sim) # batch, num_classes, n_ind_max, hidden_2_1
        ref_sim = self.sigmoid(ref_sim) # batch, num_classes, num_classes * n_ind_pan_model

        ref_sim = self.parallel_attention_set_model3(ref_sim) # batch, num_classes, self.hidden3, hidden_2_1
        ref_sim = self.relu(ref_sim)

        ref_sim = self.parallel_attention_set_model4(ref_sim) 
        ref_sim = ref_sim.squeeze(-2) # batch, num_classes, hidden_2_1
        ref_sim = self.linear6(ref_sim).squeeze(-1) # batch, num_classes

        return ref_sim
    
class InputProcessor:
    def __init__(self, input_size, device):
        self.input_size = input_size
        self.min_pos_prob_tensor = torch.tensor([[min_pos_prob]], device=device)
        label_id = torch.arange(num_classes, device=device).unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        cl = torch.arange(num_classes, device=device).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        self.labels_cl = label_id + 3 * cl
        
        self.OHE_values = OHE_values.to(device)
        self.device = device


    def set_lambda_f(self, num_generations, admixture_proportion, population_size):
        admixture_proportion = torch.tensor(sorted([admixture_proportion, 1 - admixture_proportion]), dtype=torch.float, device=self.device)
        self.lambda_f = 2 * (num_generations - 1) * admixture_proportion
        self.lambda_f *= exp(-0.5 * (num_generations - 1) / population_size)
        self.lambda_f_iszero = (num_generations == 0)

    def get_inputs(self, SOI, refs, labels, positions):

        # SOI             # batch, input_size
        # positions       # batch, input_size
        # refs            # batch, n_ind_max, input_size
        # labels          # batch, n_ind_max, input_size, num_classes

        positions = (positions - positions[:,self.input_size // 2].unsqueeze(-1)).abs()
        
        if self.lambda_f_iszero:
            pos_probs = (~positions.isinf()).float().unsqueeze(-1).expand(-1, -1, 2)
        else:
            pos_probs = (-self.lambda_f * positions.unsqueeze(-1)).exp()

        min_pos_prob_tensor = self.min_pos_prob_tensor.expand(pos_probs.shape[0], -1)
        position_cutoff_left = torch.searchsorted(pos_probs[:, :self.input_size // 2, 0].contiguous(), min_pos_prob_tensor, side="right")
        position_cutoff_right = self.input_size - torch.searchsorted(pos_probs[:, self.input_size // 2:, 0].flip(-1).contiguous(), min_pos_prob_tensor, side="right")

        # pos_probs[:, :, 0] = 0
        # pos_probs[:, :, 1] = positions
        SOI = SOI.unsqueeze(1).unsqueeze(1).unsqueeze(-1) # batch, 1, 1, input_size, 1

        refs = refs.unsqueeze(1).unsqueeze(-1) # batch, 1, num_refs, input_size, 1

        labels = labels.unsqueeze(1).expand(-1, 3, -1, -1, -1)

        # labels = F.softmax(28 * labels.log(), dim=-1)
        # cl # 1, num_classes, 1, 1, 1
        # label_id # 1, 1, 1, 1, 3
        # labels   # batch, num_refs, input_size, 3

        ref_sim = self.labels_cl + (num_classes ** 2) * refs + (num_classes ** 3) * SOI
        ref_sim = self.OHE_values[ref_sim.long()]

        out = torch.zeros((*ref_sim.shape[:-1], n_embd), dtype=torch.float, device=self.device)
        out.scatter_add_(dim=-1, index=ref_sim, src=labels)

        # out = F.one_hot(out.argmax(dim=-1), num_classes=36).float()

        pos_probs = pos_probs.unsqueeze(1).unsqueeze(1).expand(-1, num_classes, out.shape[2], -1, -1)
        out = torch.cat((out, pos_probs), dim=-1)

        for i in range(out.shape[0]):
            out[i, :, :, :position_cutoff_left[i]] = 0
            out[i, :, :, position_cutoff_right[i]:] = 0

        return out
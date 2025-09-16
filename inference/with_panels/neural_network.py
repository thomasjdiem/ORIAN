from shared import *
import torch
import torch.nn as nn


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

        self.embedding0 = nn.Embedding(n_embd, self.hidden0)
        self.linear0_partial = nn.Linear(n_embd_model - n_embd, self.hidden0, bias=False)
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

        self.saved_ref_sim = None
        # self.Transition = Transition
    
    def forward(self, ref_sim, idxs=None, next_idx=None):

        ref_sim, position_input, position_cutoff_left, position_cutoff_right = ref_sim

        ref_sim = self.embedding0(ref_sim) 

        if idxs is not None:
            if self.saved_ref_sim is not None:
                ref_sim = torch.cat((self.saved_ref_sim, ref_sim), dim=2)
            if next_idx is None:
                self.saved_ref_sim = None
            else:
                self.saved_ref_sim = ref_sim[:, :, ref_sim.shape[2] + min(0, next_idx - idxs[-1] - self.input_size):]
            idxs = idxs - idxs[0]
            ref_sim = torch.stack([ref_sim[:, :, idx:idx + self.input_size] for idx in idxs], dim=0)


        ref_sim += self.linear0_partial(position_input).unsqueeze(1).unsqueeze(1)
        for i in range(ref_sim.shape[0]):
            ref_sim[i, :, :, :position_cutoff_left[i]] = 0
            ref_sim[i, :, :, position_cutoff_right[i]:] = 0

        ref_sim = ref_sim.reshape(*ref_sim.shape[:3], -1) # batch, num_classes, n_ind_max, input_size * self.hidden0
        ref_sim = self.linear1(ref_sim) # batch, num_classes, n_ind_max, self.hidden1

        if ref_sim.shape[-2] < n_ind_max:
            padding = self.linear1.bias.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(*ref_sim.shape[:2], n_ind_max - ref_sim.shape[-2], -1)
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
    def __init__(self, input_size, label_groups, device):
        self.input_size = input_size
        self.min_pos_prob_tensor = torch.tensor([[min_pos_prob]], device=device)
        self.cl_labels = []
        for num_refs0_pass, num_refs1_pass in label_groups:
            labels = torch.empty((num_refs0_pass + num_refs1_pass,), dtype=torch.int8, device=device)
            labels[:num_refs0_pass] = 0
            labels[-num_refs0_pass:] = 2
            labels = labels + num_classes * torch.arange(num_classes, dtype=torch.int8, device=device).unsqueeze(-1)
            labels = labels.unsqueeze(-1)
            self.cl_labels.append(labels)

        self.OHE_values = OHE_values.to(device)
        self.device = device

    def set_lambda_f(self, num_generations_values, num_generations_probs, admixture_proportion, population_size):
        admix_time = num_generations_values.float()
        admixture_proportion = torch.tensor(sorted([admixture_proportion, 1 - admixture_proportion]), dtype=torch.float, device=self.device).unsqueeze(-1)
        self.lambda_f = 2 * (admix_time - 1) * admixture_proportion
        self.lambda_f *= (-0.5 * (admix_time - 1) / population_size).exp()
        self.lambda_f_probs = num_generations_probs
        self.lambda_f_iszero = (self.lambda_f[0] == 0)

    def get_inputs_training(self, SOI, refs, positions, label_id):

        # SOI             # batch, input_size
        # positions       # batch, input_size
        # refs            # batch, n_ind_max, input_size
        # self.lambda_f   # 2, len(num_generations_values)
        
        positions = (positions - positions[:, self.input_size // 2].unsqueeze(-1)).abs()
        
        pos_probs = (-self.lambda_f * positions.unsqueeze(-1).unsqueeze(-1)).exp()
        pos_probs[..., self.lambda_f_iszero] = (~positions.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, self.lambda_f.shape[1])[..., self.lambda_f_iszero].isinf()).float()
        pos_probs = (pos_probs * self.lambda_f_probs).sum(dim=-1)


        min_pos_prob_tensor = self.min_pos_prob_tensor.expand(pos_probs.shape[0], -1)
        position_cutoff_left = torch.searchsorted(pos_probs[:, :self.input_size // 2, 0], min_pos_prob_tensor, side="right")
        position_cutoff_right = self.input_size - torch.searchsorted(pos_probs[:, self.input_size // 2:, 0].flip(-1), min_pos_prob_tensor, side="right")

        SOI = SOI.unsqueeze(1).unsqueeze(1) # batch, 1, 1, input_size

        refs = refs.unsqueeze(1)

        ref_sim = self.cl_labels[label_id].unsqueeze(0) + (num_classes ** 2) * refs + (num_classes ** 3) * SOI
        ref_sim = self.OHE_values[ref_sim.long()]

        return ref_sim, pos_probs, position_cutoff_left, position_cutoff_right  

    def get_inputs_inference(self, SOI, refs, positions, label_id):

        # SOI             # input_size
        # positions       # batch, input_size
        # refs            # n_ind_max, input_size
        # self.lambda_f   # 2, len(num_generations_values)

        positions = (positions - positions[:, self.input_size // 2].unsqueeze(-1)).abs()
        
        pos_probs = (-self.lambda_f * positions.unsqueeze(-1).unsqueeze(-1)).exp()
        pos_probs[..., self.lambda_f_iszero] = (~positions.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, self.lambda_f.shape[1])[..., self.lambda_f_iszero].isinf()).float()
        pos_probs = (pos_probs * self.lambda_f_probs).sum(dim=-1)

        min_pos_prob_tensor = self.min_pos_prob_tensor.expand(pos_probs.shape[0], -1)
        position_cutoff_left = torch.searchsorted(pos_probs[:, :self.input_size // 2, 0], min_pos_prob_tensor, side="right")
        position_cutoff_right = self.input_size - torch.searchsorted(pos_probs[:, self.input_size // 2:, 0].flip(-1), min_pos_prob_tensor, side="right")

        SOI = SOI.unsqueeze(0).unsqueeze(0) # batch, 1, 1, input_size

        refs = refs.unsqueeze(0)

        ref_sim = self.cl_labels[label_id] + (num_classes ** 2) * refs + (num_classes ** 3) * SOI
        ref_sim = self.OHE_values[ref_sim.long()] # .int() ? # torch.gather?

        return ref_sim, pos_probs, position_cutoff_left, position_cutoff_right
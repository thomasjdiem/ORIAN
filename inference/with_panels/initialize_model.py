from shared import *
from neural_network import NeuralNetwork, InputProcessor
import torch
from scipy.stats import hypergeom
import math

@torch.no_grad()
def initialize_variables(gs):

    gs.len_chrom = (gs.positions[-1] - gs.positions[0]).item()


    def get_p_value(n0, n1):

        n0, n1 = sorted([n0, n1])

        if n0 >= n1 - 1:
            return 1.0
        
        p = 0
        for k in range(n0 + 1):
            p += hypergeom.pmf(k, 48, 24, n0 + n1)
        
        p *= 2
        return p

    p_value_threshold = 0.25

    numerator_per_denominator = {3: [1, 2], 4: [1, 3], 5: [1, 2, 3, 4]}

    best_option = None
    max_score = 0
    for denominator0 in range(1, max(5, gs.n_refs0 // min_num_refs_each)):
        for denominator1 in range(1, max(5, gs.n_refs1 // min_num_refs_each)):
            for numerator0 in numerator_per_denominator.get(denominator0, [1]):
                for numerator1 in numerator_per_denominator.get(denominator1, [1]):
                    n0 = (gs.n_refs0 * numerator0) // denominator0
                    n1 = (gs.n_refs1 * numerator1) // denominator1

                    total_refs_min_example = n0 + n1

                    if total_refs_min_example < min_num_refs_total:
                        continue

                    if (not min_num_refs_each <= n0 <= max_num_refs_each) or (not min_num_refs_each <= n1 <= max_num_refs_each):
                        continue

                    r0 = (gs.n_refs0 * numerator0) % denominator0
                    r1 = (gs.n_refs1 * numerator1) % denominator1

                    if (n0 <= n1) and (r0 < r1):
                        n1 += 1
                    elif (n0 >= n1) and (r0 > r1):
                        n0 += 1

                    p = get_p_value(n0, n1)
                    if p < p_value_threshold:
                        continue

                    min_num_refs_per_pass = min(n0, n1)
                    total_passes = math.lcm(denominator0, denominator1)

                    score = min_num_refs_per_pass ** 2 / total_passes
                    if score > max_score:
                        max_score = score
                        best_option = (denominator0, denominator1, numerator0, numerator1)


    denominator0, denominator1, numerator0, numerator1 = best_option
    n0 = (gs.n_refs0 * numerator0) // denominator0
    n1 = (gs.n_refs1 * numerator1) // denominator1

    r0 = (gs.n_refs0 * numerator0) % denominator0
    r1 = (gs.n_refs1 * numerator1) % denominator1

    total_passes = math.lcm(denominator0, denominator1)

    gs.ref_groups = []
    gs.label_ids = [-1]
    gs.label_groups = []
    gs.ref_group_weights = []
    ref0_index_start = 0
    ref1_index_start = 0
    for n_pass in range(total_passes):
        ref0_index_end = ref0_index_start + n0 + int(n_pass % denominator0 < r0)
        ref1_index_end = ref1_index_start + n1 + int(n_pass % denominator1 < r1)

        gs.ref_groups.append(torch.cat((torch.arange(ref0_index_start, ref0_index_end, dtype=torch.long, device=gs.device) % gs.n_refs0,
                                     torch.arange(ref1_index_start, ref1_index_end, dtype=torch.long, device=gs.device) % gs.n_refs1 + gs.n_refs0), dim=0))
        
        gs.ref_group_weights.append(len(gs.ref_groups[-1]))
        
        if (len(gs.ref_groups) > 1) and (gs.ref_groups[-1].shape == gs.ref_groups[-2].shape):
            gs.label_ids.append(gs.label_ids[-1])
        else:
            gs.label_ids.append(gs.label_ids[-1] + 1)
            gs.label_groups.append((ref0_index_end - ref0_index_start, ref1_index_end - ref1_index_start))
        
        ref0_index_start = ref0_index_end % gs.n_refs0
        ref1_index_start = ref1_index_end % gs.n_refs1

    gs.label_ids = gs.label_ids[1:]
    gs.label_group_frequencies = [gs.label_ids.count(i) * (gs.label_groups[i][0] + gs.label_groups[i][1]) for i in range(len(gs.label_groups))]
    gs.label_group_frequencies = torch.tensor(gs.label_group_frequencies, dtype=torch.float, device="cpu")
    gs.label_group_frequencies /= gs.label_group_frequencies.sum()

    gs.non_mono_sites = []
    for ref_group in gs.ref_groups:
        refs_sum = gs.refs[ref_group].sum(dim=0)
        gs.non_mono_sites.append((refs_sum != 0) & (refs_sum != 2 * len(ref_group)))

    gs.ref_group_weights = torch.tensor(gs.ref_group_weights, dtype=torch.float, device=gs.device)
    gs.ref_group_weights /= gs.ref_group_weights.sum()

    print("\n\n")
    print(f"Neural network inference will be broken up into {len(gs.ref_groups)} forward pass(es):")
    for i, (n0, n1) in enumerate(gs.label_groups):
        print(f"\t{n0} population 0 individuals and {n1} population 1 individuals\tx {gs.label_ids.count(i)}")

    print(f"Each reference individual from population 0 will be used in {sum(gs.label_groups[label_id][0] for label_id in gs.label_ids) // gs.n_refs0} forward pass(es)")
    print(f"Each reference individual from population 1 will be used in {sum(gs.label_groups[label_id][1] for label_id in gs.label_ids) // gs.n_refs1} forward pass(es)")
    


    if gs.num_generations is None:
        gs.num_generations_values = torch.arange(1, 1001).to(gs.device)
        gs.num_generations_prior_probs = torch.full((1000,), 1/1000, device=gs.device)
    elif isinstance(gs.num_generations, tuple):
        gs.num_generations_values = torch.tensor(gs.num_generations[0], device=gs.device)
        gs.num_generations_prior_probs = torch.tensor(gs.num_generations[1], device=gs.device)
        gs.num_generations_values = gs.num_generations_values[gs.num_generations_prior_probs > 0]
        gs.num_generations_prior_probs = gs.num_generations_prior_probs[gs.num_generations_prior_probs > 0]
        gs.num_generations_prior_probs /= gs.num_generations_prior_probs.sum()
    elif isinstance(gs.num_generations, (int, float)):
        gs.num_generations_values = torch.tensor([gs.num_generations], device=gs.device)
        gs.num_generations_prior_probs = torch.tensor([1.0], device=gs.device)


@torch.no_grad()
def initialize_model(gs):

    def get_input_size(max_dist):

        is_valid = lambda n: (gs.positions[n:] - gs.positions[:-n] > max_dist).all()
        # max dist = distance s.t. p_no_splits(dist) = min_pos_prob
        upper_bound = Input_size // 2
        lower_bound = int(len(gs.positions) * max_dist)
        if lower_bound >= upper_bound:
            return Input_size
        if not is_valid(upper_bound):
            return Input_size
        while upper_bound - lower_bound > 1:
            test_val = (upper_bound + lower_bound) // 2
            if is_valid(test_val):
                upper_bound = test_val
            else:
                lower_bound = test_val

        return 2 * upper_bound + 1
    
    lambda_f = 2 * (gs.num_generations_values - 1) * min(gs.global_admixture_proportion, 1 - gs.global_admixture_proportion) * (-0.5 * (gs.num_generations_values - 1) / gs.population_size).exp()
    lambda_f *= gs.num_generations_prior_probs
    lambda_f = lambda_f.sum().item()

    if lambda_f == 0:
        gs.input_size = Input_size
    else:
        max_dist = -math.log(min_pos_prob) / lambda_f # probability of no switch points betwen each position and position of interest
        gs.input_size = get_input_size(max_dist)

    print(f"\nNeural network input length: {gs.input_size}")

    gs.model = NeuralNetwork(gs.input_size).to(gs.device)

    pretrained_model = gs.script_dir + "/state_dict_with_panels.pth"
    original_state_dict = torch.load(pretrained_model, map_location=torch.device(gs.device), weights_only=True)
    new_state_dict = gs.model.state_dict()

    for name, param in original_state_dict.items():
        if name == "linear0.weight":
            new_state_dict["embedding0.weight"] = param[:, :-2].t()
            new_state_dict["linear0_partial.weight"] = param[:, -2:]
        elif name == "linear1.weight":
            new_state_dict[name] = param[:, n_embd_model * (Input_size // 2 - gs.input_size // 2): n_embd_model * (Input_size // 2 + gs.input_size // 2 + 1)]
        else:
            new_state_dict[name] = param

    gs.model.load_state_dict(new_state_dict)
    gs.model.eval()

    gs.input_processor = InputProcessor(gs.input_size, gs.label_groups, gs.device)
    gs.input_processor.set_lambda_f(gs.num_generations_values, gs.num_generations_prior_probs, gs.global_admixture_proportion, gs.population_size)
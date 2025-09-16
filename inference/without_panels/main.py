from shared import *
from math import exp, floor, ceil, log, sqrt
from scipy.stats import chisquare
from neural_network import NeuralNetwork, InputProcessor
import torch.nn.functional as F
import numpy as np
import allel
import argparse
import os

parser = argparse.ArgumentParser(description="Local Ancestry Inference Tool")
parser.add_argument("--adm-vcf", required=True, type=str, help="Admixed vcf") 
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("--recomb-rate", type=float, help="Constant recombination rate throughout the chromosome (M/bp)")
group.add_argument("--recomb-map-file", type=str, help="Recombination map file")
parser.add_argument("--admix-time", type=float, help="Admixture time in number of generations")
parser.add_argument("--admix-prop", required=True, type=float, help="Admixture proportion for either ancestry")
parser.add_argument("--population-size", type=float, default=10_000, help="Effective population size")
parser.add_argument("--batch-size", type=int, default=8, help="Inference batch size")
parser.add_argument("--device", type=str, help="Device to run program on (cpu or gpu). Will default to gpu if available, otherwise cpu")
parser.add_argument("--seed", type=int, default=0, help="Random seed (only relevant for fine tuning on reference panel)")
parser.add_argument("--out", required=True, type=str, help="Output directory")
parser.add_argument("--avg-seeds-per-chunk", default=3.0, help="Average number of restarts used on each chunk of chromosome")

args = parser.parse_args()

num_seed_tolerance = 4 # number of failed seeds before reducing size of window
solution_factor_start = 3600
num_predictions_per_update = 8
num_solution_factor_estimate = 32

if args.batch_size < 1:
    raise RuntimeError("Batch size must be at least 1")
batch_size = args.batch_size
batch_size = min(batch_size, num_predictions_per_update)

if args.avg_seeds_per_chunk < 1.0:
    raise RuntimeError("Avg seeds per chunk must be at least 1")

avg_num_seeds_per_chunk = args.avg_seeds_per_chunk

seed_offset = args.seed

if args.device is None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
elif args.device.strip().lower() == "cpu":
    device = "cpu"
elif args.device.strip().lower() in ["gpu", "cuda"]:
    if not torch.cuda.is_available():
        raise RuntimeError("Device set to gpu but torch.cuda.is_available() returns False")
    device = "cuda"
else:
    raise RuntimeError("Invalid device name.  Must be 'cpu' or 'gpu'")


# Process admixed vcf
callset = allel.read_vcf(args.adm_vcf, fields='*')

admixed_sample_names = list(callset["samples"])

hap_admixed = torch.tensor(callset["calldata/GT"], device=device, dtype=torch.int8)

if (hap_admixed == -1).any():
    raise RuntimeError("Missing alleles detected in admixed VCF genotype data.")

SOI = hap_admixed.sum(dim=-1).t()
num_samples = SOI.shape[0]

base_pair_positions = torch.tensor(callset["variants/POS"], device=device, dtype=torch.long)
alt_alleles = callset["variants/ALT"]
biallelic_mask = torch.tensor([sum(int(a != "") for a in alt) < 2 for alt in alt_alleles], device=device, dtype=torch.bool)

chromosome_array = callset['variants/CHROM']
adm_chrom = chromosome_array[0]
if not np.all(chromosome_array == adm_chrom):
    raise RuntimeError("Multiple chromosomes detected in admixed vcf")


# Process recombination map file/ recombination rate
if (args.recomb_rate is None) and (args.recomb_map_file is not None):
    recomb_positions_bp = []
    recomb_positions_morgans = []
    with open(args.recomb_map_file) as f:
        for recomb_map_file_line in f.readlines():
            if recomb_map_file_line.isspace():
                continue
            position_bp, position_morgan = recomb_map_file_line.strip().split("\t")

            position_bp = int(position_bp)
            position_morgan = float(position_morgan)

            recomb_positions_bp.append(position_bp)
            recomb_positions_morgans.append(position_morgan)

    recomb_positions_bp = torch.tensor(recomb_positions_bp, dtype=torch.long, device=device)

    if not torch.equal(recomb_positions_bp, base_pair_positions):
        raise RuntimeError("Base pair positions in admixed vcf and recombination map file do not match.")
    
    positions = torch.tensor(recomb_positions_morgans, dtype=torch.float, device=device)
    
    if torch.any(positions < 0):
        raise RuntimeError("Negative distances present in recombination map file.")
    
    positions = positions.cumsum(dim=0) - positions[0]

elif (args.recomb_rate is not None) and (args.recomb_map_file is None):

    positions = base_pair_positions.float() * args.recomb_rate
    positions = positions - positions[0]

else:
    raise RuntimeError("Exactly one of recombination rate and recombination map file must be specified.")


if args.admix_time < 1:
    raise RuntimeError("Admixture time must be at least 1, or -1 if unknown")

num_generations = args.admix_time


if not (0 < args.admix_prop < 1):
    raise ValueError("Admixture proportion must be between 0 and 1 (strict) if known, or -1 if unknown")

admixture_proportion = args.admix_prop

population_size = args.population_size


output_dir = args.out
output_dir = os.getcwd() + "/" + output_dir
if not output_dir.endswith("/"):
    output_dir += "/"

os.makedirs(output_dir, exist_ok=True)

print("\n")
print("Loaded inputs into memory ...")

SOI_sum = SOI.sum(dim=0)
non_monomorphic_mask = (SOI_sum > 0) & (SOI_sum < 2 * SOI.shape[0])
print(f"Pruning {len(positions) - non_monomorphic_mask.sum().item()} SNPs with invariant reference alleles and {len(positions) - biallelic_mask.sum().item()} SNPs with multiple alt alleles")
site_mask = non_monomorphic_mask & biallelic_mask

SOI = SOI[:, site_mask]
positions = positions[site_mask]
base_pair_positions = base_pair_positions[site_mask]

print(f"Number of usable SNPs: {len(positions)}")
print(f"Chromosome length (morgans): {positions[-1].item():0.3f}")
print(f"Number of admixed individuals: {num_samples}")
print(f"Admixture fraction: {admixture_proportion}")
print(f"Time since admixture (generations): {num_generations}")
print(f"Effective population size: {population_size}")
print(f"Device: {device}")
print(f"Output directory: {output_dir}")

with torch.no_grad():

    if population_size is None:
        population_size = 10_000

    if admixture_proportion is None:
        admixture_proportion = 0.5


    def get_input_size(max_dist):

        is_valid = lambda n: (positions[n:] - positions[:-n] > max_dist).all()
        # max dist = distance s.t. p_no_splits(dist) = min_pos_prob
        upper_bound = Input_size // 2
        lower_bound = int(len(positions) * max_dist)
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
    
    lambda_f = 2 * (num_generations - 1) * min(admixture_proportion, 1 - admixture_proportion) * exp(-0.5 * (num_generations - 1) / population_size)


    if lambda_f == 0:
        input_size = Input_size
    else:
        max_dist = -log(min_pos_prob) / lambda_f # probability of no switch points betwen each position and position of interest
        input_size = get_input_size(max_dist)

    model = NeuralNetwork(input_size).to(device)
    pretrained_model = "state_dict_without_panels.pth"

    original_state_dict = torch.load(pretrained_model, map_location=torch.device(device))
    if pretrained_model.endswith(".tar"):
        original_state_dict = original_state_dict["model_state_dict"]
    
    new_state_dict = model.state_dict()

    for name, param in original_state_dict.items():
        if name == "linear1.weight":
            new_state_dict[name] = param[:, n_embd_model * (Input_size // 2 - input_size // 2): n_embd_model * (Input_size // 2 + input_size // 2 + 1)]
        else:
            new_state_dict[name] = param

    model.load_state_dict(new_state_dict)
    model.eval()
    model.fuse_layers(device)


    torch.manual_seed(-1 + seed_offset)


    num_individuals, len_seq = SOI.shape

    input_processor = InputProcessor(input_size, device)
    input_processor.set_lambda_f(num_generations, admixture_proportion, population_size)

    expected_num_splits_per_chunk = 4
    window_eval_interval = num_individuals * 20
    chunk_overlap = 0.5
    max_intervals = 18

    len_chrom_morgans = (positions[-1] - positions[0]).item() * (1 + 1/len(positions)) 
    lambda_f = 4 * (num_generations - 1) * len_chrom_morgans * admixture_proportion * (1 - admixture_proportion) 
    lambda_f *= exp(-0.5 * (num_generations - 1) / population_size)

    label_mask = ~torch.eye(num_individuals, dtype=torch.bool)

    lam = 4 * (num_generations - 1) * admixture_proportion * (1 - admixture_proportion) 
    lam *= exp(-0.5 * (num_generations - 1) / population_size)

    base_output = torch.tensor([0.25, 0.5, 0.25], device=device)

    golden_ratio = (sqrt(5) + 1) / 2

    def get_certainty_of_vector(v):
        v /= base_output
        v /= v.sum(dim=-1, keepdim=True)
        v[v.isnan()] = 1/3
        out = ((v - 1/3) ** 2).sum(dim=-1)
        return out
    
    padding = torch.full((num_individuals, input_size // 2), -1).to(torch.int8).to(device)
    SOI = torch.cat((padding, SOI, padding), dim=1) # (num_individuals, len_seq + input_size - 1)

    
    num_forward_passes = ceil((num_individuals - 1) / 48)
    num_refs_per_pass = [(num_individuals - 1) // num_forward_passes + int(i < ((num_individuals - 1) % num_forward_passes)) for i in range(num_forward_passes)]
    forward_pass_weights = torch.tensor([num_refs_per_pass], dtype=torch.float, device=device).unsqueeze(-1)
    forward_pass_weights /= forward_pass_weights.sum()
    batch_size = max(batch_size, num_forward_passes)
    if min(num_refs_per_pass) != max(num_refs_per_pass):
        refs_padding_forward = torch.zeros((1, input_size), dtype=torch.int8, device=device)
        labels_padding_forward = torch.zeros((1, input_size, 3), dtype=torch.float, device=device)
    else:
        refs_padding_forward = labels_padding_forward = None


    chunk_length = 1e6 if lambda_f == 0 else expected_num_splits_per_chunk / lambda_f
    num_chunks = (1/chunk_length - chunk_overlap) / (1 - chunk_overlap)
    num_chunks = max(num_chunks, 1)
    num_chunks = floor(num_chunks) if log(num_chunks) - log(floor(num_chunks)) < log(ceil(num_chunks)) - log(num_chunks) else ceil(num_chunks)
    chunk_length = 1 / ((1 - chunk_overlap) * num_chunks + chunk_overlap)

    chunk_starts = torch.tensor([(1 - chunk_overlap) * i * chunk_length for i in range(num_chunks)], device=device)
    chunk_ends = chunk_starts + chunk_length
    chunk_starts = torch.cat((chunk_starts[0].unsqueeze(0), chunk_starts, chunk_starts[-1].unsqueeze(0) * chunk_overlap + chunk_ends[-1].unsqueeze(0) * (1 - chunk_overlap)), dim=0)
    chunk_ends = torch.cat((chunk_starts[0].unsqueeze(0) * (1 - chunk_overlap) + chunk_ends[0].unsqueeze(0) * chunk_overlap, chunk_ends, chunk_ends[-1].unsqueeze(0)), dim=0)

    chunk_starts = chunk_starts * len_chrom_morgans + positions[0]
    chunk_ends = chunk_ends * len_chrom_morgans + positions[0]

    base_num_chunks = num_chunks + 1

    num_chunks += 2

    last_path = None
    chosen_chunk = None

    idx_start_chunked = torch.searchsorted(positions, chunk_starts)
    idx_end_chunked = torch.searchsorted(positions, chunk_ends)
    len_seq_chunked = idx_end_chunked - idx_start_chunked

    positions_chunked = [positions[idx_start_chunked[chunk]:idx_end_chunked[chunk]] for chunk in range(num_chunks)]
    padding = torch.full((input_size // 2,), float("inf"), device=device)
    positions_chunked = [torch.cat((padding, positions_chunked[chunk], padding), dim=0) for chunk in range(num_chunks)]
    
    chunk_length_proportion = [0.5] + [1.0 for _ in range(num_chunks - 2)] + [0.5]
    
    current_seeds_chunked = [[0] for _ in range(num_chunks)]
    total_seeds_chunked = [1 for _ in range(num_chunks)]
    
    num_iterations_current_chunked = {(chunk, seed): 0 for chunk in range(num_chunks) for seed in current_seeds_chunked[chunk]}
    num_iterations_allocated_chunked = {(chunk, seed): int(window_eval_interval * chunk_length_proportion[chunk]) for chunk in range(num_chunks) for seed in current_seeds_chunked[chunk]}
    num_intervals_chunked = {(chunk, seed): 1 for chunk in range(num_chunks) for seed in current_seeds_chunked[chunk]}
    
    solution_factor_chunked = {(chunk, seed): solution_factor_start for chunk in range(num_chunks) for seed in current_seeds_chunked[chunk]}
    
    next_chunks = {chunk: {chunk + 1} for chunk in range(num_chunks - 1)}
    next_chunks["start node"] = {0}
    next_chunks[num_chunks - 1] = {"end node"}
    next_chunks["end node"] = set()

    overlap_factors = {(chunk, chunk + 1): 1 for chunk in range(num_chunks - 1)}
    overlap_factors[("start node", 0)] = 0
    overlap_factors[(num_chunks - 1, "end node")] = 0

    parent_chunks = {chunk: None for chunk in range(num_chunks)}
    child_chunks = {chunk: None for chunk in range(num_chunks)}

    copies_chunked = {(chunk, seed): None for chunk in range(num_chunks) for seed in current_seeds_chunked[chunk]}
    
    has_changed_chunked = {}
    p_values_chunked = {}


    rand_starts_chunked = [torch.randperm(num_individuals).long() for _ in range(num_chunks)]

    predictions_strict_chunked = {}
    pairwise_sim_chunked = {}
    is_flipped_chunked = {}
    is_flipped_chunked_skipping = {}

    avg_certainties_chunked = {}
    certainty_chunked = {}
    predictions_start_idx_chunked = {}
    predictions_end_idx_chunked = {}
    predictions_chunked = []
    current_prediction_idx = input_size // 2
    for chunk in range(num_chunks):
        for seed in current_seeds_chunked[chunk]:

            positions_morgans_tmp = (positions_chunked[chunk][input_size // 2: -(input_size // 2)] - positions_chunked[chunk][len_seq_chunked[chunk] // 2 + input_size // 2]).abs()
            transition_aa_haploid = 0.5 + 0.5 * torch.exp(-lam * positions_morgans_tmp)

            infered_tract0 = 0.5 * transition_aa_haploid
            infered_tract2 = 0.5 - infered_tract0
                
            rand_start = rand_starts_chunked[chunk][seed % num_individuals]

            predictions_chunked.append(torch.empty((num_individuals, len_seq_chunked[chunk], num_classes), device=device))
            predictions_chunked[-1][:] = base_output 
            predictions_chunked[-1][rand_start, :, 0] = infered_tract0
            predictions_chunked[-1][rand_start, :, 2] = infered_tract2

            predictions_start_idx_chunked[(chunk, seed)] = current_prediction_idx
            current_prediction_idx += len_seq_chunked[chunk].item()
            predictions_end_idx_chunked[(chunk, seed)] = current_prediction_idx

            certainty_chunked[(chunk, seed)] = torch.zeros((num_individuals, len_seq_chunked[chunk]), device=device)

    padding = torch.zeros((num_individuals, input_size // 2, num_classes), device=device)
    predictions_chunked = torch.cat([padding] + predictions_chunked + [padding], dim=1)

    while True:
        for chunk in range(num_chunks):

            for seed in current_seeds_chunked[chunk]:
                has_changed_chunked[(chunk, seed)] = False
                
                prob_selection_step = ceil(len_seq_chunked[chunk].item() / (2 ** 24)) # reduce prob tensor size if extremely large (multinomial cannot take extremely large tensor)
                avg_input_distance = (positions_chunked[chunk][-(input_size // 2 + 1)] - positions_chunked[chunk][input_size // 2]).item() * (input_size // 2) / len_seq_chunked[chunk]
                while num_iterations_current_chunked[(chunk, seed)] < num_iterations_allocated_chunked[(chunk, seed)]:

                    # fix hard coding in the next few lines
                    is_evaluation_iteration = (80 <= num_iterations_current_chunked[(chunk, seed)] % 400 < 80 + num_solution_factor_estimate)# and (iteration > 200)
                    num_predictions_iteration = num_solution_factor_estimate if is_evaluation_iteration else num_predictions_per_update

                    random_seed_iteration = 400 * (seed * num_chunks + chunk) + seed_offset + num_iterations_current_chunked[(chunk, seed)]
                    torch.manual_seed(random_seed_iteration)

                    probabilities = 1 - predictions_chunked[:, predictions_start_idx_chunked[(chunk, seed)]: predictions_end_idx_chunked[(chunk, seed)]].amax(dim=-1).cpu()
                    if num_iterations_current_chunked[(chunk, seed)] == 0:
                        probabilities[rand_starts_chunked[chunk][seed % num_individuals]] = 0

                    ind3 = torch.multinomial(probabilities.sum(dim=-1), num_predictions_iteration, replacement=is_evaluation_iteration)
                    ind4 = torch.multinomial(probabilities[ind3][:, ::prob_selection_step], 1).squeeze(-1) * prob_selection_step

                    out = []
                    new_certainty = []
                    for idx_start in range(0, num_predictions_iteration, batch_size // num_forward_passes):
                        idx_end = min(idx_start + batch_size // num_forward_passes, num_predictions_iteration)
                        
                        if num_forward_passes == 1:
                            SOI_batch = torch.stack([SOI[:, idx_start_chunked[chunk]: idx_end_chunked[chunk] + input_size - 1][ind3[j], ind4[j]:ind4[j] + input_size] for j in range(idx_start, idx_end)])
                            refs_batch = torch.stack([SOI[:, idx_start_chunked[chunk]: idx_end_chunked[chunk] + input_size - 1][:, ind4[j]:ind4[j] + input_size][label_mask[ind3[j]]] for j in range(idx_start, idx_end)])
                            labels_batch = torch.stack([predictions_chunked[:, predictions_start_idx_chunked[(chunk, seed)] - input_size // 2: predictions_end_idx_chunked[(chunk, seed)] + input_size // 2][:, ind4[j]:ind4[j]+input_size][label_mask[ind3[j]]] for j in range(idx_start, idx_end)])
                            positions_batch = torch.stack([positions_chunked[chunk][ind4[j]:ind4[j] + input_size] for j in range(idx_start, idx_end)])
                        else:
                            SOI_batch = []
                            refs_batch = []
                            labels_batch = []
                            positions_batch = []
                            for j in range(idx_start, idx_end):

                                SOI_batch_id = SOI[ind3[j], idx_start_chunked[chunk]: idx_end_chunked[chunk] + input_size - 1]
                                refs_batch_id = SOI[:, idx_start_chunked[chunk]: idx_end_chunked[chunk] + input_size - 1][label_mask[ind3[j]]]
                                labels_batch_id = predictions_chunked[:, predictions_start_idx_chunked[(chunk, seed)] - input_size // 2: predictions_end_idx_chunked[(chunk, seed)] + input_size // 2][label_mask[ind3[j]]]

                                middle_idx = ind4[j].item() + input_size // 2

                                refs_perm = torch.randperm(num_individuals - 1)
                                start_ref_id = 0
                                
                                for n_pass in range(num_forward_passes):

                                    end_ref_id = start_ref_id + num_refs_per_pass[n_pass]
                                    refs_pass_idx = refs_perm[start_ref_id: end_ref_id]
                                    start_ref_id = end_ref_id

                                    refs_batch_id_pass = refs_batch_id[refs_pass_idx]

                                    left_cutoff = input_size // 2 # leftmost idx that is valid within chunk
                                    right_cutoff = idx_end_chunked[chunk].item() - idx_start_chunked[chunk].item() + (input_size // 2) # rightmost idx that is valid within chunk

                                    left_idx = max(left_cutoff, middle_idx - input_size // 2 - 1)
                                    refs_example_sum = refs_batch_id_pass[:, left_idx:middle_idx].sum(dim=0) + SOI_batch_id[left_idx:middle_idx]
                                    mask = (refs_example_sum > 0) & (refs_example_sum < 2 * num_refs_per_pass[n_pass] + 2)
                                    valid_mask_left = torch.nonzero(mask) + left_idx
                                    while (left_idx > left_cutoff) and (len(valid_mask_left) < input_size // 2 + 1): 
                                        left_idx_new = max(left_cutoff, left_idx - input_size // 2 - 1 + len(valid_mask_left))
                                        refs_example_sum = refs_batch_id_pass[:, left_idx_new:left_idx].sum(dim=0) + SOI_batch_id[left_idx_new:left_idx]
                                        mask = (refs_example_sum > 0) & (refs_example_sum < 2 * num_refs_per_pass[n_pass] + 2)
                                        valid_mask_left = torch.cat((torch.nonzero(mask) + left_idx_new, valid_mask_left), dim=0)
                                        left_idx = left_idx_new

                                    valid_mask_left = valid_mask_left.squeeze(-1)

                                    right_idx = min(right_cutoff, middle_idx + input_size // 2 + 1)
                                    refs_example_sum = refs_batch_id_pass[:, middle_idx:right_idx].sum(dim=0) + SOI_batch_id[middle_idx:right_idx]
                                    mask = (refs_example_sum > 0) & (refs_example_sum < 2 * num_refs_per_pass[n_pass] + 2)
                                    valid_mask_right = torch.nonzero(mask) + middle_idx
                                    while (right_idx < right_cutoff) and (len(valid_mask_right) < input_size // 2 + 1): 
                                        right_idx_new = min(right_cutoff, right_idx + input_size // 2 + 1 - len(valid_mask_right))
                                        refs_example_sum = refs_batch_id_pass[:, right_idx:right_idx_new].sum(dim=0) + SOI_batch_id[right_idx:right_idx_new]
                                        mask = (refs_example_sum > 0) & (refs_example_sum < 2 * num_refs_per_pass[n_pass] + 2)
                                        valid_mask_right = torch.cat((valid_mask_right, torch.nonzero(mask) + right_idx), dim=0)
                                        right_idx = right_idx_new

                                    valid_mask_right = valid_mask_right.squeeze(-1)

                                    if len(valid_mask_right) == 0:
                                        left_padding = input_size // 2 + 1 - len(valid_mask_left)
                                        right_padding = input_size // 2
                                    elif len(valid_mask_left) == 0:
                                        left_padding = input_size // 2
                                        right_padding = input_size // 2 + 1 - len(valid_mask_right)
                                    elif valid_mask_right[0] == middle_idx:
                                        left_padding = input_size // 2 + 1 - len(valid_mask_left)
                                        right_padding = input_size // 2 + 1 - len(valid_mask_right)
                                        valid_mask_left = valid_mask_left[1:]
                                    elif torch.rand((1,)).item() < 0.5:
                                        left_padding = input_size // 2 + 1 - len(valid_mask_left)
                                        right_padding = input_size // 2 + 1 - len(valid_mask_right)
                                        valid_mask_left = valid_mask_left[1:]
                                    else:
                                        left_padding = input_size // 2 + 1 - len(valid_mask_left)
                                        right_padding = input_size // 2 + 1 - len(valid_mask_right)
                                        valid_mask_right = valid_mask_right[:-1]

                                    positional_idx = torch.cat((torch.arange(left_cutoff - left_padding, left_cutoff, dtype=torch.long, device=device),
                                                                valid_mask_left, valid_mask_right,
                                                                torch.arange(right_cutoff, right_cutoff + right_padding, dtype=torch.long, device=device)), dim=0)

                                    refs_batch_id_pass = refs_batch_id_pass[:, positional_idx]
                                    SOI_batch_id_pass = SOI_batch_id[positional_idx]

                                    positions_batch_id_pass = positions_chunked[chunk][positional_idx]

                                    labels_batch_id_pass = labels_batch_id[refs_pass_idx][:, positional_idx]

                                    if num_refs_per_pass[n_pass] < max(num_refs_per_pass):
                                        refs_batch_id_pass = torch.cat((refs_batch_id_pass, refs_padding_forward), dim=0)
                                        labels_batch_id_pass = torch.cat((labels_batch_id_pass, labels_padding_forward), dim=0)

                                    SOI_batch.append(SOI_batch_id_pass)
                                    refs_batch.append(refs_batch_id_pass)
                                    labels_batch.append(labels_batch_id_pass)
                                    positions_batch.append(positions_batch_id_pass)

                            SOI_batch = torch.stack(SOI_batch)
                            refs_batch = torch.stack(refs_batch)
                            labels_batch = torch.stack(labels_batch)
                            positions_batch = torch.stack(positions_batch)

                        ref_sim = input_processor.get_inputs(SOI_batch, refs_batch, labels_batch, positions_batch)

                        out.append(model(ref_sim))

                        new_certainty_batched = get_certainty_of_vector(labels_batch).sum(dim=1)
                        new_certainty_batched *= (-solution_factor_chunked[(chunk, seed)] * (positions_batch - positions_batch[:, input_size // 2].unsqueeze(-1)).abs()).exp() * (solution_factor_chunked[(chunk, seed)] / (1 - exp(-avg_input_distance * solution_factor_chunked[(chunk, seed)])))
                        new_certainty_batched = new_certainty_batched.sum(dim=-1)

                        new_certainty.append(new_certainty_batched)


                    out = torch.cat(out, dim=0)
                    out = F.softmax(out, dim=-1) # batch, num_classes
                    out = out.reshape(num_predictions_iteration, num_forward_passes, 3)
                    out *= forward_pass_weights
                    out = out.sum(dim=1)
                    
                    new_certainty = torch.cat(new_certainty, dim=0)
                    new_certainty = new_certainty.reshape(num_predictions_iteration, num_forward_passes).sum(dim=-1)

                    positions_diff = (positions_chunked[chunk][input_size // 2: -(input_size // 2)] - positions_chunked[chunk][ind4 + input_size // 2].unsqueeze(-1)).abs() # batch, len_seq + input_size

                    transition_00_hap = 0.5 + 0.5 * torch.exp(-lam * positions_diff) # batch, len_seq + input_size - 1
                    transition_01_hap = 1 - transition_00_hap

                    transition_00 = transition_00_hap ** 2
                    transition_02 = transition_01_hap ** 2
                    transition_10 = transition_00_hap * transition_01_hap
                    transition_01 = transition_10 * 2
                    
                    transitions = torch.empty((num_predictions_iteration, len_seq_chunked[chunk], num_classes, num_classes), device=device)
                    transitions[:, :, 0, 0] = transition_00
                    transitions[:, :, 0, 1] = transition_01
                    transitions[:, :, 0, 2] = transition_02
                    transitions[:, :, 1, 0] = transition_10
                    transitions[:, :, 1, 1] = transition_00 + transition_02
                    transitions[:, :, 1, 2] = transition_10
                    transitions[:, :, 2, 0] = transition_02
                    transitions[:, :, 2, 1] = transition_01
                    transitions[:, :, 2, 2] = transition_00

                    old_certainty = certainty_chunked[(chunk, seed)][ind3].unsqueeze(-1)
                    out_smoothed = (out.unsqueeze(1).unsqueeze(1) @ transitions).squeeze(-2) 

                    if is_evaluation_iteration:

                        def combined_certainty(s):

                            new_certainty_local = (new_certainty.unsqueeze(-1) * (-s * positions_diff).exp()).unsqueeze(-1)

                            proportion_new = new_certainty_local / (new_certainty_local + old_certainty)
                            proportion_new[proportion_new.isnan()] = 0.0
                            updated_predictions = predictions_chunked[:, predictions_start_idx_chunked[(chunk, seed)]: predictions_end_idx_chunked[(chunk, seed)]][ind3] * (1 - proportion_new) + out_smoothed * proportion_new

                            return get_certainty_of_vector(updated_predictions).mean()

                        s_low = 0
                        s_high = 2e5
                        s1 = s_high - (s_high - s_low) / golden_ratio
                        s2 = s_low + (s_high - s_low) / golden_ratio

                        while s_high - s_low > 1:
                            certainty1 = combined_certainty(s1)
                            certainty2 = combined_certainty(s2)
                            
                            if certainty1 > certainty2:
                                s_high = s2
                            else:
                                s_low = s1
                            
                            s1 = s_high - (s_high - s_low) / golden_ratio
                            s2 = s_low + (s_high - s_low) / golden_ratio

                        solution_factor_chunked[(chunk, seed)] = (s_low + s_high) / 2

                    else:
                        has_changed_chunked[(chunk, seed)] = True

                        new_certainty = (new_certainty.unsqueeze(-1) * (-solution_factor_chunked[(chunk, seed)] * positions_diff).exp()).unsqueeze(-1)

                        proportion_new = new_certainty / (new_certainty + old_certainty)
                        proportion_new[proportion_new.isnan()] = 0.0
                        predictions_chunked[ind3, predictions_start_idx_chunked[(chunk, seed)]: predictions_end_idx_chunked[(chunk, seed)]] = predictions_chunked[ind3, predictions_start_idx_chunked[(chunk, seed)]: predictions_end_idx_chunked[(chunk, seed)]] * (1 - proportion_new) + out_smoothed * proportion_new

                        certainty_chunked[(chunk, seed)][ind3] = torch.maximum(old_certainty, new_certainty).squeeze(-1)


                    num_iterations_current_chunked[(chunk, seed)] += num_predictions_iteration


        for chunk in range(num_chunks):
            for seed in current_seeds_chunked[chunk]:
                if has_changed_chunked[(chunk, seed)]:
                    predictions_strict_chunked[(chunk, seed)] = predictions_chunked[:, predictions_start_idx_chunked[(chunk, seed)]: predictions_end_idx_chunked[(chunk, seed)]].argmax(dim=-1).to(torch.int8)


        # remove windows where number of 0 to 2 (or vice versa) switches within distance is extremely improbable
        p_value_hard_threshold = 5e-4
        window_distance = 0.1

        new_windows = []
        bad_windows = []
        for chunk in range(num_chunks):


            for seed in current_seeds_chunked[chunk]:

                if not has_changed_chunked[(chunk, seed)]:
                    continue

                idx_distance = int(window_distance * len_seq_chunked[chunk].item())

                positions_bin_size = (positions_chunked[chunk][input_size // 2 + idx_distance: -(input_size // 2):idx_distance // 2] - positions_chunked[chunk][input_size // 2: -(input_size // 2 + idx_distance):idx_distance // 2]).double().cpu()
                positions_bin_size /= positions_bin_size.sum()

                num_02_switches = ((predictions_strict_chunked[(chunk, seed)][:, idx_distance::idx_distance // 2] - predictions_strict_chunked[(chunk, seed)][:, :-idx_distance:idx_distance // 2]).abs() == 2).sum(dim=0).double().cpu()
                if num_02_switches.sum() == 0:  
                    p_values_chunked[(chunk, seed)] = 1.0
                    continue

                expected_num_02_switches = positions_bin_size * num_02_switches.sum()
                
                if len(num_02_switches) % 2 == 1:
                    observed_counts1 = torch.cat((num_02_switches[0].unsqueeze(0), num_02_switches[1::2] + num_02_switches[2::2]), dim=0)
                    expected_counts1 = torch.cat((expected_num_02_switches[0].unsqueeze(0), expected_num_02_switches[1::2] + expected_num_02_switches[2::2]), dim=0)

                    observed_counts2 = torch.cat((num_02_switches[:-1:2] + num_02_switches[1:-1:2], num_02_switches[-1].unsqueeze(0)), dim=0)
                    expected_counts2 = torch.cat((expected_num_02_switches[:-1:2] + expected_num_02_switches[1:-1:2], expected_num_02_switches[-1].unsqueeze(0)), dim=0)
                else:
                    observed_counts1 = (num_02_switches[::2] + num_02_switches[1::2]).tolist()
                    expected_counts1 = (expected_num_02_switches[::2] + expected_num_02_switches[1::2]).tolist()

                    observed_counts2 = torch.cat((num_02_switches[0].unsqueeze(0), num_02_switches[1:-1:2] + num_02_switches[2:-1:2], num_02_switches[-1].unsqueeze(0)), dim=0)
                    expected_counts2 = torch.cat((expected_num_02_switches[0].unsqueeze(0), expected_num_02_switches[1:-1:2] + expected_num_02_switches[2:-1:2], expected_num_02_switches[-1].unsqueeze(0)), dim=0)
                
                _, p_value1 = chisquare(observed_counts1, expected_counts1)
                if p_value1 < p_value_hard_threshold:
                    bad_windows.append((chunk, seed))
                    continue

                _, p_value2 = chisquare(observed_counts2, expected_counts2)
                if p_value2 < p_value_hard_threshold:
                    bad_windows.append((chunk, seed))
                    continue

                p_values_chunked[(chunk, seed)] = min(p_value1, p_value2)

        for chunk, seed in bad_windows:
            print(f"Removing window ({chunk}, {seed}) due to p-value test")
            current_seeds_chunked[chunk].remove(seed)
            num_iterations_current_chunked.pop((chunk, seed))
            num_iterations_allocated_chunked.pop((chunk, seed))
            num_intervals_chunked.pop((chunk, seed))
            certainty_chunked.pop((chunk, seed))
            predictions_strict_chunked.pop((chunk, seed))
            if len(current_seeds_chunked[chunk]) == 0:
                new_windows.append(chunk)

        if all(len(current_seeds_chunked[chunk]) > 0 for chunk in range(num_chunks)):

            min_current_intervals = min(num_intervals_chunked.values())
            if min_current_intervals == max_intervals:

                certainty_chunked.clear()

                for chunk1 in range(num_chunks):
                    for chunk2 in next_chunks[chunk1]:
                        if chunk2 == "end node":
                            continue
                        for seed1 in current_seeds_chunked[chunk1]:
                            for seed2 in current_seeds_chunked[chunk2]:
                                if ((chunk1, seed1), (chunk2, seed2)) not in pairwise_sim_chunked:
                                    overlap_start = max(idx_start_chunked[chunk1].item(), idx_start_chunked[chunk2].item())
                                    overlap_end = min(idx_end_chunked[chunk1].item(), idx_end_chunked[chunk2].item())
                                    
                                    chunk1_start = overlap_start - idx_start_chunked[chunk1].item()
                                    chunk1_end = overlap_end - idx_start_chunked[chunk1].item()

                                    chunk2_start = overlap_start - idx_start_chunked[chunk2].item()
                                    chunk2_end = overlap_end - idx_start_chunked[chunk2].item()

                                    left_prediction = predictions_strict_chunked[(chunk1, seed1)][:, chunk1_start:chunk1_end]
                                    right_prediction = predictions_strict_chunked[(chunk2, seed2)][:, chunk2_start:chunk2_end]

                                    similarity = (left_prediction == right_prediction).sum().item() / left_prediction.numel()
                                    similarity_flipped = (left_prediction == 2 - right_prediction).sum().item() / left_prediction.numel()

                                    if (similarity >= similarity_flipped):
                                        pairwise_sim_chunked[((chunk1, seed1), (chunk2, seed2))] = similarity
                                        is_flipped_chunked[((chunk1, seed1), (chunk2, seed2))] = 0
                                    else:
                                        pairwise_sim_chunked[((chunk1, seed1), (chunk2, seed2))] = similarity_flipped
                                        is_flipped_chunked[((chunk1, seed1), (chunk2, seed2))] = 1

                chunks_topo_pos = (chunk_starts + chunk_ends) / 2

                sim_values = {}
                chunk_paths = {}
                is_flipped_paths = {}
                sim_paths = {}

                start_chunk = "start node"
                sim_values[start_chunk] = {0: 0}
                chunk_paths[start_chunk] = {0: [(start_chunk, 0)]}
                is_flipped_paths[start_chunk] = {0: []}
                sim_paths[start_chunk] = {0: []}

                current_chunks = {start_chunk}
                possible_chunks = next_chunks[start_chunk]
                explored_chunks = {start_chunk}

                if (last_path is None) and (chosen_chunk is not None):
                    forced_window = (chosen_chunk, current_seeds_chunked[chosen_chunk][-1])
                else:
                    forced_window = None

                while True:
                    next_chunk = min(possible_chunks, key=lambda x: chunks_topo_pos[x] if isinstance(x, int) else float("inf"))

                    next_seeds = current_seeds_chunked[next_chunk] if isinstance(next_chunk, int) else [0]

                    sim_values[next_chunk] = {seed: -1 for seed in next_seeds}
                    chunk_paths[next_chunk] = {}
                    is_flipped_paths[next_chunk] = {}
                    sim_paths[next_chunk] = {}

                    for current_chunk in current_chunks:
                        if next_chunk in next_chunks[current_chunk]:
                            current_seeds = current_seeds_chunked[current_chunk] if isinstance(current_chunk, int) else [0]
                            for current_seed in current_seeds:
                                for next_seed in next_seeds:
                                    new_sim_value = sim_values[current_chunk][current_seed] + pairwise_sim_chunked.get(((current_chunk, current_seed), (next_chunk, next_seed)), 0) * overlap_factors[(current_chunk, next_chunk)] + 100 * int((current_chunk, current_seed) == forced_window)
                                    if new_sim_value > sim_values[next_chunk][next_seed]:
                                        sim_values[next_chunk][next_seed] = new_sim_value
                                        chunk_paths[next_chunk][next_seed] = chunk_paths[current_chunk][current_seed] + [(next_chunk, next_seed)]
                                        is_flipped_paths[next_chunk][next_seed] = is_flipped_paths[current_chunk][current_seed] + [is_flipped_chunked.get(((current_chunk, current_seed), (next_chunk, next_seed)), 0)]
                                        sim_paths[next_chunk][next_seed] = sim_paths[current_chunk][current_seed] + [pairwise_sim_chunked.get(((current_chunk, current_seed), (next_chunk, next_seed)), 0)]

                    if next_chunk == "end node":
                        break

                    explored_chunks.add(next_chunk)
                    current_chunks.add(next_chunk)
                    current_chunks = {chunk for chunk in current_chunks if not next_chunks[chunk].issubset(explored_chunks)}
                    possible_chunks = set().union(*[next_chunks[chunk] for chunk in current_chunks]) - current_chunks
                    
                    for key in list(sim_values.keys()):
                        if key not in current_chunks:
                            sim_values.pop(key)
                            chunk_paths.pop(key)
                            is_flipped_paths.pop(key)
                            sim_paths.pop(key)
                                
                chunk_path = chunk_paths["end node"][0][1:-1]
                is_flipped_path = is_flipped_paths["end node"][0][1:-1]
                sim_path = sim_paths["end node"][0][1:-1]

                # find trios where flipping is inconsistent
                bad_trios = []
                for i in range(len(chunk_path) - 2):
                    is_flipped_trio = (is_flipped_path[i] + is_flipped_path[i+1] == 1)

                    chunk1, seed1 = chunk_path[i]
                    chunk2, seed2 = chunk_path[i+2]

                    if ((chunk1, seed1), (chunk2, seed2)) in is_flipped_chunked_skipping:

                        is_flipped_direct = is_flipped_chunked_skipping[((chunk1, seed1), (chunk2, seed2))]

                    else:
                        overlap_start = max(idx_start_chunked[chunk1].item(), idx_start_chunked[chunk2].item())
                        overlap_end = min(idx_end_chunked[chunk1].item(), idx_end_chunked[chunk2].item())
                        
                        chunk1_start = overlap_start - idx_start_chunked[chunk1].item()
                        chunk1_end = overlap_end - idx_start_chunked[chunk1].item()

                        chunk2_start = overlap_start - idx_start_chunked[chunk2].item()
                        chunk2_end = overlap_end - idx_start_chunked[chunk2].item()

                        if overlap_start >= overlap_end:
                            chunk1_end = len_seq_chunked[chunk1].item()
                            chunk1_start = chunk1_end - 1
                            chunk2_start = 0
                            chunk2_end = 1

                        left_prediction = predictions_strict_chunked[(chunk1, seed1)][:, chunk1_start:chunk1_end]
                        right_prediction = predictions_strict_chunked[(chunk2, seed2)][:, chunk2_start:chunk2_end]

                        similarity = (left_prediction == right_prediction).sum().item() / left_prediction.numel()
                        similarity_flipped = (left_prediction == 2 - right_prediction).sum().item() / left_prediction.numel()

                        is_flipped_direct = (similarity_flipped > similarity)

                        is_flipped_chunked_skipping[((chunk1, seed1), (chunk2, seed2))] = is_flipped_direct

                    if (is_flipped_direct != is_flipped_trio):
                        bad_trios.append(chunk_path[i:i+3].copy())

                if len(bad_trios) == 0:
                    last_path = (chunk_path, is_flipped_path)


                total_chunk_size = sum(chunk_length_proportion[chunk] for chunk in range(num_chunks) for seed in current_seeds_chunked[chunk] if copies_chunked[chunk, seed] is None)
                print(f"Completed chunk length: {total_chunk_size:0.3f}, maximum: {avg_num_seeds_per_chunk * base_num_chunks:0.3f}")
                if total_chunk_size >= avg_num_seeds_per_chunk * base_num_chunks:
                    print("Completed iterative process. Combining predictions and writing solution...")
                    # exit while loop
                    break
                else:
                    for chunk, seed in chunk_path:
                        if (chunk, seed) not in avg_certainties_chunked:
                            if copies_chunked[(chunk, seed)] is None:
                                prediction_window = predictions_chunked[:, predictions_start_idx_chunked[(chunk, seed)]: predictions_end_idx_chunked[(chunk, seed)]]
                            else:
                                copy_chunk, copy_seed = copies_chunked[(chunk, seed)]
                                start_idx = predictions_start_idx_chunked[(copy_chunk, copy_seed)] + idx_start_chunked[chunk] - idx_start_chunked[copy_chunk]
                                end_idx = predictions_end_idx_chunked[(copy_chunk, copy_seed)] + idx_end_chunked[chunk] - idx_end_chunked[copy_chunk]
                                prediction_window = predictions_chunked[:, start_idx: end_idx]
                            avg_certainties_chunked[(chunk, seed)] = prediction_window.amax(dim=-1).mean().item()
                        
                    for chunk in range(num_chunks):
                        for seed in current_seeds_chunked[chunk]:
                            if copies_chunked[(chunk, seed)] is None:
                                prediction_window = predictions_chunked[:, predictions_start_idx_chunked[(chunk, seed)]: predictions_end_idx_chunked[(chunk, seed)]]
                            else:
                                copy_chunk, copy_seed = copies_chunked[(chunk, seed)]
                                start_idx = predictions_start_idx_chunked[(copy_chunk, copy_seed)] + idx_start_chunked[chunk] - idx_start_chunked[copy_chunk]
                                end_idx = predictions_end_idx_chunked[(copy_chunk, copy_seed)] + idx_end_chunked[chunk] - idx_end_chunked[copy_chunk]
                                prediction_window = predictions_chunked[:, start_idx: end_idx]
                            
                        
                    choose_chunk_score = []
                    bad_trio_windows = [window for trio in bad_trios for window in trio]
                    bad_trio_windows_count = [bad_trio_windows.count(window) for window in chunk_path]

                    if last_path is None:
                        for i in range(len(chunk_path)):

                            if i == 0:
                                sim_score = 1 - sim_path[0]
                            elif i == len(chunk_path) - 1:
                                sim_score = 1 - sim_path[-1]
                            else:
                                sim_score = 1 - min(sim_path[i-1], sim_path[i])

                            chunk, seed = chunk_path[i]
                            avg_certainty_score = 1 - avg_certainties_chunked[(chunk, seed)]

                            score = bad_trio_windows_count[i]
                            score += 0.4 * sum(int(p_values_chunked[(chunk, s)] < 1e-2) for s in current_seeds_chunked[chunk])
                            score += 0.3 * sum(int(p_values_chunked[(chunk, s)] < 2e-3) for s in current_seeds_chunked[chunk])

                            score -= len(current_seeds_chunked[chunk])

                            score += 0.09 * sim_score + 0.01 * avg_certainty_score

                            choose_chunk_score.append(score)

                    else:
                        max_count = max(bad_trio_windows_count)
                        if max_count > 0:
                            bad_trio_windows_count = [count / max_count for count in bad_trio_windows_count]

                        for i in range(len(chunk_path)):
                            if i == 0:
                                sim_score = 1 - sim_path[0]
                            elif i == len(chunk_path) - 1:
                                sim_score = 1 - sim_path[-1]
                            else:
                                sim_score = 1 - min(sim_path[i-1], sim_path[i])

                            chunk, seed = chunk_path[i] #

                            avg_certainty_score = 1 - avg_certainties_chunked[(chunk, seed)]

                            # acting_num_seeds = len(current_seeds_chunked[chunk_path[i][0]]) - 0.4 * int(p_values_chunked[(chunk, seed)] < 1e-2) - 0.3 * int(p_values_chunked[(chunk, seed)] < 2e-3)
                            acting_num_seeds = len(current_seeds_chunked[chunk])
                            acting_num_seeds -= 0.4 * sum(int(p_values_chunked[(chunk, s)] < 1e-2) for s in current_seeds_chunked[chunk])
                            acting_num_seeds -= 0.3 * sum(int(p_values_chunked[(chunk, s)] < 2e-3) for s in current_seeds_chunked[chunk])

                            acting_num_seeds -= bad_trio_windows_count[i]
                            acting_num_seeds = max(0, acting_num_seeds)

                            chunk_improvement_prob = 1 / (acting_num_seeds + 1)
                            
                            choose_chunk_score.append(chunk_improvement_prob * (0.9 * sim_score + 0.1 * avg_certainty_score))

                    chosen_idx = max(list(range(len(chunk_path))), key=lambda x: choose_chunk_score[x])
                    chosen_chunk, _ = chunk_path[chosen_idx]
                    
                    new_windows.append(chosen_chunk)

            else:
                max_current_intervals = max(num_intervals_chunked.values())
                max_current_intervals += int(min_current_intervals == max_current_intervals)
                for chunk in range(num_chunks):
                    for seed in current_seeds_chunked[chunk]:
                        if num_intervals_chunked[(chunk, seed)] < max_current_intervals:
                            num_intervals_chunked[(chunk, seed)] += 1
                            num_iterations_allocated_chunked[(chunk, seed)] = num_intervals_chunked[(chunk, seed)] * window_eval_interval * chunk_length_proportion[chunk]

        predictions_chunked = torch.cat(
            [predictions_chunked[:, :input_size // 2]] +
            [predictions_chunked[:, predictions_start_idx_chunked[(chunk, seed)]: predictions_end_idx_chunked[(chunk, seed)]] 
            for chunk in range(num_chunks) 
            for seed in current_seeds_chunked[chunk] 
            if copies_chunked[(chunk, seed)] is None],
            dim=1)
        
        current_prediction_idx = input_size // 2
        for chunk in range(num_chunks):
            for seed in current_seeds_chunked[chunk]:
                if copies_chunked[(chunk, seed)] is None:
                    predictions_start_idx_chunked[(chunk, seed)] = current_prediction_idx
                    current_prediction_idx += len_seq_chunked[chunk].item()
                    predictions_end_idx_chunked[(chunk, seed)] = current_prediction_idx
        
        set_next_chunks = False
        predictions_chunked_new = []
        for chunk in new_windows.copy():
            seed = total_seeds_chunked[chunk]
            print(f"Creating new seed ({seed}) for chunk {chunk}")

            if (seed % num_seed_tolerance == 0) and (parent_chunks[chunk] is None) and (child_chunks[chunk] is None):
                set_next_chunks = True
                print(f"Breaking chunk {chunk} into chunks ", end="")
                if len(current_seeds_chunked[chunk]) == 0:
                    print(f"{chunk} and {num_chunks} (discarding original)")
                    num_chunks += 1

                    child_chunks[num_chunks - 1] = None
                    parent_chunks[num_chunks - 1] = None

                    chunk_starts = torch.cat((chunk_starts, (chunk_starts[chunk] * 2/3 + chunk_ends[chunk] * 1/3).unsqueeze(0)), dim=0)
                    chunk_ends = torch.cat((chunk_ends, (chunk_ends[chunk]).unsqueeze(0)), dim=0)
                    chunk_ends[chunk] = chunk_starts[chunk] * 1/3 + chunk_ends[chunk] * 2/3

                    idx_end_chunked = torch.cat((idx_end_chunked, torch.zeros((1,), dtype=torch.long, device=device)))
                    idx_end_chunked[-1] = idx_end_chunked[chunk]
                    idx_end_chunked[chunk] = torch.searchsorted(positions_chunked[chunk][input_size // 2: -(input_size // 2)], chunk_ends[chunk]) + idx_start_chunked[chunk]

                    idx_start_chunked = torch.cat((idx_start_chunked, torch.zeros((1,), dtype=torch.long, device=device)))
                    idx_start_chunked[-1] = torch.searchsorted(positions_chunked[chunk][input_size // 2: -(input_size // 2)], chunk_starts[-1]) + idx_start_chunked[chunk]

                    len_seq_chunked = idx_end_chunked - idx_start_chunked

                    padding = torch.full((input_size // 2,), float("inf"), device=device)
                    positions_chunked.append(torch.cat((padding, positions_chunked[chunk][-len_seq_chunked[-1] - (input_size // 2):]), dim=0))
                    positions_chunked[chunk] = torch.cat((positions_chunked[chunk][:input_size // 2 + len_seq_chunked[chunk]], padding), dim=0)

                    chunk_length_proportion[chunk] *= 2/3
                    chunk_length_proportion.append(chunk_length_proportion[chunk])

                    with torch.random.fork_rng():
                        torch.manual_seed(-1 + seed_offset)
                        randperm = torch.cat((torch.randperm(seed % num_individuals), torch.randperm(num_individuals - (seed % num_individuals))), dim=0).long()
                        rand_starts_chunked.append(rand_starts_chunked[chunk][randperm])

                    new_windows.append(num_chunks - 1)
                    current_seeds_chunked.append([])
                    total_seeds_chunked.append(seed)

                else:
                    print(f"{num_chunks} and {num_chunks + 2} (keeping original)")

                    num_chunks += 2
                    
                    child_chunks[chunk] = [num_chunks - 2, num_chunks - 1]
                    parent_chunks[num_chunks - 2] = chunk
                    parent_chunks[num_chunks - 1] = chunk
                    child_chunks[num_chunks - 2] = None
                    child_chunks[num_chunks - 1] = None

                

                    chunk_starts = torch.cat((chunk_starts, chunk_starts[chunk].unsqueeze(0), (chunk_starts[chunk] * 2/3 + chunk_ends[chunk] * 1/3).unsqueeze(0)), dim=0)
                    chunk_ends = torch.cat((chunk_ends, (chunk_starts[chunk] * 1/3 + chunk_ends[chunk] * 2/3).unsqueeze(0), chunk_ends[chunk].unsqueeze(0)), dim=0)

                    idx_end_chunked = torch.cat((idx_end_chunked, torch.zeros((2,), dtype=torch.long, device=device)))
                    idx_end_chunked[-2] = torch.searchsorted(positions_chunked[chunk][input_size // 2: -(input_size // 2)], chunk_ends[-2]) + idx_start_chunked[chunk]
                    idx_end_chunked[-1] = idx_end_chunked[chunk]

                    idx_start_chunked = torch.cat((idx_start_chunked, torch.zeros((2,), dtype=torch.long, device=device)))
                    idx_start_chunked[-2] = idx_start_chunked[chunk]
                    idx_start_chunked[-1] = torch.searchsorted(positions_chunked[chunk][input_size // 2: -(input_size // 2)], chunk_starts[-1]) + idx_start_chunked[chunk]

                    len_seq_chunked = idx_end_chunked - idx_start_chunked

                    padding = torch.full((input_size // 2,), float("inf"), device=device)
                    positions_chunked.append(torch.cat((positions_chunked[chunk][:input_size // 2 + len_seq_chunked[-2]], padding), dim=0))
                    positions_chunked.append(torch.cat((padding, positions_chunked[chunk][-len_seq_chunked[-1] - (input_size // 2):]), dim=0))

                    chunk_length_proportion.extend([chunk_length_proportion[chunk] * 2/3, chunk_length_proportion[chunk] * 2/3])

                    with torch.random.fork_rng():
                        torch.manual_seed(-1 + seed_offset)
                        randperm = torch.cat((torch.randperm(seed % num_individuals), torch.randperm(num_individuals - (seed % num_individuals))), dim=0).long()
                        rand_starts_chunked.append(rand_starts_chunked[chunk][randperm])
                        randperm = torch.cat((torch.randperm(seed % num_individuals), torch.randperm(num_individuals - (seed % num_individuals))), dim=0).long()
                        rand_starts_chunked.append(rand_starts_chunked[chunk][randperm])

                    new_windows.remove(chunk)
                    new_windows.extend([num_chunks - 2, num_chunks - 1])
                    current_seeds_chunked.extend([[], []])
                    total_seeds_chunked.extend([seed, seed])

                    for seed in current_seeds_chunked[chunk]:
                        for ch in [num_chunks - 2, num_chunks - 1]:
                            current_seeds_chunked[ch].append(seed)
                            copies_chunked[(ch, seed)] = (chunk, seed)
                            num_intervals_chunked[(ch, seed)] = num_intervals_chunked[(chunk, seed)]
                            num_iterations_current_chunked[(ch, seed)] = num_iterations_current_chunked[(chunk, seed)] * 2/3
                            num_iterations_allocated_chunked[(ch, seed)] = num_iterations_allocated_chunked[(chunk, seed)] * 2/3
                            solution_factor_chunked[(ch, seed)] = solution_factor_chunked[(chunk, seed)]
                            p_values_chunked[(ch, seed)] = p_values_chunked[(chunk, seed)]
                            has_changed_chunked[(ch, seed)] = False

                            predictions_strict_chunked[(ch, seed)] = predictions_strict_chunked[(chunk, seed)][:, idx_start_chunked[ch] - idx_start_chunked[chunk]: len_seq_chunked[chunk] - idx_end_chunked[chunk] + idx_end_chunked[ch]]
                        pairwise_sim_chunked[((num_chunks - 2, seed), (num_chunks - 1, seed))] = 0
                        is_flipped_chunked[((num_chunks - 2, seed), (num_chunks - 1, seed))] = 0

        if set_next_chunks:
            root_path = []
            leaf_path = []
            chunks_topo_pos = chunk_starts + chunk_ends
            chunks_topo_order = torch.argsort(chunks_topo_pos).tolist()
            # for chunk in chunks_topo_order:
            #     root = chunk if parent_chunks[chunk] is None else parent_chunks[chunk]
            #     if (len(root_path) > 0) and (root == root_path[-1]):
            #         continue
            #     root_path.append(root)
            #     leaves = [root] if child_chunks[root] is None else child_chunks[root]
            #     leaves = sorted(leaves, key=lambda x: chunks_topo_pos[x])
            #     leaf_path.extend(leaves)

            for chunk in chunks_topo_order:
                if parent_chunks[chunk] is not None:
                    continue
                root_path.append(chunk)
                leaves = [chunk] if child_chunks[chunk] is None else child_chunks[chunk]
                leaves = sorted(leaves, key=lambda x: chunks_topo_pos[x])
                leaf_path.extend(leaves)

            root_path = ["start node"] + root_path + ["end node"]
            leaf_path = ["start node"] + leaf_path + ["end node"]

            root_idx = {val: i for i, val in enumerate(root_path)}
            leaf_idx = {val: i for i, val in enumerate(leaf_path)}
            common_chunks = set(root_path) & set(leaf_path)
            common_chunks = sorted(list(common_chunks), key=lambda x: root_idx[x])
            
            root_idx = [root_idx[chunk] for chunk in common_chunks]
            leaf_idx = [leaf_idx[chunk] for chunk in common_chunks]

            # overlap factor between index i and i+1
            root_overlap_factor = [0] + [1 for _ in range(len(root_path) - 3)] + [0]
            leaf_overlap_factor = [0] + [1 for _ in range(len(leaf_path) - 3)] + [0]

            for i in range(len(leaf_idx) - 1):
                leaf_idx_start = leaf_idx[i]
                leaf_idx_end = leaf_idx[i+1]
                root_idx_start = root_idx[i]
                root_idx_end = root_idx[i+1]

                root_num_overlaps = root_idx_end - root_idx_start - int(root_idx_start == 0) - int(root_idx_end == len(root_path) - 1)
                leaf_num_overlaps = leaf_idx_end - leaf_idx_start - int(leaf_idx_start == 0) - int(leaf_idx_end == len(leaf_path) - 1)

                overlap_factor = root_num_overlaps if root_num_overlaps == 0 else root_num_overlaps / leaf_num_overlaps

                for j in range(leaf_idx_start, leaf_idx_end):
                    leaf_overlap_factor[j] *= overlap_factor

            overlap_factors = {}
            next_chunks = {}
            for i in range(len(root_path) - 1):
                next_chunks[root_path[i]] = {root_path[i+1]}
                overlap_factors[(root_path[i], root_path[i+1])] = root_overlap_factor[i]

            for i in range(len(leaf_path) - 1):
                if leaf_path[i] in next_chunks:
                    next_chunks[leaf_path[i]].add(leaf_path[i+1])
                else:
                    next_chunks[leaf_path[i]] = {leaf_path[i+1]}
                overlap_factors[(leaf_path[i], leaf_path[i+1])] = leaf_overlap_factor[i]

            next_chunks["end node"] = set()
        

        for chunk in new_windows:

            seed = total_seeds_chunked[chunk]

            positions_morgans_tmp = (positions_chunked[chunk][input_size // 2: -(input_size // 2)] - positions_chunked[chunk][len_seq_chunked[chunk] // 2 + input_size // 2]).abs()
            transition_aa_haploid = 0.5 + 0.5 * torch.exp(-lam * positions_morgans_tmp)

            infered_tract0 = 0.5 * transition_aa_haploid
            infered_tract2 = 0.5 - infered_tract0
                
            rand_start = rand_starts_chunked[chunk][seed % num_individuals]

            predictions_chunked_new.append(torch.empty((num_individuals, len_seq_chunked[chunk], num_classes), device=device))
            predictions_chunked_new[-1][:] = base_output 
            predictions_chunked_new[-1][rand_start, :, 0] = infered_tract0
            predictions_chunked_new[-1][rand_start, :, 2] = infered_tract2
                    
            predictions_start_idx_chunked[(chunk, seed)] = current_prediction_idx
            current_prediction_idx += len_seq_chunked[chunk].item()
            predictions_end_idx_chunked[(chunk, seed)] = current_prediction_idx

            certainty_chunked[(chunk, seed)] = torch.zeros((num_individuals, len_seq_chunked[chunk]), device=device)


            current_seeds_chunked[chunk].append(total_seeds_chunked[chunk])
            total_seeds_chunked[chunk] += 1
            num_iterations_current_chunked[(chunk, seed)] = 0
            num_iterations_allocated_chunked[(chunk, seed)] = window_eval_interval
            num_intervals_chunked[(chunk, seed)] = 1
            solution_factor_chunked[(chunk, seed)] = solution_factor_start
            copies_chunked[(chunk, seed)] = None

        predictions_chunked = torch.cat([predictions_chunked] + predictions_chunked_new + [predictions_chunked[:, :input_size // 2]], dim=1)
    
    if last_path is not None:
        chunk_path, is_flipped_path = last_path

    is_flipped_path = torch.tensor([0] + is_flipped_path, dtype=torch.int32)
    is_flipped_path = is_flipped_path.cumsum(0) % 2

    # make this more memory efficient
    predictions = torch.zeros((num_individuals, len_seq, num_classes), device=device)
    for (chunk, seed), is_flipped in zip(chunk_path, is_flipped_path):
        if copies_chunked[(chunk, seed)] is None:
            prediction_window = predictions_chunked[:, predictions_start_idx_chunked[(chunk, seed)]: predictions_end_idx_chunked[(chunk, seed)]]
        else:
            copy_chunk, copy_seed = copies_chunked[(chunk, seed)]
            start_idx = idx_start_chunked[chunk] - idx_start_chunked[copy_chunk] + predictions_start_idx_chunked[(copy_chunk, copy_seed)]
            end_idx = idx_end_chunked[chunk] - idx_end_chunked[copy_chunk] + predictions_end_idx_chunked[(copy_chunk, copy_seed)]
            
            prediction_window = predictions_chunked[:, start_idx: end_idx]

        if is_flipped:
            prediction_window = prediction_window[...,[2,1,0]]

        predictions[:, idx_start_chunked[chunk]: idx_end_chunked[chunk]] += prediction_window #* certainty_window.unsqueeze(-1)
    
    predictions /= predictions.sum(dim=-1, keepdim=True)


    # write outputs
    for admixed_sample_name, predicted_sample in zip(admixed_sample_names, predictions):
        with open(output_dir + "predictions." + admixed_sample_name + ".tsv", "w") as f: 
            f.write("# Ancestry predictions\n")
            f.write("position\t2,0\t1,1\t0,2\n")
            for base_pair, output_vector in zip(base_pair_positions, predicted_sample):
                f.write(str(base_pair.item()) + "\t" + "\t".join([str(el) for el in output_vector.tolist()]) + "\n")
    
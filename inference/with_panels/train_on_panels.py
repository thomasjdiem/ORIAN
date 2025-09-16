from shared import *
import math
import torch
import torch.nn as nn

@torch.no_grad()
def train_on_panels(gs):

    # Simulate switch points for indiviudals based on admixture time and proportion
    # admixture proportion (prop individuals ancestry 0)
    # first split of 0 means we start at ancestry 1

    print("\n\n")
    print("Simulating admixed individuals from reference panel ...")

    torch.manual_seed(gs.seed)
    num_simulations = max(900, 15 * len(gs.num_generations_values))
    
    splits_num_generations = {}
    for i in range(num_simulations):

        if i % 15 == 0:
            idx = torch.multinomial(gs.num_generations_prior_probs, num_samples=1, replacement=False)
            num_generations_iter = gs.num_generations_values[idx].item()
            lam = 2 * gs.population_size * (1 - math.exp(-(num_generations_iter - 1) / (2*gs.population_size)))
            lam0 = lam * (1 - gs.global_admixture_proportion) # lam for tracts staring at ancestry 1
            lam1 = lam * gs.global_admixture_proportion # lam for tracts starting at ancestry 0
            if num_generations_iter not in splits_num_generations:
                splits_num_generations[num_generations_iter] = []

        split_rand_idx = [0, 0]
        split_lengths = []
        split = []
        current_position = 0

        if torch.rand((1,)).item() > gs.global_admixture_proportion:
            split.append(-1)
            current_ancestry = 1
        else:
            current_ancestry = 0

        while current_position < gs.len_chrom:
            try:
                current_position += split_lengths[current_ancestry][split_rand_idx[current_ancestry]]
                split_rand_idx[current_ancestry] += 1
                current_ancestry = 1 - current_ancestry
                split.append(current_position)
            except IndexError:
                split_rand_idx = [0, 0]
                split_lengths = -(torch.rand((2, 200))).log()
                split_lengths[0] /= lam0
                split_lengths[1] /= lam1
                split_lengths = split_lengths.tolist()
        
        splits_num_generations[num_generations_iter].append(split[:-1])


    y_hap = {}
    num_generations_train_values = list(splits_num_generations.keys())
    num_generations_frequencies = []
    for ng_val in num_generations_train_values:
        y_hap_val = []
        for split in splits_num_generations[ng_val]:

            indices = torch.searchsorted(torch.tensor(split), gs.positions.cpu(), side='right')

            y_ind = (indices % 2).bool()

            y_hap_val.append(y_ind)

        num_generations_frequencies.append(len(y_hap_val))
        y_hap[ng_val] = torch.stack(y_hap_val).to(gs.device)

    num_generations_frequencies = torch.tensor(num_generations_frequencies, device="cpu", dtype=torch.float)
    num_generations_frequencies /= num_generations_frequencies.sum()
        

    gs.model.train()
    criterion = nn.CrossEntropyLoss()
        
    train_batch_size = max(1, gs.batch_size // 2)
    train_batch_size = min(4, train_batch_size)
    num_test = 320
    eval_interval = 30 * 16 // train_batch_size
    num_train_iterations = 8000 // train_batch_size if isinstance(gs.num_generations, (int, float)) else 4000 // train_batch_size
    num_train_iterations = int(num_train_iterations * gs.training_fraction)
    num_train_iterations = (num_train_iterations // eval_interval) * eval_interval

    lr_scheduler = "ReduceOnPlateau"
    if lr_scheduler == "Exponential":
        lr_start = 1e-5
        lr_end = lr_start / 100
        
        optimizer = torch.optim.NAdam(gs.model.parameters(), lr=lr_start)
        lr_factor = (lr_end / lr_start) ** (1/num_train_iterations)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_factor)
    elif lr_scheduler == "ReduceOnPlateau":
        lr_start = 2.5e-5
        lr_patience = 4
        lr_factor = 0.6
        lr_min = 5e-7
        continue_from_best_model = True

        optimizer = torch.optim.NAdam(gs.model.parameters(), lr=lr_start)
        # lr_factor = (lr_lowest / lr_start) ** (lr_patience * eval_interval/num_train_iterations)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=lr_factor, patience=lr_patience)
    elif lr_scheduler == "Cyclic":
        lr_start = 1e-5
        lr_lowest = lr_start / 75

        optimizer = torch.optim.NAdam(gs.model.parameters(), lr=lr_start)
        step_size = 5
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=lr_lowest, max_lr=lr_start, step_size_up=step_size, mode='triangular', cycle_momentum=False)
    elif lr_scheduler == "Fixed":
        lr_start = 2.3301900520010766e-06
        # num_train_iterations = 600 * 16 // train_batch_size
        optimizer = torch.optim.NAdam(gs.model.parameters(), lr=lr_start)
    else:
        assert False

    # fine tune on phased ref panels
    # hap_refs_pop0  #2 * n_refs0, len_seq
    # hap_refs_pop1  #2 * n_refs1, len_seq

    hap_refs_pop0, hap_refs_pop1 = gs.hap_refs

    def evaluate():
        with torch.set_grad_enabled(False):
            total_acc = 0
            total_loss = 0
            for iteration in range(num_test // gs.batch_size):
                torch.manual_seed(iteration + gs.seed)
                label_id = torch.multinomial(gs.label_group_frequencies, num_samples=1, replacement=True).item()
                n_refs0, n_refs1 = gs.label_groups[label_id]

                y_batch = []
                SOI_batch = []
                refs_pop0_batch = []
                refs_pop1_batch = []
                positions_batch = []
                for _ in range(gs.batch_size):

                    middle_idx = torch.randint(0, gs.len_seq, (1,)).item()

                    # choose random permutation of reference haplotypes
                    # first two from each are used to simulate admixed individual
                    perm = torch.randperm(2 * n_refs0)
                    hap_refs_pop0_example = hap_refs_pop0[perm]
                    perm = torch.randperm(2 * n_refs1)
                    hap_refs_pop1_example = hap_refs_pop1[perm]

                    # choose sites where reference individuals are not monomorphic, starting from middle idx and expanding outwards
                    left_idx = max(0, middle_idx - gs.input_size // 2 - 1)
                    refs_example_sum = hap_refs_pop0_example[2:, left_idx: middle_idx].sum(dim=0) + hap_refs_pop1_example[2:, left_idx: middle_idx].sum(dim=0)
                    mask = (refs_example_sum > 0) & (refs_example_sum < 2 * (n_refs0 + n_refs1) - 4)
                    valid_mask_left = torch.nonzero(mask) + left_idx
                    while (left_idx > 0) and (len(valid_mask_left) < gs.input_size // 2 + 1): 
                        left_idx_new = max(0, left_idx - gs.input_size // 2 - 1 + len(valid_mask_left))
                        refs_example_sum = hap_refs_pop0_example[2:, left_idx_new: left_idx].sum(dim=0) + hap_refs_pop1_example[2:, left_idx_new: left_idx].sum(dim=0)
                        mask = (refs_example_sum > 0) & (refs_example_sum < 2 * (n_refs0 + n_refs1) - 4)
                        valid_mask_left = torch.cat((torch.nonzero(mask) + left_idx_new, valid_mask_left), dim=0)
                        left_idx = left_idx_new

                    right_idx = min(gs.len_seq, middle_idx + gs.input_size // 2 + 1)
                    refs_example_sum = hap_refs_pop0_example[2:, middle_idx: right_idx].sum(dim=0) + hap_refs_pop1_example[2:, middle_idx: right_idx].sum(dim=0)
                    mask = (refs_example_sum > 0) & (refs_example_sum < 2 * (n_refs0 + n_refs1) - 4)
                    valid_mask_right = torch.nonzero(mask) + middle_idx
                    while (right_idx < gs.len_seq) and (len(valid_mask_right) < gs.input_size // 2 + 1): 
                        right_idx_new = min(gs.len_seq, right_idx + gs.input_size // 2 + 1 - len(valid_mask_right))
                        refs_example_sum = hap_refs_pop0_example[2:, right_idx: right_idx_new].sum(dim=0) + hap_refs_pop1_example[2:, right_idx: right_idx_new].sum(dim=0)
                        mask = (refs_example_sum > 0) & (refs_example_sum < 2 * (n_refs0 + n_refs1) - 4)
                        valid_mask_right = torch.cat((valid_mask_right, torch.nonzero(mask) + right_idx), dim=0)
                        right_idx = right_idx_new

                    positional_idx = torch.cat((valid_mask_left, valid_mask_right), dim=0).squeeze(-1)
                    if len(valid_mask_right) == 0:
                        left_padding = 0
                        right_padding = gs.input_size // 2
                    elif len(valid_mask_left) == 0:
                        left_padding = gs.input_size // 2
                        right_padding = 0
                    elif torch.rand((1,)).item() < 0.5:
                        positional_idx = positional_idx[1:]
                        left_padding = gs.input_size // 2 + 1 - len(valid_mask_left)
                        right_padding = gs.input_size // 2 + 1 - len(valid_mask_right)
                    else:
                        positional_idx = positional_idx[:-1]
                        left_padding = gs.input_size // 2 + 1 - len(valid_mask_left)
                        right_padding = gs.input_size // 2 + 1 - len(valid_mask_right)

                    hap_refs_pop0_example = hap_refs_pop0_example[:, positional_idx]
                    hap_refs_pop1_example = hap_refs_pop1_example[:, positional_idx]
                    positions_example = gs.positions[positional_idx]


                    hap0_refs_example_true = hap_refs_pop0_example[0]
                    hap0_refs_example_false = hap_refs_pop1_example[0]
                    hap1_refs_example_true = hap_refs_pop0_example[1]
                    hap1_refs_example_false = hap_refs_pop1_example[1]

                    idx = torch.multinomial(num_generations_frequencies, num_samples=1, replacement=False)
                    ng_val = num_generations_train_values[idx]
                    perm = torch.randperm(y_hap[ng_val].shape[0])[:2]
                    transitions = y_hap[ng_val][perm][:, positional_idx]

                    y_example = transitions[:, gs.input_size // 2 - left_padding].sum().item()

                    SOI_example = torch.full((gs.input_size,), -1, dtype=torch.int8, device=gs.device)
                    SOI_example[left_padding: gs.input_size - right_padding][~transitions[0]] = hap0_refs_example_true[~transitions[0]]
                    SOI_example[left_padding: gs.input_size - right_padding][transitions[0]] = hap0_refs_example_false[transitions[0]]
                    SOI_example[left_padding: gs.input_size - right_padding][~transitions[1]] += hap1_refs_example_true[~transitions[1]]
                    SOI_example[left_padding: gs.input_size - right_padding][transitions[1]] += hap1_refs_example_false[transitions[1]]

                    refs_pop0_example = torch.full((n_refs0, gs.input_size), -1, device=gs.device)
                    refs_pop1_example = torch.full((n_refs1, gs.input_size), -1, device=gs.device)

                    refs_pop0_example[:n_refs0 - 1, left_padding: gs.input_size - right_padding] = hap_refs_pop0_example[2::2] + hap_refs_pop0_example[3::2]
                    refs_pop0_example[n_refs0 - 1, left_padding: gs.input_size - right_padding] = hap_refs_pop0_example[2] + hap_refs_pop0_example[4]

                    refs_pop1_example[:n_refs1 - 1, left_padding: gs.input_size - right_padding] = hap_refs_pop1_example[2::2] + hap_refs_pop1_example[3::2]
                    refs_pop1_example[n_refs1 - 1, left_padding: gs.input_size - right_padding] = hap_refs_pop1_example[2] + hap_refs_pop1_example[4]

                    positions_example_padded = torch.full((gs.input_size,), float("inf"), device=gs.device)
                    positions_example_padded[left_padding: gs.input_size - right_padding] = positions_example

                    y_batch.append(y_example)
                    SOI_batch.append(SOI_example)
                    refs_pop0_batch.append(refs_pop0_example)
                    refs_pop1_batch.append(refs_pop1_example)
                    positions_batch.append(positions_example_padded)

                y_batch = torch.tensor(y_batch, device=gs.device)
                SOI_batch = torch.stack(SOI_batch)
                refs_pop0_batch = torch.stack(refs_pop0_batch)
                refs_pop1_batch = torch.stack(refs_pop1_batch)
                positions_batch = torch.stack(positions_batch)
                
                refs_batch = torch.cat((refs_pop0_batch, refs_pop1_batch), dim=1)

                ref_sim = gs.input_processor.get_inputs_training(SOI_batch, refs_batch, positions_batch, label_id)
                y_pred = gs.model(ref_sim)

                accuracy = (y_pred.argmax(dim=-1) == y_batch).sum().item()
                loss = criterion(y_pred, y_batch).item() * y_pred.shape[0]

                total_acc += accuracy
                total_loss += loss

        return total_loss / num_test, total_acc / num_test
            

    best_loss, best_acc = evaluate()
    print(f"initial eval, loss {best_loss:0.5f}, acc {best_acc}")

    best_model_state = {k: v.clone() for k, v in gs.model.state_dict().items()}
    with torch.set_grad_enabled(True):
        for iteration in range(num_train_iterations):
            torch.manual_seed(num_test // gs.batch_size + iteration + gs.seed)

            label_id = torch.multinomial(gs.label_group_frequencies, num_samples=1, replacement=True).item()
            n_refs0, n_refs1 = gs.label_groups[label_id]

            y_batch = []
            SOI_batch = []
            refs_pop0_batch = []
            refs_pop1_batch = []
            positions_batch = []
            for _ in range(train_batch_size):

                middle_idx = torch.randint(0, gs.len_seq, (1,)).item()

                # choose random permutation of reference haplotypes
                # first two from each are used to simulate admixed individual
                perm = torch.randperm(2 * n_refs0)
                hap_refs_pop0_example = hap_refs_pop0[perm]
                perm = torch.randperm(2 * n_refs1)
                hap_refs_pop1_example = hap_refs_pop1[perm]

                # choose sites where reference individuals are not monomorphic, starting from middle idx and expanding outwards
                left_idx = max(0, middle_idx - gs.input_size // 2 - 1)
                refs_example_sum = hap_refs_pop0_example[2:, left_idx: middle_idx].sum(dim=0) + hap_refs_pop1_example[2:, left_idx: middle_idx].sum(dim=0)
                mask = (refs_example_sum > 0) & (refs_example_sum < 2 * (n_refs0 + n_refs1) - 4)
                valid_mask_left = torch.nonzero(mask) + left_idx
                while (left_idx > 0) and (len(valid_mask_left) < gs.input_size // 2 + 1): 
                    left_idx_new = max(0, left_idx - gs.input_size // 2 - 1 + len(valid_mask_left))
                    refs_example_sum = hap_refs_pop0_example[2:, left_idx_new: left_idx].sum(dim=0) + hap_refs_pop1_example[2:, left_idx_new: left_idx].sum(dim=0)
                    mask = (refs_example_sum > 0) & (refs_example_sum < 2 * (n_refs0 + n_refs1) - 4)
                    valid_mask_left = torch.cat((torch.nonzero(mask) + left_idx_new, valid_mask_left), dim=0)
                    left_idx = left_idx_new

                right_idx = min(gs.len_seq, middle_idx + gs.input_size // 2 + 1)
                refs_example_sum = hap_refs_pop0_example[2:, middle_idx: right_idx].sum(dim=0) + hap_refs_pop1_example[2:, middle_idx: right_idx].sum(dim=0)
                mask = (refs_example_sum > 0) & (refs_example_sum < 2 * (n_refs0 + n_refs1) - 4)
                valid_mask_right = torch.nonzero(mask) + middle_idx
                while (right_idx < gs.len_seq) and (len(valid_mask_right) < gs.input_size // 2 + 1): 
                    right_idx_new = min(gs.len_seq, right_idx + gs.input_size // 2 + 1 - len(valid_mask_right))
                    refs_example_sum = hap_refs_pop0_example[2:, right_idx: right_idx_new].sum(dim=0) + hap_refs_pop1_example[2:, right_idx: right_idx_new].sum(dim=0)
                    mask = (refs_example_sum > 0) & (refs_example_sum < 2 * (n_refs0 + n_refs1) - 4)
                    valid_mask_right = torch.cat((valid_mask_right, torch.nonzero(mask) + right_idx), dim=0)
                    right_idx = right_idx_new

                positional_idx = torch.cat((valid_mask_left, valid_mask_right), dim=0).squeeze(-1)
                if len(valid_mask_right) == 0:
                    left_padding = 0
                    right_padding = gs.input_size // 2
                elif len(valid_mask_left) == 0:
                    left_padding = gs.input_size // 2
                    right_padding = 0
                elif torch.rand((1,)).item() < 0.5:
                    positional_idx = positional_idx[1:]
                    left_padding = gs.input_size // 2 + 1 - len(valid_mask_left)
                    right_padding = gs.input_size // 2 + 1 - len(valid_mask_right)
                else:
                    positional_idx = positional_idx[:-1]
                    left_padding = gs.input_size // 2 + 1 - len(valid_mask_left)
                    right_padding = gs.input_size // 2 + 1 - len(valid_mask_right)

                hap_refs_pop0_example = hap_refs_pop0_example[:, positional_idx]
                hap_refs_pop1_example = hap_refs_pop1_example[:, positional_idx]
                positions_example = gs.positions[positional_idx]


                hap0_refs_example_true = hap_refs_pop0_example[0]
                hap0_refs_example_false = hap_refs_pop1_example[0]
                hap1_refs_example_true = hap_refs_pop0_example[1]
                hap1_refs_example_false = hap_refs_pop1_example[1]

                idx = torch.multinomial(num_generations_frequencies, num_samples=1, replacement=False)
                ng_val = num_generations_train_values[idx]
                perm = torch.randperm(y_hap[ng_val].shape[0])[:2]
                transitions = y_hap[ng_val][perm][:, positional_idx]

                y_example = transitions[:, gs.input_size // 2 - left_padding].sum().item()

                SOI_example = torch.full((gs.input_size,), -1, dtype=torch.int8, device=gs.device)
                SOI_example[left_padding: gs.input_size - right_padding][~transitions[0]] = hap0_refs_example_true[~transitions[0]]
                SOI_example[left_padding: gs.input_size - right_padding][transitions[0]] = hap0_refs_example_false[transitions[0]]
                SOI_example[left_padding: gs.input_size - right_padding][~transitions[1]] += hap1_refs_example_true[~transitions[1]]
                SOI_example[left_padding: gs.input_size - right_padding][transitions[1]] += hap1_refs_example_false[transitions[1]]

                refs_pop0_example = torch.full((n_refs0, gs.input_size), -1, device=gs.device)
                refs_pop1_example = torch.full((n_refs1, gs.input_size), -1, device=gs.device)

                refs_pop0_example[:n_refs0 - 1, left_padding: gs.input_size - right_padding] = hap_refs_pop0_example[2::2] + hap_refs_pop0_example[3::2]
                refs_pop0_example[n_refs0 - 1, left_padding: gs.input_size - right_padding] = hap_refs_pop0_example[2] + hap_refs_pop0_example[4]

                refs_pop1_example[:n_refs1 - 1, left_padding: gs.input_size - right_padding] = hap_refs_pop1_example[2::2] + hap_refs_pop1_example[3::2]
                refs_pop1_example[n_refs1 - 1, left_padding: gs.input_size - right_padding] = hap_refs_pop1_example[2] + hap_refs_pop1_example[4]

                positions_example_padded = torch.full((gs.input_size,), float("inf"), device=gs.device)
                positions_example_padded[left_padding: gs.input_size - right_padding] = positions_example

                y_batch.append(y_example)
                SOI_batch.append(SOI_example)
                refs_pop0_batch.append(refs_pop0_example)
                refs_pop1_batch.append(refs_pop1_example)
                positions_batch.append(positions_example_padded)

            y_batch = torch.tensor(y_batch, device=gs.device)
            SOI_batch = torch.stack(SOI_batch)
            refs_pop0_batch = torch.stack(refs_pop0_batch)
            refs_pop1_batch = torch.stack(refs_pop1_batch)
            positions_batch = torch.stack(positions_batch)
            

            refs_batch = torch.cat((refs_pop0_batch, refs_pop1_batch), dim=1)

            optimizer.zero_grad()

            ref_sim = gs.input_processor.get_inputs_training(SOI_batch, refs_batch, positions_batch, label_id)
            y_pred = gs.model(ref_sim)
            loss = criterion(y_pred, y_batch)

            loss.backward()
            optimizer.step()
            if lr_scheduler in ["Exponential", "Cyclic"]:
                scheduler.step()

            if iteration % eval_interval == eval_interval - 1:
                loss, acc = evaluate()
                if lr_scheduler == "ReduceOnPlateau":
                    prev_lr = optimizer.param_groups[0]['lr']
                    scheduler.step(loss)
                    new_lr = optimizer.param_groups[0]['lr']

                    if new_lr < lr_min:
                        break

                    if continue_from_best_model and (new_lr < prev_lr):
                        gs.model.load_state_dict(best_model_state)

                print(f"iteration {iteration + 1}/{num_train_iterations}, loss {loss:0.5f}, acc {acc}, lr {optimizer.param_groups[0]['lr']}")
                if loss < best_loss:
                    best_loss = loss
                    best_model_state = {k: v.clone() for k, v in gs.model.state_dict().items()}


    gs.model.load_state_dict(best_model_state)
    gs.model.eval()

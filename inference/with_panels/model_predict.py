from shared import *
import torch
import torch.nn.functional as F

@torch.no_grad()
def model_predict(gs, set_gap=False):

    print(f"\nInferring ancestry of admixed sample {gs.adm_sample_idx + 1}/{gs.n_admixed} ...")
    num_predictions = gs.num_NN_iterations * gs.batch_size
    gs.outputs_predicted = torch.zeros((num_predictions, num_classes), dtype=torch.float, device=gs.device)
    gs.idx_sampled = torch.linspace(0.5 * gs.len_seq / num_predictions , gs.len_seq - 1 - 0.5 * gs.len_seq/num_predictions, num_predictions, dtype=torch.long)
    
    for ref_group_id in range(len(gs.ref_groups)):

        outputs_predicted_group = torch.empty((num_predictions, num_classes), dtype=torch.float, device=gs.device)
        
        valid_sites = gs.non_mono_sites[ref_group_id]
        ref_group = gs.ref_groups[ref_group_id]
        label_id = gs.label_ids[ref_group_id]
        ref_group_weight = gs.ref_group_weights[ref_group_id]

        padding = torch.full((gs.input_size // 2,), -1, device=gs.device)
        SOI_padded = torch.cat((padding, gs.SOI[valid_sites], padding), dim=0).to(torch.int8)

        padding = torch.full((gs.input_size // 2,), float("inf"), device=gs.device)
        positions_padded = torch.cat((padding, gs.positions[valid_sites], padding), dim=0)

        padding = torch.full((len(ref_group), gs.input_size // 2), -1, device=gs.device)
        refs_padded = torch.cat((padding, gs.refs[ref_group][:, valid_sites], padding), dim=-1).to(torch.int8)

        valid_sites = valid_sites.long().cumsum(dim=0) - 1
        idx_sampled = valid_sites[gs.idx_sampled]

        # make sure first site(s) for group occur after first non-monomorphic site (this should be very rare)
        for i in range(len(idx_sampled)):
            if idx_sampled[i] < 0:
                idx_sampled[i] = 0
            else:
                break

        last_idx = -gs.input_size
        for iteration in range(gs.num_NN_iterations):

            if iteration % 25 == 24:
                print(f"pass {ref_group_id + 1}/{len(gs.ref_groups)}, iteration {(iteration + 1) * gs.batch_size}/{gs.num_NN_iterations * gs.batch_size}")

            idxs = idx_sampled[iteration * gs.batch_size: (iteration + 1) * gs.batch_size]
            next_idx = None if iteration == gs.num_NN_iterations - 1 else idx_sampled[(iteration + 1) * gs.batch_size]

            positions_batch = torch.stack([positions_padded[idx:idx+gs.input_size] for idx in idxs])
            SOI_batch = SOI_padded[max(last_idx + gs.input_size, idxs[0]):idxs[-1]+gs.input_size]
            refs_batch = refs_padded[:, max(last_idx + gs.input_size, idxs[0]):idxs[-1]+gs.input_size]

            ref_sim = gs.input_processor.get_inputs_inference(SOI_batch, refs_batch, positions_batch, label_id)
            outputs_predicted_group[iteration * gs.batch_size: (iteration + 1) * gs.batch_size] = gs.model(ref_sim, idxs=idxs, next_idx=next_idx)

            last_idx = idxs[-1]

        gs.outputs_predicted += F.softmax(outputs_predicted_group, dim=-1) * ref_group_weight

    if set_gap:
        class_freq = F.one_hot(gs.outputs_predicted.argmax(dim=-1), 3).sum(dim=0)
        sampled_admixture_proportion = (class_freq.cpu().float() * torch.tensor([1, 0.5, 0])).sum().item() / (gs.num_NN_iterations * gs.batch_size)
        gs.global_admixture_proportion = (gs.global_admixture_proportion * gs.adm_sample_idx + sampled_admixture_proportion) / (gs.adm_sample_idx + 1)
        print(f"Global admixture proportion set to {gs.global_admixture_proportion:0.4f}")
        return

    if (not isinstance(gs.num_generations, (int, float))) and (gs.main_iteration < gs.num_main_iterations * gs.n_admixed):

        outputs_predicted_strict = gs.outputs_predicted.argmax(dim=-1).to(torch.int8)

        # if NN output is very similar to last one, this will be the last iteration
        if (outputs_predicted_strict == gs.last_outputs_predicted[gs.adm_sample_idx]).sum().item() / len(outputs_predicted_strict) > 0.99:
            gs.main_iteration = gs.num_main_iterations * gs.n_admixed

        gs.last_outputs_predicted[gs.adm_sample_idx] = outputs_predicted_strict

    gs.positions_predicted = gs.positions[gs.idx_sampled] # (num_predictions)

    # Merge outputs with same genetic position
    gs.positions_predicted, inverse = torch.unique(gs.positions_predicted, return_inverse=True)

    output_sums = torch.zeros((gs.positions_predicted.shape[0], 3), dtype=gs.outputs_predicted.dtype, device=gs.device)
    output_sums = output_sums.scatter_add(0, inverse[:, None].expand(-1, 3), gs.outputs_predicted)

    gs.outputs_predicted = output_sums / output_sums.sum(dim=1, keepdim=True)


    gs.positions_diff = gs.positions_predicted[1:] - gs.positions_predicted[:-1]

    class_freq = F.one_hot(gs.outputs_predicted.argmax(dim=-1), 3).sum(dim=0)
    gs.admixture_proportion = (class_freq.cpu().float() * torch.tensor([1, 0.5, 0])).sum().item() / (gs.num_NN_iterations * gs.batch_size)

    if isinstance(gs.num_generations, (int, float)) or (gs.main_iteration >= gs.num_main_iterations * gs.n_admixed):
        base_pair_positions = gs.base_pair_positions[gs.idx_sampled].tolist()
        NN_output_probs = gs.outputs_predicted.tolist()
        with open(gs.output_dir + "raw_nn_output." + gs.admixed_sample_name + ".tsv", "w") as f: # add admixed sample name to file name
            f.write("# Raw ancestry output probabilities of neural network\n")
            f.write("position\t2,0\t1,1\t0,2\n")
            for base_pair, NN_output_vector in zip(base_pair_positions, NN_output_probs):
                f.write(str(base_pair) + "\t" + "\t".join([str(el) for el in NN_output_vector]) + "\n")
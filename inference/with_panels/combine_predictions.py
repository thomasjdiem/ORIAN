from shared import * 
import math 
from scipy.stats import poisson
import torch
import torch.nn.functional as F

@torch.no_grad()
def combine_predictions(gs):

    def calculate_likelihood(outputs_predicted, lam):

        if lam == 0:
            class_probs = outputs_predicted.sum(dim=0)
            L = class_probs.logsumexp(dim=0)
            return L.item(), None, class_probs.unsqueeze(0)
        
        with torch.set_grad_enabled(True):
            lam = torch.tensor([lam], device=gs.device, dtype=torch.double, requires_grad=True)

            half = torch.tensor(0.5, dtype=torch.double, device=gs.device).log()

            transition_00_hap = torch.logaddexp(half, half + (-lam * gs.positions_diff))
            transition_01_hap = (1 - transition_00_hap.exp()).log() 

            transition_00 = transition_00_hap * 2
            transition_02 = transition_01_hap * 2
            transition_10 = transition_00_hap + transition_01_hap
            transition_01 = transition_10 + math.log(2)
            
            transitions = torch.empty((outputs_predicted.shape[0] - 1, num_classes, num_classes), dtype=torch.double, device=gs.device)
            transitions[:, 0, 0] = transition_00
            transitions[:, 0, 1] = transition_01
            transitions[:, 0, 2] = transition_02
            transitions[:, 1, 0] = transition_10
            transitions[:, 1, 1] = torch.logaddexp(transition_00, transition_02)
            transitions[:, 1, 2] = transition_10
            transitions[:, 2, 0] = transition_02
            transitions[:, 2, 1] = transition_01
            transitions[:, 2, 2] = transition_00

            L1 = torch.zeros_like(outputs_predicted)
            L1[0] = outputs_predicted[0] 
            for i in range(1, outputs_predicted.shape[0]):
                L1[i] = outputs_predicted[i] + (transitions[i - 1] + L1[i - 1].unsqueeze(0)).logsumexp(dim=-1)

            L2 = torch.zeros_like(outputs_predicted)
            L2[-1] = outputs_predicted[-1] 
            for i in range(outputs_predicted.shape[0] - 2, -1, -1):
                L2[i] = outputs_predicted[i] + (transitions[i] + L2[i + 1].unsqueeze(0)).logsumexp(dim=-1)

            solution = (L1 + L2 - outputs_predicted)
            L = (solution).logsumexp(dim=-1).mean()

            L.backward()

        return L.item(), lam.grad.item(), solution.detach()
    
    def get_solution(L, lam, solution):
        if L > MLE_lam.L_lam0:
            strict_solution = solution.argmax(dim=-1)
            strict_solution_diff = strict_solution[1:] - strict_solution[:-1]
            num_splits = strict_solution_diff.abs().sum().item()

            split_indices_predicted = torch.nonzero(strict_solution_diff).squeeze(-1)
            grouped_solution = torch.cat((strict_solution[split_indices_predicted], strict_solution[-1].unsqueeze(0)), dim=0)
            probs_left = solution[split_indices_predicted]
            probs_right = solution[split_indices_predicted + 1]
            probs_diff_left = probs_left[torch.arange(len(split_indices_predicted), dtype=torch.long), grouped_solution[:-1]] - probs_left[torch.arange(len(split_indices_predicted), dtype=torch.long), grouped_solution[1:]] 
            probs_diff_right = probs_right[torch.arange(len(split_indices_predicted), dtype=torch.long), grouped_solution[:-1]] - probs_right[torch.arange(len(split_indices_predicted), dtype=torch.long), grouped_solution[1:]] 
            positions_left = gs.positions_predicted[split_indices_predicted]
            positions_right = gs.positions_predicted[split_indices_predicted + 1]
            positions_switch_idx = positions_left - probs_diff_left * (positions_right - positions_left) / (probs_diff_right - probs_diff_left)
            switch_idx = torch.searchsorted(gs.positions, positions_switch_idx, side="right")
            switch_idx = torch.cat((torch.tensor([0], dtype=torch.long, device=gs.device), switch_idx, torch.tensor([gs.len_seq], dtype=torch.long, device=gs.device)), dim=0)

            # print("return lam", lam)
            return lam, strict_solution, num_splits, "true solution", (grouped_solution, switch_idx)
        
        else:
            strict_solution = MLE_lam.solution_lam0.argmax(dim=-1)
            # print("return null")
            return 0, strict_solution, 0, "true solution", (strict_solution, torch.tensor([0, gs.len_seq], dtype=torch.long, device=gs.device))
        
    def check_lam_low(outputs_predicted, lam_low):

        L, fpos, solution = calculate_likelihood(outputs_predicted, lam_low)
        if fpos < MLE_lam.tol:
            if fpos > -MLE_lam.tol:
                return get_solution(L, lam_low, solution)
            if MLE_lam.increasing_region is not None:
                if MLE_lam.increasing_region[1] - MLE_lam.increasing_region[0] < 0.02:
                    return get_solution(L, lam_low, solution)

                if L > MLE_lam.increasing_region[3]:
                    MLE_lam.increasing_region[1] = lam_low
                    MLE_lam.increasing_region[3] = L

                    result = test_upper_bound(solution, L, lam_low)
                    if result is not None:
                        return result

                    # fix all arithmetic avg to geometric avg
                    return MLE_lam(outputs_predicted, 
                                    lam_low=(MLE_lam.increasing_region[0] + lam_low)/2, 
                                    lam_high=lam_low,
                                    priority_low=1, priority_high=0,
                                    recursive_call=True)
                
                elif L < MLE_lam.increasing_region[2]:
                    MLE_lam.increasing_region[0] = lam_low
                    MLE_lam.increasing_region[2] = L

                    result = test_lower_bound(solution, L, lam_low)
                    if result is not None:
                        return result

                    return MLE_lam(outputs_predicted, 
                                    lam_low=(MLE_lam.increasing_region[1] + lam_low)/2 , 
                                    lam_high=MLE_lam.increasing_region[1], 
                                    priority_low=1, priority_high=0, 
                                    recursive_call=True)
                
                else:
                    assert False, "Multiple modes found"


            if MLE_lam.insert_idx == 0:
                L_lower = MLE_lam.L_lam0
                lam_lower = 0
            else:
                L_lower = MLE_lam.Ls[MLE_lam.insert_idx - 1]
                lam_lower = MLE_lam.lams[MLE_lam.insert_idx - 1]

            if L > L_lower:
                if lam_low - lam_lower < 0.01:
                    return get_solution(L, lam_low, solution)
                    
                MLE_lam.increasing_region = [lam_lower, lam_low, L_lower, L]
                return MLE_lam(outputs_predicted, 
                                lam_low=(lam_lower + lam_low)/2, 
                                lam_high=lam_low, 
                                priority_low=1, priority_high=0, 
                                recursive_call=True)

            
            if MLE_lam.insert_idx == len(MLE_lam.lams):
                L_higher = float("-inf")
            else:
                L_higher = MLE_lam.Ls[MLE_lam.insert_idx]
                lam_higher = MLE_lam.lams[MLE_lam.insert_idx]

            if L < L_higher:
                if lam_higher - lam_low < 0.01:
                    return get_solution(L, lam_low, solution)
                
                MLE_lam.increasing_region = [lam_low, lam_higher, L, L_higher]
                return MLE_lam(outputs_predicted, 
                                lam_low=(lam_low + lam_higher) / 2, 
                                lam_high=lam_higher, 
                                priority_low=1, priority_high=0, 
                                recursive_call=True)

            if len(MLE_lam.lams) == 5:
                strict_solution = MLE_lam.solution_lam0.argmax(dim=-1)
                return 0, strict_solution, 0, "true solution", (strict_solution, torch.tensor([0, gs.len_seq], dtype=torch.long, device=gs.device))


            MLE_lam.lams.insert(MLE_lam.insert_idx, lam_low)
            MLE_lam.mus.insert(MLE_lam.insert_idx, math.exp(-lam_low))
            MLE_lam.Ls.insert(MLE_lam.insert_idx, L)
            MLE_lam.fs.insert(MLE_lam.insert_idx, fpos)

            mus_extended = [1] + MLE_lam.mus + [0]
            MLE_lam.insert_idx = max(list(range(len(MLE_lam.lams) + 1)), key=lambda i: mus_extended[i] - mus_extended[i+1])

            mu = (mus_extended[MLE_lam.insert_idx] + mus_extended[MLE_lam.insert_idx+1]) / 2
            lam = -math.log(mu)
            

            if MLE_lam.insert_idx == len(MLE_lam.lams):
                return MLE_lam(outputs_predicted,
                                lam_low=lam,
                                lam_high=2 * lam,
                                priority_low=2, priority_high=1, 
                                recursive_call=True)
        
            else:
                return MLE_lam(outputs_predicted,
                                lam_low=lam,
                                lam_high=MLE_lam.lams[MLE_lam.insert_idx],
                                priority_low=1, priority_high=0,
                                recursive_call=True)
            
        return None
    
    def check_lam_high(outputs_predicted, lam_high):
        L, fneg, solution = calculate_likelihood(outputs_predicted, lam_high)

        if fneg > -MLE_lam.tol:
            if fneg < MLE_lam.tol:
                return get_solution(L, lam_high, solution)
            
            result = test_lower_bound(solution, L, lam_high)
            if result is not None:
                return result
            
            return MLE_lam(outputs_predicted, 
                            lam_low=lam_high, 
                            lam_high=lam_high * 2, 
                            priority_low=0, priority_high=1,
                            recursive_call=True)
                            
        MLE_lam.lams.append(lam_high)
        MLE_lam.mus.append(math.exp(-lam_high))
        MLE_lam.Ls.append(L)
        MLE_lam.fs.append(fneg)

        return None
    
    def test_lower_bound(solution, L, lam):
        # if lower bound for lam has too many split points, then we deduce that the solution has too many split points
        if MLE_lam.test_lower_bound and (MLE_lam.L_lam0 < L or MLE_lam.min_num_splits > 0):
            strict_solution = solution.argmax(dim=-1)
            num_splits = (strict_solution[1:] - strict_solution[:-1]).abs().sum().item()
            if num_splits > MLE_lam.max_num_splits:
                # print(f"the lower bound for λ ({lam}) predicts {num_splits} switch points which is too large to be sufficiently likely")
                return lam, strict_solution, num_splits, "upper bound", None
            if num_splits > MLE_lam.min_num_splits:
                MLE_lam.test_upper_bound = False

        return None 

    def test_upper_bound(solution, L, lam):
        # if upper bound for lam has too few split points, then we deduce that the solution has too few split points
        if MLE_lam.test_upper_bound:
            strict_solution = solution.argmax(dim=-1)
            num_splits = (strict_solution[1:] - strict_solution[:-1]).abs().sum().item()
            if num_splits < MLE_lam.min_num_splits:
                # print(f"the new upper bound for λ ({lam}) predicts {num_splits} switch points which is too small to be sufficiently likely")
                return lam, strict_solution, num_splits, "lower bound", None
            if num_splits < MLE_lam.max_num_splits:
                MLE_lam.test_lower_bound = False

        return None 
            

    def MLE_lam(outputs_predicted, lam_low=1e-3, lam_high=1500, priority_low=1, priority_high=2, recursive_call=False):

        #maximum likelihood occcurs either when lam==0 or when lam.grad == 0

        lam_low = max(lam_low, 1e-3)

        # print("lam_low", lam_low, ", lam_high", lam_high)

        if not recursive_call:
            MLE_lam.test_lower_bound = (MLE_lam.max_num_splits < MLE_lam.outputs_num_splits)
            MLE_lam.test_upper_bound = (MLE_lam.min_num_splits > 0)
            MLE_lam.increasing_region = None
            MLE_lam.insert_idx = 0
            MLE_lam.lams = []
            MLE_lam.mus = []
            MLE_lam.Ls = []
            MLE_lam.fs = []
            MLE_lam.L_lam0, _, MLE_lam.solution_lam0 = calculate_likelihood(outputs_predicted, 0)

        # print(MLE_lam.lams, MLE_lam.mus, MLE_lam.Ls, MLE_lam.fs)
        # print("increasing region params", MLE_lam.increasing_region)
        # print("priorities", priority_low, priority_high)

        if ((lam_low == lam_high) and (priority_low == priority_high == 0)):
            L, _, solution = calculate_likelihood(outputs_predicted, lam_low)
            return get_solution(L, lam_low, solution)


        if priority_low > priority_high:
            
            result_low = check_lam_low(outputs_predicted, lam_low)
            if result_low is not None:
                return result_low

            if priority_high > 0:
                result_high = check_lam_high(outputs_predicted, lam_high)
                if result_high is not None:
                    return result_high
                

        else:   
            result_high = check_lam_high(outputs_predicted, lam_high)
            if result_high is not None:
                return result_high

            if priority_low > 0:
                result_low = check_lam_low(outputs_predicted, lam_low)
                if result_low is not None:
                    return result_low


        for _ in range(100):
            # print("bounds found, lam_low", lam_low, ", lam_high", lam_high)

            lam_new = ((lam_low + 1) * (lam_high + 1)) ** 0.5 - 1

            L, fnew, solution = calculate_likelihood(outputs_predicted, lam_new)

            if abs(fnew) < MLE_lam.tol / max(lam_high - lam_low, 1):
                return get_solution(L, lam_new, solution)
            
            if fnew > 0:
                result = test_lower_bound(solution, L, lam_new)
                if result is not None:
                    return result
                lam_low = lam_new

            else:
                result = test_upper_bound(solution, L, lam_new)
                if result is not None:
                    return result
                lam_high = lam_new

        raise ValueError("MLE did not converge")
        
    
    def calculate_lambdaf_probs(num_split_probs_posterior):

        lambda_f = 4 * (gs.num_generations_values - 1) * gs.len_chrom * gs.admixture_proportion * (1 - gs.admixture_proportion) 
        lambda_f *= (-0.5 * (gs.num_generations_values - 1) / gs.population_size).exp()

        ps = poisson.pmf(torch.arange(2000), lambda_f.unsqueeze(-1).cpu())
        ps = torch.tensor(ps, device=gs.device)
        ps = (ps * num_split_probs_posterior).sum(dim=-1) 
        ps /= ps.sum()

        return ps
    
    num_split_probs = {}
    def calculate_num_split_probs(solution_prob_threshold):

        lambda_f = 4 * (gs.num_generations_values - 1) * gs.len_chrom * gs.admixture_proportion * (1 - gs.admixture_proportion) 
        lambda_f *= (-0.5 * (gs.num_generations_values - 1) / gs.population_size).exp()


        ps = poisson.pmf(torch.arange(2000), lambda_f.unsqueeze(-1).cpu())
        
        ps = torch.tensor(ps, device=gs.device)
        ps = (ps * gs.num_generations_prior_probs.unsqueeze(-1)).sum(dim=0)
        threshold_p = ps.amax().item() * solution_prob_threshold
        for i, p in enumerate(ps):
            if p > threshold_p:
                num_split_probs[i] = p.item()

        MLE_lam.min_num_splits = min(num_split_probs.keys())
        MLE_lam.max_num_splits = max(num_split_probs.keys())

        return ps
    

    def iterate_solutions():
        

        print("\nNumerically integrating over possible solutions ...")
        
        lambda_f = 2 * (gs.num_generations_values - 1) * max(gs.admixture_proportion, 1 - gs.admixture_proportion) * (-0.5 * (gs.num_generations_values - 1) / gs.population_size).exp()
        lambda_f *= gs.num_generations_prior_probs

        exponent_variation1 = torch.zeros((gs.outputs_predicted.shape[0],), dtype=torch.float, device=gs.device)
        exponent_variation1[0] = 1 
        for i in range(1, gs.outputs_predicted.shape[0]):
            exponent_variation1[i] = exponent_variation1[i - 1] * ((-lambda_f * gs.positions_diff[i - 1]).exp() * gs.num_generations_prior_probs).sum() + 1

        exponent_variation2 = torch.zeros((gs.outputs_predicted.shape[0],), dtype=torch.float, device=gs.device)
        exponent_variation2[-1] = 1 
        for i in range(gs.outputs_predicted.shape[0] - 2, -1, -1):
            exponent_variation2[i] = exponent_variation2[i + 1] * ((-lambda_f * gs.positions_diff[i]).exp() * gs.num_generations_prior_probs).sum() + 1

        exponent_variation = exponent_variation1 + exponent_variation2 - 1
        exponent_variation = exponent_variation.log() 
        exponent_variation -= exponent_variation.mean()
        exponent_variation = 1 / exponent_variation.exp().unsqueeze(-1)


        MLE_lam.tol = 1e-3
        solution_total_weight_threshold = 1e-8
        solution_prob_threshold = 0.01
        exponent_limit = 256

        num_splits_prior_probs = calculate_num_split_probs(solution_prob_threshold)
        # between 15 and 32
        x1, y1 = 1, 15
        x2, y2 = 1000, 32
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1
        num_values = min(gs.num_generations_values.amax() - gs.num_generations_values.amin() + 1, 1000)
        max_num_predictions = slope * num_values + intercept

        avg_certainty_list = []
        solution_list = []
        solution_prob_list = []
        num_splits_list = []
        exponent_list = []
        lam_list = []
        full_solution_compact_list = []

        max_avg_certainty = 1
        max_solution = None
        max_exp = 100
        max_lam = 1500
        max_est = False
        max_full_solution_compact = None

        min_avg_certainty = 1/num_classes
        min_solution = None
        min_exp = 0.001
        min_lam = 0
        min_est = False
        min_full_solution_compact = None

        outputs_strict = gs.outputs_predicted.argmax(dim=-1)
        MLE_lam.outputs_num_splits = (outputs_strict[1:] - outputs_strict[:-1]).abs().sum().item()
        
            

        if MLE_lam.min_num_splits == 0:
            min_est = True
            min_exp = 1/exponent_limit
            outputs_predicted_tested = outputs_predicted_log * (min_exp ** exponent_variation)
            outputs_predicted_tested -= outputs_predicted_tested.logsumexp(dim=-1, keepdim=True)
            min_avg_certainty = outputs_predicted_tested.amax(dim=-1).exp().mean().item()

            print(f"α: {min_exp:0.3f}, c(α): {min_avg_certainty:0.3f}, ", end="")
            min_lam, min_solution, min_num_splits, result, min_full_solution_compact = MLE_lam(
                                                                outputs_predicted_tested,
                                                                lam_low=min_lam,
                                                                lam_high=max_lam)
            if result == "lower bound":
                print(f"the upper bound for λ ({min_lam:0.3f}) predicts {min_num_splits} switch points which is too small to be sufficiently probable")
            elif result == "upper bound":
                print(f"the lower bound for λ ({min_lam:0.3f}) predicts {min_num_splits} switch points which is too large to be sufficiently probable")
            else:
                print(f"λ: {min_lam:0.3f}, switch points: {min_num_splits}")

            
            if result == "upper bound":

                # should be very rare
                positions_left = torch.zeros((gs.len_seq,), dtype=torch.float, device=gs.device)
                positions_right = torch.zeros((gs.len_seq,), dtype=torch.float, device=gs.device)
                outputs_log_left = torch.zeros((gs.len_seq, num_classes), dtype=torch.float, device=gs.device)
                outputs_log_right = torch.zeros((gs.len_seq, num_classes), dtype=torch.float, device=gs.device)

                positions_left[:gs.idx_sampled[0]] = -1
                outputs_log_left[:gs.idx_sampled[0]] = outputs_predicted_log[0]
                positions_right[:gs.idx_sampled[0] + 1] = gs.positions[gs.idx_sampled[0]]
                outputs_log_right[:gs.idx_sampled[0] + 1] = outputs_predicted_log[0]
                for i in range(len(gs.idx_sampled) - 1):
                    positions_left[gs.idx_sampled[i]:gs.idx_sampled[i+1]] = gs.positions[gs.idx_sampled[i]]
                    positions_right[gs.idx_sampled[i] + 1:gs.idx_sampled[i+1] + 1] = gs.positions[gs.idx_sampled[i+1]]
                    outputs_log_left[gs.idx_sampled[i]:gs.idx_sampled[i+1]] = outputs_predicted_log[i]
                    outputs_log_right[gs.idx_sampled[i] + 1:gs.idx_sampled[i+1] + 1] = outputs_predicted_log[i+1]
                positions_left[gs.idx_sampled[-1]:] = gs.positions[gs.idx_sampled[-1]]
                outputs_log_left[gs.idx_sampled[-1]:] = outputs_predicted_log[-1]
                positions_right[gs.idx_sampled[-1] + 1:] = -1
                outputs_log_right[gs.idx_sampled[-1] + 1:] = outputs_predicted_log[-1]

                combined_outputs = (outputs_log_right - outputs_log_left) * (gs.positions - positions_left).unsqueeze(-1) / (positions_right - positions_left).unsqueeze(-1) + outputs_log_left
                combined_outputs[gs.idx_sampled] = outputs_predicted_log.float()

                with open(gs.output_dir + "predictions." + gs.admixed_sample_name + ".tsv", "w") as f:
                    f.write("# Ancestry predictions\n")
                    f.write("position\t2,0\t1,1\t0,2\n")
                    for base_pair, output_vector in zip(gs.base_pair_positions, combined_outputs):
                        f.write(str(base_pair.item()) + "\t" + "\t".join([str(el) for el in output_vector.tolist()]) + "\n")

                return True
            
            min_solution_prob = num_split_probs[min_num_splits]
            

        if MLE_lam.max_num_splits >= MLE_lam.outputs_num_splits:
            max_est = True
            max_exp = exponent_limit
            outputs_predicted_tested = outputs_predicted_log * (max_exp ** exponent_variation)
            outputs_predicted_tested -= outputs_predicted_tested.logsumexp(dim=-1, keepdim=True)
            max_avg_certainty = outputs_predicted_tested.amax(dim=-1).exp().mean().item()

            print(f"α: {max_exp:0.3f}, c(α): {max_avg_certainty:0.3f}, ", end="")
            max_lam, max_solution, max_num_splits, result, max_full_solution_compact = MLE_lam(
                                                                outputs_predicted_tested,
                                                                lam_low=min_lam,
                                                                lam_high=max_lam)
            if result == "lower bound":
                print(f"the upper bound for λ ({max_lam:0.3f}) predicts {max_num_splits} switch points which is too small to be sufficiently probable")
            elif result == "upper bound":
                print(f"the lower bound for λ ({max_lam:0.3f}) predicts {max_num_splits} switch points which is too large to be sufficiently probable")
            else:
                print(f"λ: {max_lam:0.3f}, switch points: {max_num_splits}")
            
            if result == "lower bound":

                positions_left = torch.zeros((gs.len_seq,), dtype=torch.float, device=gs.device)
                positions_right = torch.zeros((gs.len_seq,), dtype=torch.float, device=gs.device)
                outputs_log_left = torch.zeros((gs.len_seq, num_classes), dtype=torch.float, device=gs.device)
                outputs_log_right = torch.zeros((gs.len_seq, num_classes), dtype=torch.float, device=gs.device)

                positions_left[:gs.idx_sampled[0]] = -1
                outputs_log_left[:gs.idx_sampled[0]] = outputs_predicted_log[0]
                positions_right[:gs.idx_sampled[0] + 1] = gs.positions[gs.idx_sampled[0]]
                outputs_log_right[:gs.idx_sampled[0] + 1] = outputs_predicted_log[0]
                for i in range(len(gs.idx_sampled) - 1):
                    positions_left[gs.idx_sampled[i]:gs.idx_sampled[i+1]] = gs.positions[gs.idx_sampled[i]]
                    positions_right[gs.idx_sampled[i] + 1:gs.idx_sampled[i+1] + 1] = gs.positions[gs.idx_sampled[i+1]]
                    outputs_log_left[gs.idx_sampled[i]:gs.idx_sampled[i+1]] = outputs_predicted_log[i]
                    outputs_log_right[gs.idx_sampled[i] + 1:gs.idx_sampled[i+1] + 1] = outputs_predicted_log[i+1]
                positions_left[gs.idx_sampled[-1]:] = gs.positions[gs.idx_sampled[-1]]
                outputs_log_left[gs.idx_sampled[-1]:] = outputs_predicted_log[-1]
                positions_right[gs.idx_sampled[-1] + 1:] = -1
                outputs_log_right[gs.idx_sampled[-1] + 1:] = outputs_predicted_log[-1]

                combined_outputs = (outputs_log_right - outputs_log_left) * (gs.positions - positions_left).unsqueeze(-1) / (positions_right - positions_left).unsqueeze(-1) + outputs_log_left
                combined_outputs[gs.idx_sampled] = outputs_predicted_log.float()

                with open(gs.output_dir + "predictions." + gs.admixed_sample_name + ".tsv", "w") as f:
                    f.write("# Ancestry predictions\n")
                    f.write("position\t2,0\t1,1\t0,2\n")
                    for base_pair, output_vector in zip(gs.base_pair_positions, combined_outputs):
                        f.write(str(base_pair.item()) + "\t" + "\t".join([str(el) for el in output_vector.tolist()]) + "\n")

                return True

            max_solution_prob = num_split_probs[max_num_splits]


        if (not min_est) or (not max_est):
            exponent = 1
            outputs_predicted_tested = outputs_predicted_log * (exponent ** exponent_variation)
            outputs_predicted_tested -= outputs_predicted_tested.logsumexp(dim=-1, keepdim=True)

            avg_certainty = outputs_predicted_tested.amax(dim=-1).exp().mean().item()


            print(f"α: {exponent:0.3f}, c(α): {avg_certainty:0.3f}, ", end="")
            lam, strict_solution, num_splits, result, full_solution_compact = MLE_lam(
                                                        outputs_predicted_tested,
                                                        lam_low=min_lam,
                                                        lam_high=max_lam)
            if result == "lower bound":
                print(f"the upper bound for λ ({lam:0.3f}) predicts {num_splits} switch points which is too small to be sufficiently probable")
            elif result == "upper bound":
                print(f"the lower bound for λ ({lam:0.3f}) predicts {num_splits} switch points which is too large to be sufficiently probable")
            else:
                print(f"λ: {lam:0.3f}, switch points: {num_splits}")


            solution_prob = num_split_probs.get(num_splits, 0)
            if (avg_certainty < 1/num_classes + solution_total_weight_threshold) or (num_splits < MLE_lam.min_num_splits):
                result = "lower bound"

            elif (avg_certainty > 1 - solution_total_weight_threshold) or (num_splits > MLE_lam.max_num_splits):
                result = "upper bound"
            

            if result == "lower bound":
                
                min_est = True
                min_lam = lam
                min_exp = exponent
                min_avg_certainty = avg_certainty
                min_solution = strict_solution
                min_solution_prob = solution_prob
                min_num_splits = num_splits
                min_full_solution_compact = full_solution_compact

                while not max_est:

                    exponent *= 2
                    outputs_predicted_tested = outputs_predicted_log * (exponent ** exponent_variation)
                    outputs_predicted_tested -= outputs_predicted_tested.logsumexp(dim=-1, keepdim=True)

                    avg_certainty = outputs_predicted_tested.amax(dim=-1).exp().mean().item()

                    print(f"α: {exponent:0.3f}, c(α): {avg_certainty:0.3f}, ", end="")
                    lam, strict_solution, num_splits, result, full_solution_compact = MLE_lam(
                                                                    outputs_predicted_tested,
                                                                    lam_low=min_lam,
                                                                    lam_high=max_lam)
                    if result == "lower bound":
                        print(f"the upper bound for λ ({lam:0.3f}) predicts {num_splits} switch points which is too small to be sufficiently probable")
                    elif result == "upper bound":
                        print(f"the lower bound for λ ({lam:0.3f}) predicts {num_splits} switch points which is too large to be sufficiently probable")
                    else:
                        print(f"λ: {lam:0.3f}, switch points: {num_splits}")
                    
                    if result == "true solution":

                        solution_prob = num_split_probs.get(num_splits, 0)
                        if (exponent <= 1/exponent_limit) or (avg_certainty < 1/num_classes + solution_total_weight_threshold) or (num_splits < MLE_lam.min_num_splits):
                            min_exp = exponent
                            min_lam = lam
                            min_avg_certainty = avg_certainty
                            min_solution = strict_solution
                            min_solution_prob = solution_prob
                            min_num_splits = num_splits
                            min_full_solution_compact = full_solution_compact

                        elif (exponent >= exponent_limit) or (avg_certainty > 1 - solution_total_weight_threshold) or (num_splits > MLE_lam.max_num_splits):
                            max_est = True
                            max_exp = exponent
                            max_lam = lam
                            max_avg_certainty = avg_certainty
                            max_solution = strict_solution
                            max_solution_prob = solution_prob
                            max_num_splits = num_splits
                            max_full_solution_compact = full_solution_compact
                            break

                        else:
                            exponent_list.append(exponent)
                            avg_certainty_list.append(avg_certainty)
                            solution_list.append(strict_solution)
                            num_splits_list.append(num_splits)
                            solution_prob_list.append(solution_prob)
                            lam_list.append(lam)
                            full_solution_compact_list.append(full_solution_compact)
                            break

                    elif result == "upper bound":
                        max_lam = lam
                        max_exp = exponent
                        max_avg_certainty = avg_certainty
                        max_est = True
                        max_solution = strict_solution
                        max_solution_prob = 0
                        max_num_splits = num_splits
                        max_full_solution_compact = full_solution_compact
                        break

                    else:
                        min_lam = lam
                        min_exp = exponent
                        min_avg_certainty = avg_certainty
                        min_solution = strict_solution
                        min_solution_prob = 0
                        min_num_splits = num_splits
                        min_full_solution_compact = full_solution_compact

            elif result == "upper bound":

                max_est = True
                max_lam = lam
                max_exp = exponent
                max_avg_certainty = avg_certainty
                max_solution = strict_solution
                max_solution_prob = solution_prob
                max_num_splits = num_splits
                max_full_solution_compact = full_solution_compact

                while not min_est:

                    exponent /= 2
                    outputs_predicted_tested = outputs_predicted_log * (exponent ** exponent_variation)
                    outputs_predicted_tested -= outputs_predicted_tested.logsumexp(dim=-1, keepdim=True)

                    avg_certainty = outputs_predicted_tested.amax(dim=-1).exp().mean().item()

                    print(f"α: {exponent:0.3f}, c(α): {avg_certainty:0.3f}, ", end="")
                    lam, strict_solution, num_splits, result, full_solution_compact = MLE_lam(
                                                                    outputs_predicted_tested,
                                                                    lam_low=min_lam,
                                                                    lam_high=max_lam)
                    if result == "lower bound":
                        print(f"the upper bound for λ ({lam:0.3f}) predicts {num_splits} switch points which is too small to be sufficiently probable")
                    elif result == "upper bound":
                        print(f"the lower bound for λ ({lam:0.3f}) predicts {num_splits} switch points which is too large to be sufficiently probable")
                    else:
                        print(f"λ: {lam:0.3f}, switch points: {num_splits}")
                    

                    if result == "true solution":

                        solution_prob = num_split_probs.get(num_splits, 0)
                        if (exponent <= 1/exponent_limit) or (avg_certainty < 1/num_classes + solution_total_weight_threshold) or (num_splits < MLE_lam.min_num_splits):
                            min_est = True
                            min_exp = exponent
                            min_lam = lam
                            min_avg_certainty = avg_certainty
                            min_solution = strict_solution
                            min_solution_prob = solution_prob
                            min_num_splits = num_splits
                            min_full_solution_compact = full_solution_compact
                            break

                        elif (exponent >= exponent_limit) or (avg_certainty > 1 - solution_total_weight_threshold) or (num_splits > MLE_lam.max_num_splits):
                            max_exp = exponent
                            max_lam = lam
                            max_avg_certainty = avg_certainty
                            max_solution = strict_solution
                            max_solution_prob = solution_prob
                            max_num_splits = num_splits
                            max_full_solution_compact = full_solution_compact

                        else:
                            exponent_list.append(exponent)
                            avg_certainty_list.append(avg_certainty)
                            solution_list.append(strict_solution)
                            num_splits_list.append(num_splits)
                            solution_prob_list.append(solution_prob)
                            lam_list.append(lam)
                            full_solution_compact_list.append(full_solution_compact)
                            break

                    elif result == "lower bound":
                        min_est = True
                        min_lam = lam
                        min_exp = exponent
                        min_avg_certainty = avg_certainty
                        min_solution = strict_solution
                        min_solution_prob = 0
                        min_num_splits = num_splits
                        min_full_solution_compact = full_solution_compact
                        break

                    else:
                        max_lam = lam
                        max_exp = exponent
                        max_avg_certainty = avg_certainty
                        max_solution = strict_solution
                        max_solution_prob = 0
                        max_num_splits = num_splits
                        max_full_solution_compact = full_solution_compact

            else:
                exponent_list.append(exponent)
                avg_certainty_list.append(avg_certainty)
                solution_list.append(strict_solution)
                num_splits_list.append(num_splits)
                solution_prob_list.append(solution_prob)
                lam_list.append(lam)
                full_solution_compact_list.append(full_solution_compact)

        while not max_est:
            exponent = max(exponent_list) * 2
            outputs_predicted_tested = outputs_predicted_log * (exponent ** exponent_variation)
            outputs_predicted_tested -= outputs_predicted_tested.logsumexp(dim=-1, keepdim=True)

            avg_certainty = outputs_predicted_tested.amax(dim=-1).exp().mean().item()

            print(f"α: {exponent:0.3f}, c(α): {avg_certainty:0.3f}, ", end="")
            lam, strict_solution, num_splits, result, full_solution_compact = MLE_lam(outputs_predicted_tested, 
                                                                lam_low=max(lam_list),
                                                                lam_high=max_lam)
            if result == "lower bound":
                print(f"the upper bound for λ ({lam:0.3f}) predicts {num_splits} switch points which is too small to be sufficiently probable")
            elif result == "upper bound":
                print(f"the lower bound for λ ({lam:0.3f}) predicts {num_splits} switch points which is too large to be sufficiently probable")
            else:
                print(f"λ: {lam:0.3f}, switch points: {num_splits}")


            if result == "upper bound":
                max_est = True  
                max_lam = lam
                max_exp = exponent
                max_avg_certainty = avg_certainty
                max_solution = strict_solution
                max_solution_prob = 0
                max_num_splits = num_splits
                max_full_solution_compact = full_solution_compact
            elif (exponent >= exponent_limit) or (avg_certainty > 1 - solution_total_weight_threshold) or (num_splits > MLE_lam.max_num_splits):
                solution_prob = num_split_probs.get(num_splits, 0)
                max_est = True
                max_lam = lam
                max_exp = exponent
                max_avg_certainty = avg_certainty
                max_solution = strict_solution
                max_solution_prob = solution_prob
                max_num_splits = num_splits
                max_full_solution_compact = full_solution_compact
            else:
                solution_prob = num_split_probs.get(num_splits, 0)
                exponent_list.append(exponent)
                avg_certainty_list.append(avg_certainty)
                solution_list.append(strict_solution)
                num_splits_list.append(num_splits)
                solution_prob_list.append(solution_prob)
                lam_list.append(lam)
                full_solution_compact_list.append(full_solution_compact)



        while not min_est:
            exponent = min(exponent_list) / 2
            outputs_predicted_tested = outputs_predicted_log * (exponent ** exponent_variation)
            outputs_predicted_tested -= outputs_predicted_tested.logsumexp(dim=-1, keepdim=True)

            avg_certainty = outputs_predicted_tested.amax(dim=-1).exp().mean().item()

            print(f"α: {exponent:0.3f}, c(α): {avg_certainty:0.3f}, ", end="")
            lam, strict_solution, num_splits, result, full_solution_compact = MLE_lam(outputs_predicted_tested, 
                                                                lam_low=min_lam,
                                                                lam_high=min(lam_list))
            if result == "lower bound":
                print(f"the upper bound for λ ({lam:0.3f}) predicts {num_splits} switch points which is too small to be sufficiently probable")
            elif result == "upper bound":
                print(f"the lower bound for λ ({lam:0.3f}) predicts {num_splits} switch points which is too large to be sufficiently probable")
            else:
                print(f"λ: {lam:0.3f}, switch points: {num_splits}")



            if result == "lower bound":
                min_est = True 
                min_lam = lam
                min_exp = exponent
                min_avg_certainty = avg_certainty
                min_solution = strict_solution
                min_solution_prob = 0
                min_num_splits = num_splits
                min_full_solution_compact = full_solution_compact
            elif (exponent <= 1/exponent_limit) or (avg_certainty < 1/num_classes + solution_total_weight_threshold) or (num_splits < MLE_lam.min_num_splits):
                solution_prob = num_split_probs.get(num_splits, 0)
                min_est = True
                min_lam = lam
                min_exp = exponent
                min_avg_certainty = avg_certainty
                min_solution = strict_solution
                min_solution_prob = solution_prob
                min_num_splits = num_splits
                min_full_solution_compact = full_solution_compact
            else:
                solution_prob = num_split_probs.get(num_splits, 0)
                exponent_list.append(exponent)
                avg_certainty_list.append(avg_certainty)
                solution_list.append(strict_solution)
                num_splits_list.append(num_splits)
                solution_prob_list.append(solution_prob)
                lam_list.append(lam)
                full_solution_compact_list.append(full_solution_compact)


        def get_expected_solution_prob(num_splits1, num_splits2):
            num_splits1, num_splits2 = sorted([num_splits1, num_splits2])
            return max(num_split_probs.get(n, 0) for n in range(num_splits1, num_splits2 + 1))
        

        exponent_list.extend([min_exp, max_exp])
        avg_certainty_list.extend([min_avg_certainty, max_avg_certainty])
        solution_list.extend([min_solution, max_solution])
        solution_prob_list.extend([min_solution_prob, max_solution_prob])
        num_splits_list.extend([min_num_splits, max_num_splits])
        lam_list.extend([min_lam, max_lam])
        full_solution_compact_list.extend([min_full_solution_compact, max_full_solution_compact])


        num_predictions = len(exponent_list)
        sorted_idx = sorted(list(range(num_predictions)), key=lambda i: exponent_list[i])
        exponent_list = [exponent_list[idx] for idx in sorted_idx]
        avg_certainty_list = [avg_certainty_list[idx] for idx in sorted_idx]
        solution_list = [solution_list[idx] for idx in sorted_idx]
        solution_prob_list = [solution_prob_list[idx] for idx in sorted_idx]
        num_splits_list = [num_splits_list[idx] for idx in sorted_idx]
        lam_list = [lam_list[idx] for idx in sorted_idx]
        full_solution_compact_list = [full_solution_compact_list[idx] for idx in sorted_idx]

        solution_diff = [(solution_list[i] != solution_list[i + 1]).sum().item() / len(solution_list[i]) for i in range(num_predictions - 1)]
        avg_certainty_diff = [avg_certainty_list[i+1] - avg_certainty_list[i] for i in range(num_predictions - 1)]
        expected_solution_prob = [get_expected_solution_prob(num_splits_list[i], num_splits_list[i+1]) for i in range(num_predictions - 1)]

        while num_predictions < max_num_predictions:

            # print()
            # print()
            # print()
            # print("upper and lower bound found")
            # print("exponents", exponent_list)
            # print("avg certainty", avg_certainty_list)
            # print("solution prob", solution_prob_list)
            # print("num_splits", num_splits_list)
            # print("lam values", lam_list)

            # print()
            # print("solution diff", solution_diff)
            # print("avg certainty diff", avg_certainty_diff)
            # print("expected solution prob", expected_solution_prob)


            idx_max = max(list(range(num_predictions - 1)), key=lambda i: (solution_diff[i] ** 0.25) * avg_certainty_diff[i] * expected_solution_prob[i])
            
            # print()
            # print("chosen idx", idx_max)
            
            if solution_diff[idx_max] * avg_certainty_diff[idx_max] * expected_solution_prob[idx_max] < solution_total_weight_threshold:
                break
            

            exponent = (exponent_list[idx_max] + exponent_list[idx_max + 1]) / 2
            
            outputs_predicted_tested = outputs_predicted_log * (exponent ** exponent_variation)
            outputs_predicted_tested -= outputs_predicted_tested.logsumexp(dim=-1, keepdim=True)

            avg_certainty = outputs_predicted_tested.amax(dim=-1).exp().mean().item()



            print(f"α: {exponent:0.3f}, c(α): {avg_certainty:0.3f}, ", end="")
            lam, strict_solution, num_splits, result, full_solution_compact = MLE_lam(outputs_predicted_tested, 
                                                                lam_low=lam_list[idx_max],
                                                                lam_high=lam_list[idx_max+1])
            if result == "lower bound":
                print(f"the upper bound for λ ({lam:0.3f}) predicts {num_splits} switch points which is too small to be sufficiently probable")
            elif result == "upper bound":
                print(f"the lower bound for λ ({lam:0.3f}) predicts {num_splits} switch points which is too large to be sufficiently probable")
            else:
                print(f"λ: {lam:0.3f}, switch points: {num_splits}")


            
            solution_prob = num_split_probs.get(num_splits, 0)
                
            if ((result == "lower bound") or (avg_certainty < 1/num_classes + solution_total_weight_threshold) or (num_splits < MLE_lam.min_num_splits)) \
                and (solution_prob_list[idx_max] == 0) and (idx_max == 0):
                # replace first
                exponent_list[0] = exponent
                avg_certainty_list[0] = avg_certainty
                solution_list[0] = strict_solution
                num_splits_list[0] = num_splits
                solution_prob_list[0] = solution_prob
                lam_list[0] = lam
                full_solution_compact_list[0] = full_solution_compact

                solution_diff[0] = (solution_list[1] != strict_solution).sum().item() / len(strict_solution)
                avg_certainty_diff[0] = avg_certainty_list[1] - avg_certainty
                expected_solution_prob[0] = get_expected_solution_prob(num_splits_list[idx_max + 1], num_splits)

            elif ((result == "upper bound") or (avg_certainty > 1 - solution_total_weight_threshold) or (num_splits > MLE_lam.max_num_splits)) \
                and (solution_prob_list[idx_max] == 0) and (idx_max == num_predictions - 2):
                # replace last
                exponent_list[-1] = exponent
                avg_certainty_list[-1] = avg_certainty
                solution_list[-1] = strict_solution
                num_splits_list[-1] = num_splits
                solution_prob_list[-1] = solution_prob
                lam_list[-1] = lam
                full_solution_compact_list[-1] = full_solution_compact

                solution_diff[-1] = (solution_list[-2] != strict_solution).sum().item() / len(strict_solution)
                avg_certainty_diff[-1] = avg_certainty - avg_certainty_list[-2]
                expected_solution_prob[-1] = get_expected_solution_prob(num_splits_list[-2], num_splits)

            else:
                # insert
                solution_diff[idx_max] = (solution_list[idx_max] != strict_solution).sum().item() / len(strict_solution)
                solution_diff.insert(idx_max + 1, (solution_list[idx_max + 1] != strict_solution).sum().item() / len(strict_solution))

                avg_certainty_diff[idx_max] = avg_certainty - avg_certainty_list[idx_max]
                avg_certainty_diff.insert(idx_max + 1, avg_certainty_list[idx_max + 1] - avg_certainty)

                expected_solution_prob[idx_max] = get_expected_solution_prob(num_splits_list[idx_max], num_splits) 
                expected_solution_prob.insert(idx_max + 1, get_expected_solution_prob(num_splits_list[idx_max + 1], num_splits))

                exponent_list.insert(idx_max + 1, exponent)
                avg_certainty_list.insert(idx_max + 1, avg_certainty)
                solution_list.insert(idx_max + 1, strict_solution)
                num_splits_list.insert(idx_max + 1, num_splits)
                solution_prob_list.insert(idx_max + 1, solution_prob)
                lam_list.insert(idx_max + 1, lam)
                full_solution_compact_list.insert(idx_max + 1, full_solution_compact)

            num_predictions = len(exponent_list)


        if not isinstance(gs.num_generations, (int, float)):
            num_split_probs_posterior = torch.zeros((2000,), device=gs.device)
            for i in range(1, num_predictions - 1):
                num_split_probs_posterior[num_splits_list[i]] += ((avg_certainty_list[i + 1] - avg_certainty_list[i - 1]) / 2)

            num_split_probs_posterior[num_splits_list[0]] += (avg_certainty_list[1] - avg_certainty_list[0]) / 2
            num_split_probs_posterior[num_splits_list[-1]] += (avg_certainty_list[-1] - avg_certainty_list[-2]) / 2

            torch.set_printoptions(threshold=2001)

            # print("nspost", num_split_probs_posterior[:200])

            # num_predictions generations distribution given evidence and uniform num_predictions split distribution (and no prior)
            num_generations_freq_probs = calculate_lambdaf_probs(num_split_probs_posterior)

            num_generations_round_trip_probs = calculate_lambdaf_probs(num_splits_prior_probs)

            # print("ngfreq", num_generations_freq_probs)
            # print("ngrt", num_generations_round_trip_probs)
            gs.num_generations_post_probs = num_generations_freq_probs * gs.num_generations_prior_probs / num_generations_round_trip_probs
            gs.num_generations_post_probs /= gs.num_generations_post_probs.sum()
            # print("ngpost", gs.num_generations_post_probs)

            expected_value = (gs.num_generations_post_probs * gs.num_generations_values).sum().item()
            cum_probs = gs.num_generations_post_probs.cumsum(dim=0)
            
            idx = torch.searchsorted(cum_probs, 0.05, side="left")
            confidence_low = gs.num_generations_values[idx].item()
            
            idx = torch.searchsorted(cum_probs, 0.95, side="right")
            confidence_high = gs.num_generations_values[idx].item()

            print(f"Updated distribution of admixture time - expected value {expected_value:0.2f} and 90% confidence interval [{confidence_low}, {confidence_high}]")

            if gs.main_iteration < gs.num_main_iterations * gs.n_admixed:
                gs.main_iteration += 1
                gs.num_generations_prior_probs = gs.num_generations_post_probs.float()

                return False
            
            with open(gs.output_dir + "admixture_time.tsv", "w") as f:
                f.write("# Posterior distribution for admixture time (number of generations)\n")
                f.write("Admixture time\tProbability\n")
                for value, prob in zip(gs.num_generations_values.tolist(), gs.num_generations_post_probs.tolist()):
                    f.write(str(value) + "\t" + str(prob) + "\n")



        combined_outputs = torch.zeros((gs.len_seq, num_classes), device=gs.device, dtype=torch.float)
        for i in range(1, num_predictions - 1):
            solution_weight = solution_prob_list[i] * (avg_certainty_list[i + 1] - avg_certainty_list[i - 1]) / 2
            if solution_weight == 0:
                continue
            full_solution_values, full_solution_switch_idx = full_solution_compact_list[i]
            full_solution_values = F.one_hot(full_solution_values.long(), num_classes=num_classes).to(gs.device).float()
            full_solution_values *= solution_weight
            for j in range(full_solution_values.shape[0]):
                combined_outputs[full_solution_switch_idx[j]:full_solution_switch_idx[j+1]] += full_solution_values[j].unsqueeze(0)
            
        solution_weight = solution_prob_list[0] * (avg_certainty_list[1] - avg_certainty_list[0]) / 2
        if solution_weight > 0:
            full_solution_values, full_solution_switch_idx = full_solution_compact_list[0]
            full_solution_values = F.one_hot(full_solution_values.long(), num_classes=num_classes).to(gs.device).float()
            full_solution_values *= solution_weight
            for j in range(full_solution_values.shape[0]):
                combined_outputs[full_solution_switch_idx[j]:full_solution_switch_idx[j+1]] += full_solution_values[j].unsqueeze(0)

        solution_weight = solution_prob_list[-1] * (avg_certainty_list[-1] - avg_certainty_list[-2]) / 2
        if solution_weight > 0:
            full_solution_values, full_solution_switch_idx = full_solution_compact_list[-1]
            full_solution_values = F.one_hot(full_solution_values.long(), num_classes=num_classes).to(gs.device).float()
            full_solution_values *= solution_weight
            for j in range(full_solution_values.shape[0]):
                combined_outputs[full_solution_switch_idx[j]:full_solution_switch_idx[j+1]] += full_solution_values[j].unsqueeze(0)

        combined_outputs /= combined_outputs[0].sum()

        with open(gs.output_dir + "predictions." + gs.admixed_sample_name + ".tsv", "w") as f:
            f.write("# Ancestry predictions\n")
            f.write("position\t2,0\t1,1\t0,2\n")
            for base_pair, output_vector in zip(gs.base_pair_positions, combined_outputs):
                f.write(str(base_pair.item()) + "\t" + "\t".join([str(el) for el in output_vector.tolist()]) + "\n")

        return True
        
    outputs_predicted_log = gs.outputs_predicted.double().log()
    gs.solution_found[gs.adm_sample_idx] = iterate_solutions()

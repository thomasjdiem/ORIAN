from shared import *
from initialize_model import initialize_variables, initialize_model
from train_on_panels import train_on_panels
from model_predict import model_predict
from combine_predictions import combine_predictions
import torch
import os
import allel
import numpy as np
import gzip
import argparse

class GlobalState:
    def __init__(self):

        parser = argparse.ArgumentParser(description="Local Ancestry Inference Tool")
        parser.add_argument("--ref-vcf", required=True, type=str, help="Reference panel vcf")   
        parser.add_argument("--adm-vcf", required=True, type=str, help="Admixed vcf") 
        parser.add_argument("--sample-file", required=True, type=str, help="Sample file")
        group = parser.add_mutually_exclusive_group(required=True)
        group.add_argument("--recomb-rate", type=float, help="Constant recombination rate throughout the chromosome (M/bp)")
        group.add_argument("--recomb-map-file", type=str, help="Recombination map file")
        group = parser.add_mutually_exclusive_group(required=True)
        group.add_argument("--admix-time", type=float, help="Admixture time in number of generations. -1 if not known with any certainty.  Prior distribution can be specified with --admix-time-file option")
        group.add_argument("--admix-time-file", type=str, help="Prior distribution of admixture time in number of generations")
        parser.add_argument("--admix-prop", required=True, type=float, help="Admixture proportion, where larger value indicates higher proportion of ancestry 0, or -1 if unknown")
        parser.add_argument("--population-size", type=float, default=10_000, help="Effective population size")
        parser.add_argument("--omit-training", action="store_true", help="Omit fine tuning the neural network on data from the (computationally phased) reference panel")
        parser.add_argument("--training-fraction", type=float, default=1.0, help="Number of training iterations in units of default value")
        parser.add_argument("--num-NN-predict", type=int, default=4000, help="Number of neural network predictions during inference")
        parser.add_argument("--batch-size", type=int, default=16, help="Inference batch size. Training batch size will be half of this in most cases")
        parser.add_argument("--device", type=str, help="Device to run program on (cpu or gpu). Will default to gpu if available, otherwise cpu")
        parser.add_argument("--seed", type=int, default=0, help="Random seed (only relevant for fine tuning on reference panel)")
        parser.add_argument("--out", required=True, type=str, help="Output directory")

        args = parser.parse_args()

        if args.batch_size < 1:
            raise RuntimeError("Batch size must be at least 1")
        self.batch_size = args.batch_size

        if args.num_NN_predict < 100:
            raise RuntimeError("Number of neural network predictions must be at least 100")
        self.num_NN_iterations = args.num_NN_predict // self.batch_size

        self.seed = args.seed

        if args.training_fraction < 0:
            raise RuntimeError("Training fraction must be at least 0")
        self.training_fraction = args.training_fraction

        if args.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        elif args.device.strip().lower() == "cpu":
            self.device = "cpu"
        elif args.device.strip().lower() in ["gpu", "cuda"]:
            if not torch.cuda.is_available():
                raise RuntimeError("Device set to gpu but torch.cuda.is_available() returns False")
            self.device = "cuda"
        else:
            raise RuntimeError("Invalid device name.  Must be 'cpu' or 'gpu'")


        # Process sample file
        sample_names = []
        ref_ancestries = []
        with open(args.sample_file, "r") as f:
            for sample_file_line in f.readlines():
                if sample_file_line.isspace():
                    continue
                sample_name, ancestry = sample_file_line.strip().split("\t")
                sample_names.append(sample_name)
                ref_ancestries.append(ancestry)

        ref_ancestries_unique = sorted(list(set(ref_ancestries)))
        if (ref_ancestries_unique[0] not in ["0", "1"]) or (ref_ancestries_unique[1] not in ["0", "1"]):
            raise RuntimeError(f"Ancestries values in sample file must be either '0' or '1'. Detected {len(ref_ancestries_unique)} ancestries: {','.join(ref_ancestries_unique)}")
                               
        self.n_refs0 = ref_ancestries.count('0')
        self.n_refs1 = ref_ancestries.count('1')
        self.n_refs = len(ref_ancestries)
        if (self.n_refs0 < min_num_refs_each) or (self.n_refs1 < min_num_refs_each) or (self.n_refs < min_num_refs_total):
            raise RuntimeError("Must be at least 7 reference individuals per ancestry and at least 18 total.")
        
        sample_to_ancestry = {s: int(a) for s, a in zip(sample_names, ref_ancestries)}
            


        # Process ref vcf
        callset = allel.read_vcf(args.ref_vcf, fields='*')

        ref_sample_names = list(callset["samples"])
        if sorted(sample_names) != sorted(ref_sample_names):
            raise RuntimeError("Sample names from sample file and ref file do not match.")

        chromosome_array = callset['variants/CHROM']
        ref_chrom = chromosome_array[0]
        if not np.all(chromosome_array == ref_chrom):
            raise RuntimeError("Multiple chromosomes detected in ref vcf")
        
        hap_refs = torch.tensor(callset["calldata/GT"], device=self.device, dtype=torch.int8)

        if (hap_refs == -1).any():
            raise RuntimeError("Missing alleles detected in reference VCF genotype data.")
        
        alt_alleles = callset["variants/ALT"]
        biallelic_mask_refs = torch.tensor([sum(int(a != "") for a in alt) < 2 for alt in alt_alleles], device=self.device, dtype=torch.bool)


        positions_bp_refs = torch.tensor(callset["variants/POS"], device=self.device, dtype=torch.long)

        labels = 2 * torch.tensor([sample_to_ancestry[sample] for sample in ref_sample_names], device=self.device, dtype=torch.int8)
        self.labels, sample_idx = torch.sort(labels)

        hap_refs = hap_refs[:, sample_idx]


        def is_vcf_phased(vcf_path, max_variants=1000):
            """
            Checks whether the VCF is phased by searching for '|' in genotype fields.
            Only reads the first `max_variants` variants for speed.
            """
            open_fn = gzip.open if vcf_path.endswith(".gz") else open
            with open_fn(vcf_path, 'rt') as f:
                for line in f:
                    if line.isspace():
                        continue
                    if line.startswith('#'):
                        continue
                    fields = line.strip().split('\t')
                    genotypes = fields[-self.n_refs:]
                    for gt in genotypes:
                        # We only check the actual genotype string, before any ":" (e.g. "0|1:35")
                        if '|' not in gt.split(':')[0]:
                            return False
                        
                    max_variants -= 1
                    if max_variants == 0:
                        break
            return True
        
        hap_refs = hap_refs.reshape(-1, 2 * self.n_refs).t()
        self.refs = hap_refs[::2] + hap_refs[1::2]
        
        if (args.omit_training) or (self.training_fraction == 0):
            self.hap_refs = None
        else:
            if not is_vcf_phased(args.ref_vcf):
                raise RuntimeError("Ref vcf must be computationally phased if panel training is enabled.")
            
            self.hap_refs = (hap_refs[:2 * self.n_refs0], hap_refs[2 * self.n_refs0:])



        # Process admixed vcf
        callset = allel.read_vcf(args.adm_vcf, fields='*')

        self.admixed_sample_names = list(callset["samples"])

        hap_admixed = torch.tensor(callset["calldata/GT"], device=self.device, dtype=torch.int8)
        
        if (hap_admixed == -1).any():
            raise RuntimeError("Missing alleles detected in admixed VCF genotype data.")


        self.admixed = hap_admixed.sum(dim=-1).t()
        self.n_admixed = self.admixed.shape[0]

        positions_bp_admixed = torch.tensor(callset["variants/POS"], device=self.device, dtype=torch.long)

        alt_alleles = callset["variants/ALT"]
        biallelic_mask_adm = torch.tensor([sum(int(a != "") for a in alt) < 2 for alt in alt_alleles], device=self.device, dtype=torch.bool)

        if not torch.equal(positions_bp_admixed, positions_bp_refs):
            raise RuntimeError("Base pair positions in admixed and ref vcf do not match.")
        
        chromosome_array = callset['variants/CHROM']
        adm_chrom = chromosome_array[0]
        if not np.all(chromosome_array == adm_chrom):
            raise RuntimeError("Multiple chromosomes detected in admixed vcf")
        
        if adm_chrom != ref_chrom:
            raise RuntimeError("Chromosomes from ref and admixed vcf do not match")

        biallelic_mask = biallelic_mask_adm & biallelic_mask_refs

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

            recomb_positions_bp = torch.tensor(recomb_positions_bp, dtype=torch.long, device=self.device)


            if not torch.equal(recomb_positions_bp, positions_bp_refs):
                raise RuntimeError("Base pair positions in vcfs and recombination map file do not match.")
            
            self.positions = torch.tensor(recomb_positions_morgans, dtype=torch.float, device=self.device)
            
            if torch.any(self.positions < 0):
                raise RuntimeError("Negative distances present in recombination map file.")
            
            self.positions = self.positions.cumsum(dim=0) - self.positions[0]
            self.base_pair_positions = positions_bp_refs

        elif (args.recomb_rate is not None) and (args.recomb_map_file is None):

            self.positions = positions_bp_refs.float() * args.recomb_rate
            self.positions = self.positions - self.positions[0]
            self.base_pair_positions = positions_bp_refs

        else:
            raise RuntimeError("Exactly one of recombination rate and recombination map file must be specified.")


        if (args.admix_time is not None) and (args.admix_time_file is None):

            if (args.admix_time == -1):
                self.num_generations = None 
            else:
                if args.admix_time < 1:
                    raise RuntimeError("Admixture time must be at least 1, or -1 if unknown")
                
                self.num_generations = args.admix_time
        
        elif (args.admix_time is None) and (args.admix_time_file is not None):

            num_generations_values = []
            num_generations_prior_probs = []
            with open(args.admix_time_file) as f:
                for num_generations_file_line in f.readlines():
                    if num_generations_file_line.isspace():
                        continue
                    num_generations_val, num_generations_prior_prob = num_generations_file_line.strip().split("\t")
                    
                    num_generations_val = float(num_generations_val)
                    num_generations_prior_prob = float(num_generations_prior_prob)

                    num_generations_values.append(num_generations_val)
                    num_generations_prior_probs.append(num_generations_prior_prob)

            self.num_generations = (num_generations_values, num_generations_prior_probs)

        else:
            
            raise RuntimeError("Exactly one of num generations and num generations file must be specified.")


        if args.admix_prop == -1:
            self.global_admixture_proportion = None
        else:
            if not (0 < args.admix_prop < 1):
                raise ValueError("Admixture proportion must be between 0 and 1 (strict) if known, or -1 if unknown")
        
            self.global_admixture_proportion = args.admix_prop

        self.population_size = args.population_size

        self.output_dir = args.out
        self.output_dir = os.getcwd() + "/" + self.output_dir
        if not self.output_dir.endswith("/"):
            self.output_dir += "/"

        os.makedirs(self.output_dir, exist_ok=True)


        print("\n")
        print("Loaded inputs into memory ...")

            
        self.len_chrom = self.positions[-1].item()


        refs_sum = self.refs.sum(dim=0)
        non_monomorphic_mask = (refs_sum > 0) & (refs_sum < 2 * self.refs.shape[0])
        print(f"Pruning {len(self.positions) - non_monomorphic_mask.sum().item()} SNPs with invariant reference alleles and {len(self.positions) - biallelic_mask.sum().item()} SNPs with multiple alt alleles")
        site_mask = non_monomorphic_mask & biallelic_mask

        self.admixed = self.admixed[:, site_mask]
        self.positions = self.positions[site_mask]
        self.base_pair_positions = self.base_pair_positions[site_mask]
        self.refs = self.refs[:, site_mask]
        if self.hap_refs is not None:
            self.hap_refs = (self.hap_refs[0][:, site_mask], self.hap_refs[1][:, site_mask])

        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        
        assert self.admixed.shape[1] == self.positions.shape[0] == self.base_pair_positions.shape[0] == self.refs.shape[1]
        assert self.refs.shape[0] == self.labels.shape[0]

        self.n_refs, self.len_seq = self.refs.shape

        self.last_outputs_predicted = [-1 for _ in range(self.n_admixed)]

        self.main_iteration = 0
        self.num_main_iterations = 6
        self.solution_found = [False for _ in range(self.n_admixed)]

        print(f"Number of usable SNPs: {len(self.positions)}")
        print(f"Chromosome length (morgans): {self.len_chrom:0.3f}")
        print(f"Number of reference individuals for population 0: {self.n_refs0}")
        print(f"Number of reference individuals for population 1: {self.n_refs1}")
        print(f"Number of admixed individuals: {self.n_admixed}")

        info = "unknown" if self.global_admixture_proportion is None else self.global_admixture_proportion
        print(f"Admixture fraction for population 0: {info}")

        if self.num_generations is None:
            info = "unknown (will assume uniform prior from 1 to 1000)"
        elif isinstance(self.num_generations, tuple):
            expected_value = sum(v * p for v, p in zip(*self.num_generations)) / sum(self.num_generations[1])
            info = f"prior distribution with expected value {expected_value:0.2f}"
        elif isinstance(self.num_generations, (int, float)):
            info = self.num_generations

        print(f"Time since admixture (generations): {info}")

        print(f"Effective population size: {self.population_size}")
        print(f"Device: {self.device}")
        print(f"Output directory: {self.output_dir}")





gs = GlobalState()

initialize_variables(gs)

if gs.global_admixture_proportion is None:
    gs.global_admixture_proportion = 0.5
    
    for adm_sample_idx in range(gs.n_admixed):
        gs.adm_sample_idx = adm_sample_idx
        gs.SOI = gs.admixed[gs.adm_sample_idx]

        initialize_model(gs)
        model_predict(gs, set_gap=True)


while not all(gs.solution_found):

    initialize_model(gs)

    if gs.hap_refs is not None:
        train_on_panels(gs)

    for adm_sample_idx in range(gs.n_admixed):
        gs.adm_sample_idx = adm_sample_idx
        gs.admixed_sample_name = gs.admixed_sample_names[gs.adm_sample_idx]
        gs.SOI = gs.admixed[gs.adm_sample_idx]

        model_predict(gs)
        combine_predictions(gs)

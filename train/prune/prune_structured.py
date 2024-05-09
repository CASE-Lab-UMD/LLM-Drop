import argparse
import os 
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from importlib.metadata import version
import pdb
from time import time
import datetime
# from lib.modelling_llama_mod import LlamaForCausalLM
# from lib.eval import eval_ppl, eval_ppl_trainonly
from collections import defaultdict
import pickle as pkl
import random
from lib.scoring_model import ScoreModelHP
import wandb
from transformers.pytorch_utils import  find_pruneable_heads_and_indices, prune_linear_layer
import gc
import random

print('torch', version('torch'))
print('transformers', version('transformers'))
print('accelerate', version('accelerate'))
print('# of gpus: ', torch.cuda.device_count())

INF = 1e8

# Set the mask for current mask to investigate via forward passses 
def set_masks(module_map, all_masks, all_sampling_proba, pfrac=0.1, mlp_attn_ratio=1.0, use_complement=False):
	for k, (name, module) in module_map.items():
		this_pfrac = pfrac
		if name.endswith('self_attn'):
			this_pfrac = pfrac * mlp_attn_ratio

		module.is_using_main = False
		sampling_proba, fixed_indices, use_indices = all_sampling_proba[k]
		if use_complement:
			module.temp_mask = 1 - module.temp_mask
			module.temp_mask[:, :, fixed_indices] = 1.0
			all_masks[k].append(module.temp_mask.cpu().squeeze()[use_indices])
		else:
			mask = get_random_mask(module.main_mask.numel(), module.main_mask, sampling_proba, this_pfrac)
			module.temp_mask = torch.Tensor(mask).type(module.main_mask.type())
			all_masks[k].append(torch.Tensor(mask).squeeze()[use_indices])

# get a random mask
def get_random_mask(intermediate_sz, main_mask, sampling_proba, pfrac):
	init_set = np.ones((1, 1, intermediate_sz)) if main_mask is None else main_mask.cpu().numpy()
	num_to_zero = int(pfrac * np.sum(init_set)) + 1
	non_zero_idxs = np.squeeze(init_set).nonzero()[0]
	new_proba = sampling_proba[non_zero_idxs]
	new_proba = new_proba / np.sum(new_proba)
	chosen_idxs = np.random.choice(non_zero_idxs, size=num_to_zero, p=new_proba, replace=False)
	init_set[:, :, chosen_idxs] = 0
	return init_set

# Instantiate a virtual sub-model and run forward passes through it to get the performance of the sub-model
def get_random_mask_scores(model, tokenizer, module_map, all_sampling_proba, bsz=12, nsamples=32, mpi=100, pfrac=0.1, mlp_attn_ratio=1.0, dataset_="wikitext2"):

	# set to use main
	for k, (name, module) in module_map.items():
		module.is_using_main = False

	all_masks, all_perfs = defaultdict(list), defaultdict(list)
	seed_ = random.randint(0, 199999)
	for iter_ in range(mpi // 2):
		this_bsz = bsz

		# set the layer mask here
		set_masks(module_map, all_masks, all_sampling_proba, pfrac=pfrac, mlp_attn_ratio=mlp_attn_ratio)

		# Doing this in case the batch_size we have specified is too large for a forwar pass
		# Revert to a forward pass with batch size of 1
		try:
			this_ppl = eval_ppl_trainonly(model, tokenizer, bsz=this_bsz, nsamples=nsamples, seed=seed_, dataset=dataset_)
		except Exception as e:
			print(e)
			gc.collect()
			torch.cuda.empty_cache()
			this_bsz = 1
			this_ppl = eval_ppl_trainonly(model, tokenizer, bsz=this_bsz, nsamples=nsamples, seed=seed_, dataset=dataset_)

		print('[v1]Iter : ', iter_, ' PPL = ', this_ppl)
		this_ppl = this_ppl if this_ppl < INF else INF

		for k, (name, module) in module_map.items():
			# NB : since we want the co-efficient to be more positive for more useful modules, we input -ppl
			all_perfs[k].append(-this_ppl)

		# set the complement mask here
		set_masks(module_map, all_masks, all_sampling_proba, pfrac=pfrac, mlp_attn_ratio=mlp_attn_ratio, use_complement=True)
		this_ppl = eval_ppl_trainonly(model, tokenizer, bsz=this_bsz, nsamples=nsamples, seed=seed_, dataset=dataset_)

		print('[v2]Iter : ', iter_, ' PPL = ', this_ppl)
		this_ppl = this_ppl if this_ppl < INF else INF

		for k, (name, module) in module_map.items():
			# NB : since we want the co-efficient to be more positive for more useful modules, we input -ppl
			all_perfs[k].append(-this_ppl)

	# reset to use main
	for k, (name, module) in module_map.items():
		module.is_using_main = True

	return all_masks, all_perfs

# For getting the LLM
def get_llm(model_name, cache_dir="llm_weights"):
	model = LlamaForCausalLM.from_pretrained(
		model_name, 
		torch_dtype=torch.float16, 
		cache_dir=cache_dir, 
		low_cpu_mem_usage=True, 
		device_map="auto"
	)
	model.seqlen = model.config.max_position_embeddings 
	if ('13b' in model_name) or ('65b' in model_name):
		model.seqlen = 2048 #Based on the values from the Lora-prune paper
	return model

# Hook function for caching the activations during running of the model
def hook_fn(module_name, info_cache):
	def hook(module, in_, out_):
		flat_in = (module.intermed_cache).clone().detach().float()
		module.intermed_cache = None
		if 'in' not in info_cache[module_name]:
			info_cache[module_name]['in'] = [1, flat_in]
		else:
			info_cache[module_name]['in'] = [
				info_cache[module_name]['in'][0] + 1,  
				info_cache[module_name]['in'][1].add_(flat_in)
			]
	return hook


# Convert the dataset of (sub-model, eval_performance) to a regression problem and get the module importances
def get_score_models(score_perfs, module_map, info_cache, hp_dict, wandb_run, all_sampling_proba, parent_id='.',  model_type='local'):
	score_map = {}
	if model_type == 'local':
		for id_, (name, module) in module_map.items():
			# Get a score map
			_, _, use_indices = all_sampling_proba[id_]
			base_mask = info_cache[name]['in'][1] / info_cache[name]['in'][0]
			base_mask = (base_mask.squeeze() * module.main_mask.squeeze().float())[use_indices]
			base_mask = (base_mask / base_mask.sum()).view(-1, 1)

			sm_hp_searcher = ScoreModelHP(
				id_='{}/{}'.format(parent_id, id_), num_players=score_perfs[0][id_][0].numel(),
				base_mask=base_mask, hp_dict=hp_dict, wandb=wandb_run)

			run_info = score_perfs[0][id_], score_perfs[1][id_]

			sm_hp_searcher.search_best_linear_fit(run_info)
			score_map[id_] = sm_hp_searcher.get_best_fit()
	else:
		# do some global modelling here
		# aggregate all the data here
		xs = None
		for id_, (name, module) in module_map.items():
			if xs is None:
				xs = score_perfs[0][id_]
			else:
				xs = [torch.cat((xs[k], score_perfs[0][id_][k])) for k in range(len(xs))]
		xs = [k.cuda() for k in xs]
		ys = score_perfs[1][id_]
		sm_hp_searcher = ScoreModelHP(
				id_='{}/{}'.format(parent_id, "Global"), num_players=xs[0].numel(),
				base_mask=torch.zeros_like(xs[0]).view(-1, 1), hp_dict=hp_dict, wandb=wandb_run)

		sm_hp_searcher.search_best_linear_fit((xs, ys))
		best_fit = sm_hp_searcher.get_best_fit()
		score_map[model_type] = best_fit 
	return score_map


# Use the data obtained from running the model to build a prior for sampling probability
def run_data_to_sampling_proba(info, module, pfrac):
	avg_act_magnitudes = info['in'][1] / info['in'][0]
	sampling_proba = avg_act_magnitudes.cpu().squeeze().numpy()

	num_keep_static = 0 if pfrac is None else int(len(sampling_proba)*(1.0 - 2*pfrac)) # hard coded to look at 2x the original pruning fraction
	sorted_ = np.argsort(-sampling_proba)
	fixed_indices, use_indices = sorted_[:num_keep_static], sorted_[num_keep_static:]

	sampling_proba = sampling_proba.max() - sampling_proba
	sampling_proba[fixed_indices] = 0

	if module.main_mask is not None:
		sampling_proba *= (module.main_mask).cpu().float().squeeze().numpy()
	sampling_proba /= np.sum(sampling_proba)
	
	if np.isnan(sampling_proba).any():
		print('We got nan in the sampling probability')
		pdb.set_trace()

	assert not np.isnan(sampling_proba).any(), 'Nans encountered in the sampling probability distribution'
	return sampling_proba, fixed_indices, use_indices

# Main Code for algorithm
def investigate_score_based_mask(args, model, wandb_run, epoch_=1):

	def update_mask_one_layer(module, info, score_info, prune_frac, regression_weights, fixed_indices, use_indices, preset_qt=None):
		score_model_weights = torch.zeros_like(info['in'][1]).squeeze()
		if regression_weights is None:
			regression_weights = (info['in'][1] / info['in'][0]).squeeze()
			regression_weights = regression_weights[use_indices]

		# bias this so that we do not remove any of the fixed indices
		score_model_weights[fixed_indices] = INF
		score_model_weights[use_indices] = regression_weights

		if preset_qt is None:
			if module.main_mask is not None:
				qt = torch.quantile((score_model_weights[(module.main_mask).squeeze() > 0]).squeeze().float(), prune_frac)
			else:
				qt = torch.quantile(score_model_weights.squeeze().float(), prune_frac)
		else:
			qt = preset_qt

		mask_ = ((score_model_weights > qt)*1.0).half()
		if module.main_mask is not None:
			module.main_mask *= (mask_).view(info['in'][1].shape)
		else:
			module.main_mask = (mask_).view(info['in'][1].shape)
		return module.main_mask.mean().item()

	def compute_updated_masks_local(prune_frac, score_matrix, score_model_maps, all_sampling_proba, mlp_attn_ratio=1.0, preset_qt=None, no_regression=False):
		avgs = 0.0
		for id_, (name, module) in module_map.items():
			this_prune_frac = prune_frac
			if name.endswith('self_attn'):
				this_prune_frac = prune_frac * mlp_attn_ratio

			_, fixed_indices, use_indices = all_sampling_proba[id_]
			score_model = None if ((score_model_maps is None) or (no_regression)) else score_model_maps[id_]
			score_matrix_entry = score_matrix[id_] if score_matrix is not None else None
			this_avg = update_mask_one_layer(
										module, info_cache[name], 
										score_matrix_entry, this_prune_frac,
										score_model, fixed_indices, use_indices, preset_qt=preset_qt)
			avgs += this_avg

		# Clear the info-cache for the next round !
		for k, v in info_cache.items():
			info_cache[k] = dict()

	# def compute_updated_masks_global(prune_frac, score_matrix, score_model_map, all_sampling_proba, mlp_attn_ratio=1.0,):
	# 	start_idx = 0
	# 	for id_, (name, module) in module_map.items():
	# 		this_group = best_fit[start_idx: (start_idx + len(score_perfs[0][id_][0]))]
	# 		score_map[id_] = this_group


	# add forward hooks
	module_map = {}
	info_cache, hook_handles = defaultdict(dict), []
	for (name, module) in model.named_modules():
		# For now, only focus on the MLPs
		if name.endswith('mlp') or name.endswith('self_attn'):
			# This module has already been fully pruned.
			if module.skip_computation:
				continue

			hook_handles.append(module.register_forward_hook(hook_fn(name, info_cache)))
			id_  = '{}.{}'.format('self_attn' if name.endswith('self_attn') else 'mlp', int(name.split('.')[2]))
			module_map[id_] = (name, module)
			intermediate_sz = module.intermediate_size
			# set the module prune type here
			module.prune_method = args.prune_method

	# Initial setup to get the initial probability distribution for sampling
	tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)

	# Doing this in case the batch_size we have specified is too large for a forwar pass
	# Revert to a forward pass with batch size of 1
	# try:
	# 	eval_ppl_trainonly(model, tokenizer, bsz=args.bsz, nsamples=args.nsamples, dataset=args.dataset)
	# except Exception as e:
	# 	print(e)
	# 	gc.collect()
	# 	torch.cuda.empty_cache()
	# 	eval_ppl_trainonly(model, tokenizer, bsz=1, nsamples=args.nsamples, dataset=args.dataset)

	# Get the initial prior distribution by running the un-modified model
	hp_dict = get_linearmodel_hpdict(args)
	score_matrix = defaultdict(lambda: None)
	all_sampling_proba = defaultdict(lambda: np.ones((intermediate_sz)))
	for id_, (name, module) in module_map.items():
		this_pfrac = args.prune_frac
		if name.endswith('self_attn'):
			this_pfrac = this_pfrac * args.mlp_attn_ratio
		this_pfrac = None if args.no_perturb else this_pfrac
		all_sampling_proba[id_] = run_data_to_sampling_proba(info_cache[name], module, this_pfrac)
		module.main_mask = torch.ones_like(info_cache[name]['in'][1]).half()

	if not args.no_perturb: # We are not running a perturbation algorithm
		# Clear the info-cache for the next round !
		for k, v in info_cache.items():
			info_cache[k] = dict()

		start = time()
		score_info = get_random_mask_scores(
							model, tokenizer, module_map, all_sampling_proba,
							bsz=args.bsz, nsamples=args.nsamples,
							mpi=args.masks_per_iter, pfrac=args.prune_frac, mlp_attn_ratio=args.mlp_attn_ratio,
							dataset_=args.dataset
		)
		gen_scores_time = time() - start
		start = time()
		score_model_maps = get_score_models(score_info, module_map, info_cache, hp_dict, wandb_run, all_sampling_proba, parent_id='Iter.{}'.format(epoch_), model_type=args.sm_lin_model_type)
		time_delta = time() - start
	else:
		gen_scores_time = 0
		time_delta = 0
		score_model_maps = None
		score_matrix = None

	# Need to do some fitting to a linear model here.
	preset_qt = None
	if args.sm_lin_model_type == 'global' and (args.sm_nepochs > 0) and (not args.no_perturb):
		# Find the global best set of modules. Create a mask of these modules surviving pruning.
		best_fit = score_model_maps[args.sm_lin_model_type].cpu().numpy()
		init_param_counts = np.zeros_like(best_fit)
		start_idx = 0
		for id_, (name, module) in module_map.items():
			num_entries =  len(score_info[0][id_][0])
			if name.endswith('mlp'):
				init_param_counts[start_idx: (start_idx + num_entries)] = model.model.params_per_pruned_hidden
			else:
				init_param_counts[start_idx: (start_idx + num_entries)] = model.model.params_per_pruned_head
			start_idx += num_entries

		sort_idxs = np.argsort(best_fit)
		cum_sum_param_counts = np.cumsum(init_param_counts[sort_idxs])
		threshold = int(model.original_param_count * args.prune_frac)
		keep_idxs = (cum_sum_param_counts > threshold) * 1.0
		best_fit = torch.tensor(keep_idxs[np.argsort(sort_idxs)], device=score_model_maps[args.sm_lin_model_type].device).float()
		# We need to do some prep here
		start_idx = 0
		del score_model_maps[args.sm_lin_model_type]
		for id_, (name, module) in module_map.items():
			num_entries =  len(score_info[0][id_][0])
			this_group = best_fit[start_idx: (start_idx + num_entries)]
			score_model_maps[id_] = this_group
			start_idx += num_entries
		preset_qt = 0

	
	compute_updated_masks_local(args.prune_frac, score_matrix, score_model_maps, all_sampling_proba, mlp_attn_ratio=args.mlp_attn_ratio, preset_qt=preset_qt, no_regression=(args.sm_nepochs == 0))
	wandb_run.log({'SysStats/scoreruntime': gen_scores_time, 'SysStats/pruneruntime': time_delta})
	mask_info = {name: module.main_mask for _, (name, module) in module_map.items()}

	for handle in hook_handles:
		handle.remove()

	return mask_info

def args_to_dict(args):
	def stringify(x):
		return '-'.join([str(y) for y in eval(x)])

	return {
		'nsamp': args.nsamples,
		'sp': args.sparsity_ratio,
		'pfrac': args.prune_frac,
		'bsz': args.bsz,
		'ma_ratio': args.mlp_attn_ratio,
		'mpi': args.masks_per_iter,
		'Lin.regtype': args.sm_reg_type, 
		'pmethod': args.prune_method,
		'mlp_attn_ratio': args.mlp_attn_ratio,
		'Lin.regweight': stringify(args.sm_reg_weight),
		'Lin.lr': stringify(args.sm_lr_factor),
		'Lin.bsz': stringify(args.sm_bsz),
		'Lin.nepochs': args.sm_nepochs,
		'Lin.type': args.sm_lin_model_type,
		'name': args.wandb_project_name,
		'Adaptive': 'Yes'
	}

def args_to_str(args):
	relevant_args = args_to_dict(args)
	return '_'.join(['{}={}'.format(k, v) for k, v in relevant_args.items()])

def get_linearmodel_hpdict(args):
	base_hp = {
		'lr_factor' : eval(args.sm_lr_factor),
		'reg_weight': eval(args.sm_reg_weight),
		'reg_type': [args.sm_reg_type],
		'bsz' : eval(args.sm_bsz),
		'nepochs' : [args.sm_nepochs],
		'patience': [10],
	}
	return base_hp

def get_param_count(model, exclude=['embed', 'head']):
	return sum([p.numel() for n, p in model.named_parameters() if not any(x in n for x in exclude)])


# [NB] this pruning is specific to the LLaMA models. You will have to implement your own pruning for a custom model
def prune_mlp(mask_, module):
	# Reset pruning related information
	module.main_mask = None
	module.temp_mask = None
	module.intermed_cache = None
	module.ins_ = None

	if mask_.mean() == 0: # We are pruning the whole module here !
		print("We are pruning the whole mlp layer")
		module.gate_proj = None
		module.up_proj   = None
		module.down_proj = None
		module.intermediate_size = 0
		module.skip_computation = True
	else:
		index = mask_.squeeze().nonzero().squeeze()
		new_gate_proj = (prune_linear_layer(module.gate_proj, index)).half()
		module.gate_proj = None
		module.gate_proj = new_gate_proj
		new_up_proj = (prune_linear_layer(module.up_proj, index)).half()
		module.up_proj  = None
		module.up_proj = new_up_proj
		new_down_proj = (prune_linear_layer(module.down_proj, index, dim=1)).half()
		module.down_proj = None
		module.down_proj = new_down_proj
		module.intermediate_size = len(index)

	gc.collect()
	torch.cuda.empty_cache()

# [NB] this pruning is specific to the LLaMA models. You will have to implement your own pruning for a custom model
def prune_attn(mask_, module):

	module.main_mask = None
	module.temp_mask = None
	module.intermed_cache = None
	module.ins_ = None

	if mask_.mean() == 0: # We are pruning the whole module here !
		print('We are pruning a whole attention layer')
		module.q_proj = None
		module.k_proj = None
		module.v_proj = None
		module.o_proj = None
		module.skip_computation = True
		module.num_heads = 0
		module.hidden_size = 0
		module.intermediate_size = 0
	else:
		index = (mask_.squeeze() == 0).nonzero().squeeze()
		if index.numel() == 1:
			index = [index]

		_, updated_indices = find_pruneable_heads_and_indices(
			index, module.num_heads, module.head_dim, set()
		)

		new_q_proj = (prune_linear_layer(module.q_proj, updated_indices)).half()
		module.q_proj = None
		module.q_proj = new_q_proj

		new_k_proj = (prune_linear_layer(module.k_proj, updated_indices)).half()
		module.k_proj = None
		module.k_proj = new_k_proj

		new_v_proj = (prune_linear_layer(module.v_proj, updated_indices)).half()
		module.v_proj = None
		module.v_proj = new_v_proj

		new_o_proj = (prune_linear_layer(module.o_proj, updated_indices, dim=1)).half()
		module.o_proj = None
		module.o_proj = new_o_proj

		module.num_heads = len(mask_.squeeze().nonzero())
		module.hidden_size = module.num_heads * module.head_dim
		module.intermediate_size = module.num_heads


	gc.collect()
	torch.cuda.empty_cache() 

def prune_model(args, model, mask_info, tokenizer):
	for (name, module) in model.named_modules():
		if name not in mask_info: continue # We are not pruning this

		mask_ = mask_info[name]
		if name.endswith('mlp'):
			prune_mlp(mask_, module)
		elif name.endswith('self_attn'):
			prune_attn(mask_, module)
		else:
			raise ValueError("Invalid type found in mask_info : {}".format(name))

	gc.collect()
	torch.cuda.empty_cache() 


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--model', type=str, default='meta-llama/Llama-2-7b-hf', help='LLaMA model') # huggyllama/llama-7b
	parser.add_argument('--dataset', type=str, default="wikitext2", choices=["wikitext2", "c4"])
	parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
	parser.add_argument('--nsamples', type=int, default=32, help='Number of calibration samples.')
	parser.add_argument('--sparsity_ratio', type=float, default=0.5, help='Sparsity level')
	parser.add_argument('--prune_frac', type=float, default=0.05, help='Fraction of weights to prune at a time')
	parser.add_argument('--bsz', type=int, default=4, help='Instantaneous batch size for forward pass')
	parser.add_argument('--mlp_attn_ratio', type=float, default=1.0, help="For a given prune_frac, the ratio of the pruning for attn vrs mlp")

	parser.add_argument('--prune_method', type=str, default="wanda", choices=["magnitude", "wanda", "random"])
	parser.add_argument("--cache_dir", default="llm_weights", type=str )
	parser.add_argument('--use_variant', action="store_true", help="whether to use the wanda variant described in the appendix")
	parser.add_argument('--save', type=str, default=None, help='Path to save results.')
	parser.add_argument('--save_model', type=str, default=None, help='Path to save the pruned model.')
	parser.add_argument('--masks_per_iter', type=int, default=200, help='How many masks to generate per-iteration')
	parser.add_argument('--tol', type=float, default=0.02, help="What level of tolerance close to the target sparsity to accept")
	parser.add_argument('--no_perturb', action="store_true", help="We do not perform any perturbation")

	# Hyperparams for scoring model
	parser.add_argument('--sm_reg_weight', type=str, default='[1e2, 1e-4, 0]', help='reg-weight to use')
	parser.add_argument('--sm_lr_factor', type=str, default='[100, 10, 1, 0.1]', help='lr factor to use for fitting linear model')
	parser.add_argument('--sm_reg_type', type=str, default="l1", help='type of regularization to apply')
	parser.add_argument('--sm_lin_model_type', type=str, default="global", help='type of regularization to apply') 


	parser.add_argument('--sm_bsz', type=str, default='[32, 64, 128]', help='batch size for fitting linear model')
	parser.add_argument('--sm_nepochs', type=int, default=50, help='number of epochs to use to fit the linear model')
	
	# Wandb HP
	parser.add_argument('--wandb_project_name', type=str, default='Prune-No-Backward', help='Wandb project name')

	args = parser.parse_args()
	print(args)
	str_of_args = args_to_str(args)
	args.save = os.path.join(args.save, str_of_args)
	os.makedirs(args.save, exist_ok=True)


	# Setting seeds for reproducibility
	np.random.seed(args.seed)
	torch.random.manual_seed(args.seed)

	wandb_run = wandb.init(
		project=args.wandb_project_name,
		name=str_of_args,
		config=args_to_dict(args),
	)

	model_name = args.model.split("/")[-1]
	print(f"loading llm model {args.model}")
	model = get_llm(args.model, args.cache_dir)
	model.eval()
	tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
	
	# Getting the initial evaluation of the model
	# _, orig_test_ppl = eval_ppl(model, tokenizer, model.device, dataset=args.dataset)

	original_param_count = get_param_count(model)
	model.original_param_count = original_param_count
	cur_sparsity = 1.0 - (get_param_count(model) / original_param_count)
	epoch_ = 1
	while True:
		# If the sparsity is within a reasonable tolerance of the target, we can break
		if (abs(cur_sparsity - args.sparsity_ratio) < args.tol) or (cur_sparsity > args.sparsity_ratio):
			break

		# Need to check if we have to clip the sparsity ratio (if the current ratio causes us to overshoot)
		if (cur_sparsity + args.prune_frac) > args.sparsity_ratio:
			# We would overshoot in this case which is not idea.
			old_prune_frac = args.prune_frac
			args.prune_frac = abs(args.sparsity_ratio - cur_sparsity)
			print('We have updated the prune fraction {:.3f} -> {:.3f} to avoid overshooting'.format(old_prune_frac, args.prune_frac))


		print('Gathering statistics for pruning')
		save_loc = os.path.join(args.save, 'mask_info_{}.pkl'.format(epoch_))
		if os.path.exists(save_loc):
			print('Successfully loaded past pruning info')
			with open(save_loc, 'rb') as handle:
				mask_info = pkl.load(handle)
		else:
			mask_info = investigate_score_based_mask(args, model, wandb_run, epoch_=epoch_)
			# Save the mask info for the epoch
			with open(save_loc, 'wb') as handle:
				pkl.dump(mask_info, handle)

		print('Prune model')
		prune_model(args, model, mask_info, tokenizer) # Do some stuffs here :)
		cur_sparsity = 1.0 - (get_param_count(model) / original_param_count)
		print(model)

		# Evaluate the performance of the pruned model
		# ppl_train, ppl_test = eval_ppl(model, tokenizer, model.device, dataset=args.dataset)

		# wandb_run.log({'Sparsity': cur_sparsity, 'TrainPPL': ppl_train, 'TestPPL': ppl_test})
		# print('Sparsity = {:.3f}| Train PPL = {:.3f} | Test PPL = {:.3f}'.format(cur_sparsity, ppl_train, ppl_test))

		epoch_ += 1

	wandb_run.log({'sparsity': cur_sparsity})


if __name__ == '__main__':
    main()





# import os
# import gc
# import copy
# import random
# from copy import deepcopy
# import abc
# from abc import ABC, abstractclassmethod, abstractmethod, abstractstaticmethod

# import torch
# from torch import nn
# import numpy as np
# # from transformers import LlamaTokenizer, GenerationConfig, LlamaConfig
# from accelerate import Accelerator
# from typing import Callable, Sequence, Tuple, Dict, Tuple, List
# # from LLMPruner.models.hf_llama.modeling_llama import LlamaForCausalLM, LlamaRMSNorm, LlamaAttention, LlamaMLP

# # import LLMPruner.torch_pruning as tp 
# # from LLMPruner.pruner import hf_llama_pruner as llama_pruner
# # from LLMPruner.utils.logger import LoggerWithDepth
# # from LLMPruner.datasets.example_samples import get_examples
# # from LLMPruner.templates.prompts import prompts

# def set_random_seed(seed):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
    
    
# # Standard Modules
# TORCH_CONV = nn.modules.conv._ConvNd
# TORCH_BATCHNORM = nn.modules.batchnorm._BatchNorm
# TORCH_LAYERNORM = nn.modules.normalization.LayerNorm
# TORCH_GROUPNORM = nn.GroupNorm
# TORCH_INSTANCENORM = nn.modules.instancenorm._InstanceNorm
# TORCH_PRELU = nn.PReLU
# TORCH_LINEAR = nn.Linear
# TORCH_EMBED = nn.Embedding
# TORCH_PARAMETER = nn.Parameter
# TORCH_LSTM = nn.LSTM
# # try:
# #     TORCH_MHA = nn.MultiheadAttention
# # except:
# #     TORCH_MHA = DummyMHA  # for pytorch w/o MultiHeadAttention
# # TORCH_OTHERS = None


# class OPTYPE(object):
#     CONV = 0
#     BN = 1
#     LINEAR = 2
#     PRELU = 3
#     DEPTHWISE_CONV = 4
#     CONCAT = 5  # torch.cat
#     SPLIT = 6  # torch.split
#     CUSTOMIZED = 7  # customized module
#     ELEMENTWISE = 8  # element-wise add, sub, etc.
#     LN = 9  # nn.LayerNorm
#     EMBED = 10  # nn.Embedding
#     PARAMETER = 11  # nn.Parameter
#     MHA = 12
#     LSTM = 13
#     RESHAPE = 14
#     GN = 15  # nn.GroupNorm
#     IN = 16  # nn.InstanceNorm
    
    
# def prune_structured(args, model, dataloader, accelerator: Accelerator, num_samples):
#     # set_random_seed(args.seed)

#     # logger = LoggerWithDepth(
#     #     env_name="{}".format(args.save_ckpt_log_name), 
#     #     config=args.__dict__,
#     #     root_dir='prune_log',
#     #     setup_sublogger=True
#     # )

#     # tokenizer = LlamaTokenizer.from_pretrained(args.base_model)
#     # model = LlamaForCausalLM.from_pretrained(
#     #     args.base_model,
#     #     low_cpu_mem_usage=True if args.torch_version >=1.9 else False
#     # )
#     # if args.device != "cpu":
#     #     model.half()
#     # model.to(args.device)

#     # if args.test_before_train:
#     #     logger.log("\n==================Generation Results before Pruning================\n")
#     #     model.eval()
#     #     with torch.no_grad():
#     #         for prompt in prompts:
#     #             input_ids = tokenizer(prompt, return_tensors="pt")['input_ids'].to(args.device)

#     #             generation_output = model.generate(
#     #                 input_ids=input_ids,
#     #                 do_sample=True,
#     #                 top_k=50,
#     #                 max_length=args.max_seq_len,
#     #                 top_p=args.top_p,
#     #                 temperature=args.temperature,
#     #             )
                
#     #             result = tokenizer.decode(generation_output[0])
#     #             logger.log(result)
    
#     #     ppl = PPLMetric(model, tokenizer, ['wikitext2', 'ptb'], args.max_seq_len, device=args.device)
#     #     logger.log("PPL before pruning: {}".format(ppl))

#     pruner_type = args.pruner_type.lower()
#     assert pruner_type in ['random', 'l2', 'l1', 'taylor']

#     for param in model.parameters():
#         param.requires_grad_(True)
#     before_pruning_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
#     forward_prompts = torch.tensor([
#         [    1,   306,  4658,   278,  6593,   310,  2834,   338],
#         [    1,  3439, 17632,  1925, 29892,   278,  6368,   310],
#     ]).to(args.device) # Only for building the dependency graph. Any input will be fine since the computation result are not taken into consideration.

#     if pruner_type == 'random':
#         imp = RandomImportance()
#     elif pruner_type == 'l1':
#         imp = MagnitudeImportance(p=1)
#     elif pruner_type == 'l2':
#         imp = MagnitudeImportance(p=2)
#     elif pruner_type == 'taylor':
#         imp = TaylorImportance(group_reduction=args.grouping_strategy, taylor=args.taylor)
#     else:
#         raise NotImplementedError

#     accelerator.print("Use {} pruner...".format(pruner_type))
    
#     # if args.block_wise:
#     kwargs = {
#         "importance": imp,
#         "global_pruning": args.global_pruning,
#         "iterative_steps": args.iterative_steps,
#         "ch_sparsity": args.pruning_ratio, 
#         "ignored_layers":[],
#         "channel_groups": {
#         },
#         # "consecutive_groups": {
#         #     layer.self_attn.q_proj: layer.self_attn.head_dim for layer in model.model.layers
#         # },
#         # "customized_pruners": {
#         #     LlamaRMSNorm: llama_pruner.hf_rmsnorm_pruner,
#         # },
#         "root_module_types": None, 
#         "root_instances": 
#             # [model.model.layers[i].self_attn.q_proj for i in range(args.block_attention_layer_start, args.block_attention_layer_end)] + \
#             [model.model.layers[i].mlp.gate_proj for i in range(args.block_mlp_layer_start, args.block_mlp_layer_end)]
#     }
#     # logger.log("Pruning Attention Layer = {}".format(list(range(args.block_attention_layer_start, args.block_attention_layer_end))))
#     accelerator.print("Pruning MLP Layer = {}".format(list(range(args.block_mlp_layer_start, args.block_mlp_layer_end))))

#     pruner = MetaPruner(
#         model,
#         forward_prompts,
#         **kwargs
#     )
#     model.zero_grad()

#     accelerator.print("Start Pruning")
#     for i in range(args.iterative_steps):

#         if pruner_type in ['taylor']:
#             example_prompts = get_examples('bookcorpus', tokenizer, args.num_examples, seq_len = 64).to(args.device)
#             accelerator.print("Start Backwarding in iterative steps = {}...".format(i))
#             if args.taylor in ['param_mix', 'param_second']:
#                 for j in range(args.num_examples):
#                     batch_input = example_prompts[j].unsqueeze(0)
#                     loss = model(batch_input, labels=batch_input).loss
#                     accelerator.print("Loss = {}".format(loss))
#                     loss.backward()

#                     for module_param in model.parameters():
#                         module_param.grad = module_param.grad * module_param.grad / args.num_examples
#                         if hasattr(module_param, 'acc_grad'):
#                             module_param.acc_grad += module_param.grad
#                         else:
#                             module_param.acc_grad = copy.deepcopy(module_param.grad)
#                     model.zero_grad()
#                     del loss.grad
                
#             loss = model(example_prompts, labels=example_prompts).loss
#             accelerator.print("Loss = {}".format(loss))
#             loss.backward()

#         pruner.step()

#         after_pruning_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
#         accelerator.print("After Iter {}/{}, #parameters: {}".format(i+1, args.iterative_steps, after_pruning_parameters))
    
#         # modify inferece-related attributes
#         for layer in model.model.layers:
#             layer.self_attn.num_heads = layer.self_attn.q_proj.weight.data.shape[0] // layer.self_attn.head_dim

#     # Clean the gradient in the model
#     model.zero_grad()
#     for name, module in model.named_parameters():
#         if 'weight' in name:
#             module.grad = None

#     del pruner

#     # elif args.channel_wise:
#     #     kwargs = {
#     #         "importance": imp,
#     #         "global_pruning": args.global_pruning,
#     #         "iterative_steps": args.iterative_steps,
#     #         "ch_sparsity": args.pruning_ratio, # remove 50% channels, ResNet18 = {64, 128, 256, 512} => ResNet18_Half = {32, 64, 128, 256}
#     #         "ignored_layers":[],
#     #         #"round_to": model.config.num_attention_heads * 2,
#     #         "channel_groups": {
#     #             #layer.self_attn: layer.self_attn.num_heads for layer in model.model.layers
#     #         },
#     #         "customized_pruners": {
#     #             LlamaRMSNorm: llama_pruner.hf_rmsnorm_pruner,
#     #             #LlamaAttention: llama_pruner.hf_attention_pruner,
#     #         },
#     #         "root_module_types": [LlamaRMSNorm, LlamaAttention],
#     #     }

#     #     pruner = tp.pruner.MetaPruner(
#     #         model,
#     #         forward_prompts,
#     #         **kwargs
#     #     )
#     #     model.zero_grad()
        
#     #     logger.log("Start Pruning")
#     #     for i in range(args.iterative_steps):

#     #         if pruner_type in ['taylor']:
#     #             example_prompts = get_examples('bookcorpus', tokenizer, 10, seq_len = 64)
#     #             logger.log("Start Backwarding in iterative steps = {}...".format(i))
#     #             loss = model(example_prompts, labels=example_prompts).loss
#     #             logger.log("Loss = {}".format(loss))
#     #             loss.backward()

#     #         pruner.step()

#     #         after_pruning_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     #         logger.log("After Iter {}/{}, #parameters: {}".format(i+1, args.iterative_steps, after_pruning_parameters))

#         # Clean the gradient in the model
#         # model.zero_grad()
#         # for name, module in model.named_parameters():
#         #     if 'weight' in name:
#         #         module.grad = None

#         # modify inferece-related attributes
#         # model.config.hidden_size = model.model.embed_tokens.weight.shape[1]
#         # model.zero_grad()
        
#         # del pruner
            
#     # elif args.layer_wise:
#     #     model.model.layers = model.model.layers[:args.layer]
#     #     after_pruning_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

#     # else:
#     #     raise NotImplementedError
    
#     accelerator.print("#Param before: {}, #Param after: {}, Ratio = {:.4f}%".format(before_pruning_parameters, after_pruning_parameters,  100.0*after_pruning_parameters/before_pruning_parameters))
    
#     gc.collect()
#     torch.cuda.empty_cache()

#     # if args.save_model:
#     #     model.half()
#     #     torch.save({
#     #         'model': model, 
#     #         'tokenizer': tokenizer,
#     #     }, logger.best_checkpoint_path)

#     accelerator.print("Memory Requirement: {} MiB\n".format(torch.cuda.memory_allocated()/1024/1024))

# ##############################
# # Pruner
# ##############################
# class BasePruningFunc(ABC):
#     TARGET_MODULES = ops.TORCH_OTHERS  # None

#     def __init__(self, pruning_dim=1):
#         self.pruning_dim = pruning_dim

#     @abstractclassmethod
#     def prune_out_channels(self, layer: nn.Module, idxs: Sequence[int]):
#         raise NotImplementedError

#     @abstractclassmethod
#     def prune_in_channels(self, layer: nn.Module, idxs: Sequence[int]):
#         raise NotImplementedError

#     @abstractclassmethod
#     def get_out_channels(self, layer: nn.Module):
#         raise NotImplementedError

#     @abstractclassmethod
#     def get_in_channels(self, layer: nn.Module):
#         raise NotImplementedError

#     def check(self, layer, idxs, to_output):
#         if self.TARGET_MODULES is not None:
#             assert isinstance(layer, self.TARGET_MODULES), 'Mismatched pruner {} and module {}'.format(
#                 self.__str__, layer)
#         if to_output:
#             prunable_channels = self.get_out_channels(layer)
#         else:
#             prunable_channels = self.get_in_channels(layer)
#         if prunable_channels is not None:
#             assert all(idx < prunable_channels and idx >=
#                        0 for idx in idxs), "All pruning indices should fall into [{}, {})".format(0, prunable_channels)

#     def __call__(self, layer: nn.Module, idxs: Sequence[int], to_output: bool = True, inplace: bool = True, dry_run: bool = False) -> Tuple[nn.Module, int]:
#         idxs.sort()
#         self.check(layer, idxs, to_output)
#         pruning_fn = self.prune_out_channels if to_output else self.prune_in_channels
#         if not inplace:
#             layer = deepcopy(layer)
#         layer = pruning_fn(layer, idxs)
#         return layer
    
# class HFLinearPrunner(BasePruningFunc):
#     TORCH_LINEAR = nn.Linear
#     TARGET_MODULES = TORCH_LINEAR

#     def prune_out_channels(self, layer: nn.Module, idxs: Sequence[int]) -> nn.Module:
#         keep_idxs = list(set(range(layer.out_features)) - set(idxs))
#         keep_idxs.sort()
#         idxs.sort()
#         layer.out_features = layer.out_features-len(idxs)

#         keep_weight = layer.weight.data[keep_idxs]
#         remove_weight = layer.weight.data[idxs]

#         sim = torch.mm(remove_weight, keep_weight.t())
#         max_indices = torch.argmax(sim, dim=-1)
#         keep_weight[max_indices] += remove_weight
#         cnt = torch.ones((keep_weight.size(0), 1), device=keep_weight.device)
#         cnt[torch.max(sim, dim=-1).indices] += 1
#         keep_weight = keep_weight / cnt

#         layer.weight = torch.nn.Parameter(keep_weight)
#         if layer.bias is not None:
#             keep_bias = layer.bias.data[keep_idxs]
#             remove_bias = layer.bias.data[idxs]
#             keep_bias[max_indices] += remove_bias
#             keep_bias = keep_bias / cnt
#             layer.bias = torch.nn.Parameter(keep_bias)
#         return layer

#     def prune_in_channels(self, layer: nn.Module, idxs: Sequence[int]) -> nn.Module:
#         keep_idxs = list(set(range(layer.in_features)) - set(idxs))
#         keep_idxs.sort()
#         layer.in_features = layer.in_features-len(idxs)

#         keep_weight = layer.weight.data[:, keep_idxs]
#         remove_weight = layer.weight.data[:, idxs]

#         sim = torch.mm(remove_weight.t(), keep_weight)
#         max_indices = torch.argmax(sim, dim=-1)
#         keep_weight[:, max_indices] += remove_weight
#         cnt = torch.ones((1, keep_weight.size(1)), device=keep_weight.device)
#         cnt[:, torch.max(sim, dim=-1).indices] += 1
#         #keep_weight = keep_weight / cnt

#         layer.weight = torch.nn.Parameter(keep_weight)
#         return layer

#     def get_out_channels(self, layer):
#         return layer.out_features

#     def get_in_channels(self, layer):
#         return layer.in_features


# class MetaPruner:
#     """
#         Meta Pruner for structural pruning.

#         Args:
#             model (nn.Module): A to-be-pruned model
#             example_inputs (torch.Tensor or List): dummy inputs for graph tracing.
#             importance (Callable): importance estimator.
#             global_pruning (bool): enable global pruning. 
#             ch_sparsity (float): global channel sparisty.
#             ch_sparsity_dict (Dict[nn.Module, float]): layer-specific sparsity.
#             iterative_steps (int): number of steps for iterative pruning.
#             iterative_sparsity_scheduler (Callable): scheduler for iterative pruning.
#             max_ch_sparsity (float): maximum channel sparsity.
#             ignored_layers (List[nn.Module]): ignored modules.

#             round_to (int): channel rounding.
#             customized_pruners (dict): a dict containing module-pruner pairs.
#             unwrapped_parameters (list): nn.Parameter that does not belong to any supported layerss.
#             root_module_types (list): types of prunable modules.
#             output_transform (Callable): A function to transform network outputs.
#         """

#     def __init__(
#         self,
#         # Basic
#         model: nn.Module,
#         example_inputs: torch.Tensor,
#         importance: Callable,
#         # https://pytorch.org/tutorials/intermediate/pruning_tutorial.html#global-pruning.
#         global_pruning: bool = False,
#         ch_sparsity: float = 0.5,  # channel/dim sparsity
#         ch_sparsity_dict: Dict[nn.Module, float] = None,
#         max_ch_sparsity: float = 1.0,
#         iterative_steps: int = 1,  # for iterative pruning
#         iterative_sparsity_scheduler: Callable = linear_scheduler,
#         ignored_layers: List[nn.Module] = None,

#         # Advanced
#         round_to: int = None,  # round channels to 8x, 16x, ...
#         # for grouped channels.
#         channel_groups: Dict[nn.Module, int] = dict(),
#         # for consecutive channels.
#         consecutive_groups: Dict[nn.Module, int] = dict(),
#         # pruners for customized layers
#         customized_pruners: Dict[Any,
#                                         function.BasePruningFunc] = None,
#         # unwrapped nn.Parameters like ViT.pos_emb
#         unwrapped_parameters: List[nn.Parameter] = None,
#         root_module_types: List = [
#             ops.TORCH_CONV, ops.TORCH_LINEAR, ops.TORCH_LSTM],  # root module for each group
#         root_instances: List = None,
#         forward_fn: Callable = None,
#         output_transform: Callable = None,
#         enable_index_mapping: bool = False,
#     ):
#         self.model = model
#         self.importance = importance
#         self.ch_sparsity = ch_sparsity
#         self.ch_sparsity_dict = ch_sparsity_dict if ch_sparsity_dict is not None else {}
#         self.max_ch_sparsity = max_ch_sparsity
#         self.global_pruning = global_pruning

#         self.channel_groups = channel_groups
#         self.consecutive_groups = consecutive_groups
#         self.root_module_types = root_module_types
#         self.root_instances = root_instances
#         self.round_to = round_to

#         # Build dependency graph
#         self.DG = dependency.DependencyGraph().build_dependency(
#             model,
#             example_inputs=example_inputs,
#             forward_fn=forward_fn,
#             output_transform=output_transform,
#             unwrapped_parameters=unwrapped_parameters,
#             customized_pruners=customized_pruners,
#         )

#         self.ignored_layers = []
#         if ignored_layers:
#             for layer in ignored_layers:
#                 self.ignored_layers.extend(list(layer.modules()))

#         self.iterative_steps = iterative_steps
#         self.iterative_sparsity_scheduler = iterative_sparsity_scheduler
#         self.current_step = 0

#         # Record initial status
#         self.layer_init_out_ch = {}
#         self.layer_init_in_ch = {}
#         for m in self.DG.module2node.keys():
#             if ops.module2type(m) in self.DG.REGISTERED_PRUNERS:
#                 self.layer_init_out_ch[m] = self.DG.get_out_channels(m)
#                 self.layer_init_in_ch[m] = self.DG.get_in_channels(m)

#         # global channel sparsity for each iterative step
#         self.per_step_ch_sparsity = self.iterative_sparsity_scheduler(
#             self.ch_sparsity, self.iterative_steps
#         )

#         # The customized channel sparsity for different layers
#         self.ch_sparsity_dict = {}
#         if ch_sparsity_dict is not None:
#             for module in ch_sparsity_dict:
#                 sparsity = ch_sparsity_dict[module]
#                 for submodule in module.modules():
#                     prunable_types = tuple([ops.type2class(
#                         prunable_type) for prunable_type in self.DG.REGISTERED_PRUNERS.keys()])
#                     if isinstance(submodule, prunable_types):
#                         self.ch_sparsity_dict[submodule] = self.iterative_sparsity_scheduler(
#                             sparsity, self.iterative_steps
#                         )

#         # detect group convs & group norms
#         for m in self.model.modules():
#             if isinstance(m, ops.TORCH_CONV) \
#                 and m.groups > 1 \
#                     and m.groups != m.out_channels:
#                 self.channel_groups[m] = m.groups
#             if isinstance(m, ops.TORCH_GROUPNORM):
#                 self.channel_groups[m] = m.num_groups
        
#         if self.global_pruning: # TODO: Support both ch_groups and consecutive_groups in a single forward
#             initial_total_channels = 0
#             for group in self.DG.get_all_groups(ignored_layers=self.ignored_layers, root_module_types=self.root_module_types, root_instances=self.root_instances):
#                 ch_groups = self.get_channel_groups(group)
#                 consecutive_groups = self.get_consecutive_groups(group)
#                 # utils.count_prunable_out_channels( group[0][0].target.module )

#                 if ch_groups > 1:
#                     initial_total_channels += (self.DG.get_out_channels(
#                         group[0][0].target.module) // ch_groups)
#                 elif consecutive_groups > 1:
#                     initial_total_channels += (self.DG.get_out_channels(
#                         group[0][0].target.module) // consecutive_groups)
#                 else:
#                     initial_total_channels += self.DG.get_out_channels(group[0][0].target.module) 
                
#             self.initial_total_channels = initial_total_channels
        
#         if enable_index_mapping:
#             for node in self.DG.module2node.values():
#                 node.enable_index_mapping = True
    
#     def pruning_history(self):
#         return self.DG.pruning_history()

#     def load_pruning_history(self, pruning_history):
#         self.DG.load_pruning_history(pruning_history)

#     def get_target_sparsity(self, module):
#         s = self.ch_sparsity_dict.get(module, self.per_step_ch_sparsity)[
#             self.current_step]
#         return min(s, self.max_ch_sparsity)

#     def reset(self):
#         self.current_step = 0

#     def regularize(self, model, loss):
#         """ Model regularizor
#         """
#         pass

#     def step(self, interactive=False):
#         self.current_step += 1
#         if self.global_pruning:
#             if interactive:
#                 return self.prune_global()
#             else:
#                 for group in self.prune_global():
#                     group.prune()
#         else:
#             if interactive:
#                 return self.prune_local()
#             else:
#                 for group in self.prune_local():
#                     group.prune()

#     def estimate_importance(self, group, ch_groups=1, consecutive_groups=1):
#         return self.importance(group, ch_groups=ch_groups, consecutive_groups=consecutive_groups)

#     def _check_sparsity(self, group):
#         for dep, _ in group:
#             module = dep.target.module
#             pruning_fn = dep.handler
#             if dep.target.type == ops.OPTYPE.PARAMETER:
#                 continue
#             if self.DG.is_out_channel_pruning_fn(pruning_fn):
#                 target_sparsity = self.get_target_sparsity(module)
#                 layer_out_ch = self.DG.get_out_channels(module)
#                 if layer_out_ch is None: continue
#                 if layer_out_ch < self.layer_init_out_ch[module] * (
#                     1 - self.max_ch_sparsity
#                 ) or layer_out_ch == 1:
#                     return False

#             elif self.DG.is_in_channel_pruning_fn(pruning_fn):
#                 layer_in_ch = self.DG.get_in_channels(module)
#                 if layer_in_ch is None: continue
#                 if layer_in_ch < self.layer_init_in_ch[module] * (
#                     1 - self.max_ch_sparsity
#                 ) or layer_in_ch == 1:
#                     return False
#         return True

#     def get_channel_groups(self, group):
#         if isinstance(self.channel_groups, int):
#             return self.channel_groups
#         for dep, _ in group:
#             module = dep.target.module
#             if module in self.channel_groups:
#                 return self.channel_groups[module]
#         return 1  # no channel grouping
    
#     def get_consecutive_groups(self, group):
#         if isinstance(self.consecutive_groups, int):
#             return self.consecutive_groups
#         for dep, _ in group:
#             module = dep.target.module
#             if module in self.consecutive_groups:
#                 return self.consecutive_groups[module]
#         return 1  # no channel grouping

#     def prune_local(self):
#         if self.current_step > self.iterative_steps:
#             return
#         for group in self.DG.get_all_groups(ignored_layers=self.ignored_layers, root_module_types=self.root_module_types, root_instances=self.root_instances):
#             # check pruning rate
#             if self._check_sparsity(group):
#                 module = group[0][0].target.module
#                 pruning_fn = group[0][0].handler

#                 ch_groups = self.get_channel_groups(group)
#                 consecutive_groups = self.get_consecutive_groups(group)
#                 imp = self.estimate_importance(group, ch_groups=ch_groups, consecutive_groups=consecutive_groups)
#                 if imp is None: continue
#                 current_channels = self.DG.get_out_channels(module)
#                 target_sparsity = self.get_target_sparsity(module)
#                 n_pruned = current_channels - int(
#                     self.layer_init_out_ch[module] *
#                     (1 - target_sparsity)
#                 )

#                 if self.round_to:
#                     n_pruned = n_pruned - (n_pruned % self.round_to)
                    
#                 if n_pruned <= 0:
#                     continue

#                 if ch_groups > 1:
#                     imp = imp[:len(imp)//ch_groups]

#                 if consecutive_groups > 1:
#                     imp = imp.view(-1, consecutive_groups).sum(1)

#                 imp_argsort = torch.argsort(imp)
                
#                 if ch_groups > 1:
#                     pruning_idxs = imp_argsort[:(n_pruned//ch_groups)]
#                     group_size = current_channels//ch_groups
#                     pruning_idxs = torch.cat(
#                         [pruning_idxs+group_size*i for i in range(ch_groups)], 0)
#                 elif consecutive_groups > 1:
#                     pruning_groups = imp_argsort[:(n_pruned//consecutive_groups)]
#                     group_size = consecutive_groups
#                     pruning_idxs = torch.cat(
#                         [torch.tensor([j+group_size*i for j in range(group_size)])
#                         for i in pruning_groups], 0)
#                 else:
#                     pruning_idxs = imp_argsort[:n_pruned]

#                 group = self.DG.get_pruning_group(
#                     module, pruning_fn, pruning_idxs.tolist())
#                 if self.DG.check_pruning_group(group):
#                     yield group

#     def prune_global(self):
#         if self.current_step > self.iterative_steps:
#             return
#         global_importance = []
#         for group in self.DG.get_all_groups(ignored_layers=self.ignored_layers, root_module_types=self.root_module_types, root_instances=self.root_instances):
#             if self._check_sparsity(group):
#                 ch_groups = self.get_channel_groups(group)
#                 consecutive_groups = self.get_consecutive_groups(group)
#                 imp = self.estimate_importance(group, ch_groups=ch_groups, consecutive_groups=consecutive_groups)
#                 if imp is None: continue
#                 if ch_groups > 1:
#                     imp = imp[:len(imp)//ch_groups]
#                 if consecutive_groups > 1:
#                     imp = imp.view(-1, consecutive_groups).sum(1)
#                 global_importance.append((group, ch_groups, consecutive_groups, imp))

#         imp = torch.cat([local_imp[-1]
#                         for local_imp in global_importance], dim=0)
#         print(imp.shape, len(global_importance))
#         target_sparsity = self.per_step_ch_sparsity[self.current_step]
#         n_pruned = len(imp) - int(
#             self.initial_total_channels *
#             (1 - target_sparsity)
#         )
#         print(n_pruned, target_sparsity, self.initial_total_channels)
#         if n_pruned <= 0:
#             return
#         topk_imp, _ = torch.topk(imp, k=n_pruned, largest=False)
        
#         # global pruning through thresholding
#         thres = topk_imp[-1]
#         for group, ch_groups, consecutive_groups, imp in global_importance:
#             module = group[0][0].target.module
#             pruning_fn = group[0][0].handler
#             pruning_indices = (imp <= thres).nonzero().view(-1)
            
#             if pruning_indices.size(-1) == 0:
#                 continue
#             if ch_groups > 1:
#                 group_size = self.DG.get_out_channels(module)//ch_groups
#                 pruning_indices = torch.cat(
#                     [pruning_indices+group_size*i for i in range(ch_groups)], 0)
#             if consecutive_groups > 1:
#                 group_size = consecutive_groups
#                 pruning_indices = torch.cat(
#                     [torch.tensor([j+group_size*i for j in range(group_size)])
#                     for i in pruning_indices], 0)
#             if self.round_to:
#                 n_pruned = len(pruning_indices)
#                 n_pruned = n_pruned - (n_pruned % self.round_to)
#                 pruning_indices = pruning_indices[:n_pruned]
#             group = self.DG.get_pruning_group(
#                 module, pruning_fn, pruning_indices.tolist())
#             if self.DG.check_pruning_group(group):
#                 yield group


# hf_linear_pruner = HFLinearPrunner()


# ##############################
# # Importance
# ##############################
# class Importance(abc.ABC):
#     """ estimate the importance of a Pruning Group, and return an 1-D per-channel importance score.
#     """
#     @abc.abstractclassmethod
#     def __call__(self, group)-> torch.Tensor:
#         raise NotImplementedError
    
    
# class MagnitudeImportance(Importance):
#     def __init__(self, p=2, group_reduction="mean", normalizer=None):
#         self.p = p
#         self.group_reduction = group_reduction
#         self.normalizer = normalizer

#     def _reduce(self, group_imp):
#         if self.group_reduction == "sum":
#             group_imp = group_imp.sum(dim=0)
#         elif self.group_reduction == "mean":
#             group_imp = group_imp.mean(dim=0)
#         elif self.group_reduction == "max":
#             group_imp = group_imp.max(dim=0)[0]
#         elif self.group_reduction == "prod":
#             group_imp = torch.prod(group_imp, dim=0)
#         elif self.group_reduction=='first':
#             group_imp = group_imp[0]
#         elif self.group_reduction is None:
#             group_imp = group_imp
#         else: 
#             raise NotImplementedError
#         return group_imp

#     @torch.no_grad()
#     def __call__(self, group, ch_groups=1, consecutive_groups=1):
    
#         group_imp = []
#         for dep, idxs in group:
#             idxs.sort()
#             layer = dep.target.module
#             prune_fn = dep.handler
#             # Linear out_channels
#             if prune_fn in [
#                             tp.prune_linear_out_channels, 
#                             hf_linear_pruner.prune_out_channels]:
#                 w = layer.weight.data[idxs].flatten(1)
#                 local_norm = w.abs().pow(self.p).sum(1)
#                 group_imp.append(local_norm)
#             # Linear in_channels
#             elif prune_fn in [
#                 tp.prune_linear_in_channels, hf_linear_pruner.prune_in_channels
#             ]:    
#                 w = layer.weight
#                 local_norm = w.abs().pow(self.p).sum(0)
#                 local_norm = local_norm[idxs]
#                 group_imp.append(local_norm)
#             # # RMSNorm
#             # elif prune_fn == hf_rmsnorm_pruner.prune_out_channels:
#             #     # regularize BN
#             #     w = layer.weight.data[idxs]
#             #     local_norm = w.abs().pow(self.p)
#             #     group_imp.append(local_norm)
#             # # Embedding
#             # elif prune_fn == tp.prune_embedding_out_channels:
#             #     w = layer.weight.data[:, idxs]
#             #     local_norm = w.abs().pow(self.p)
#                 group_imp.append(local_norm)
#             # # Attention
#             # elif prune_fn == hf_attention_pruner.prune_out_channels:
#             #     local_norm = 0
#             #     for sub_layer in [layer.o_proj]:
#             #         w_out = sub_layer.weight.data[idxs]
#             #         local_norm += w_out.abs().pow(self.p).sum(1)

#             #     for sub_layer in [layer.q_proj, layer.k_proj, layer.v_proj]:
#             #         w_in = sub_layer.weight.data[:, idxs]
#             #         local_norm += w_in.abs().pow(self.p).sum(0)
#             #     group_imp.append(local_norm)

#         if len(group_imp)==0:
#             return None
#         min_imp_size = min([len(imp) for imp in group_imp])
#         aligned_group_imp = []
#         for imp in group_imp:
#             if len(imp)>min_imp_size and len(imp)%min_imp_size==0:
#                 imp = imp.view(len(imp) // min_imp_size, min_imp_size).sum(0)
#                 aligned_group_imp.append(imp)
#             elif len(imp)==min_imp_size:
#                 aligned_group_imp.append(imp)
#         group_imp = torch.stack(aligned_group_imp, dim=0)
#         group_imp = self._reduce(group_imp)
#         if self.normalizer is not None:
#             group_imp = self.normalizer(group, group_imp)
#         return group_imp

# class TaylorImportance(Importance):
#     def __init__(self, group_reduction="sum", normalizer=None, taylor=None):
#         self.group_reduction = group_reduction
#         self.normalizer = normalizer
#         self.taylor = taylor

#     def _reduce(self, group_imp):
#         if self.group_reduction == "sum":
#             group_imp = group_imp.sum(dim=0)
#         elif self.group_reduction == "mean":
#             group_imp = group_imp.mean(dim=0)
#         elif self.group_reduction == "max":
#             group_imp = group_imp.max(dim=0)[0]
#         elif self.group_reduction == "prod":
#             group_imp = torch.prod(group_imp, dim=0)
#         elif self.group_reduction=='first':
#             group_imp = group_imp[0]
#         elif self.group_reduction=='second':
#             group_imp = group_imp[1]
#         elif self.group_reduction is None:
#             group_imp = group_imp
#         else: 
#             raise NotImplementedError
#         return group_imp

#     @torch.no_grad()
#     def __call__(self, group, ch_groups=1, consecutive_groups=1):
    
#         group_imp = []
#         for dep, idxs in group:
#             idxs.sort()
#             layer = dep.target.module
#             prune_fn = dep.handler

#             if prune_fn not in [
#                 # tp.prune_linear_out_channels, tp.prune_linear_in_channels, 
#                 # hf_rmsnorm_pruner.prune_out_channels, tp.prune_embedding_out_channels, hf_attention_pruner.prune_out_channels,
#                 hf_linear_pruner.prune_out_channels, hf_linear_pruner.prune_in_channels
#             ]:
#                 continue
            
#             # if prune_fn in [hf_attention_pruner.prune_out_channels]:
#             #     salience = {}
#             #     for sub_layer in [layer.o_proj, layer.q_proj, layer.k_proj, layer.v_proj]:
#             #         salience[sub_layer] = sub_layer.weight * sub_layer.weight.grad
                    
#             #         if self.taylor in ['param_second']:
#             #             salience[sub_layer] = sub_layer.weight * sub_layer.weight.acc_grad * sub_layer.weight
#             #         elif self.taylor in ['param_mix']: 
#             #             salience[sub_layer] = -salience + 0.5 * sub_layer.weight * sub_layer.weight.acc_grad * sub_layer.weight   
#             # else:
#             salience = layer.weight * layer.weight.grad

#             if self.taylor in ['param_second']:
#                 salience = layer.weight * layer.weight.acc_grad * layer.weight
#             elif self.taylor in ['param_mix']: 
#                 salience = salience - 0.5 * layer.weight * layer.weight.acc_grad * layer.weight
                    
#             # Linear out_channels
#             if prune_fn in [tp.prune_linear_out_channels, hf_linear_pruner.prune_out_channels]:
#                 if self.taylor == 'vectorize':
#                     local_norm = salience.sum(1).abs()
#                 elif 'param' in self.taylor:
#                     local_norm = salience.abs().sum(1)
#                 else:
#                     raise NotImplementedError
#                 group_imp.append(local_norm)

#             # Linear in_channels
#             elif prune_fn in [tp.prune_linear_in_channels, hf_linear_pruner.prune_in_channels]:
#                 if self.taylor == 'vectorize':
#                     local_norm = salience.sum(0).abs()
#                 elif 'param' in self.taylor:
#                     local_norm = salience.abs().sum(0)
#                 else:
#                     raise NotImplementedError
#                 local_norm = local_norm[idxs]
#                 group_imp.append(local_norm)

#             # # RMSNorm
#             # elif prune_fn == hf_rmsnorm_pruner.prune_out_channels:
#             #     local_norm = salience.abs()
#             #     group_imp.append(local_norm)

#             # # Embedding
#             # elif prune_fn == tp.prune_embedding_out_channels:
#             #     if self.taylor == 'vectorize':
#             #         local_norm = salience[:, idxs].sum(0).abs()
#             #     elif 'param' in self.taylor:
#             #         local_norm = salience[:, idxs].abs().sum(0)
#             #     else:
#             #         raise NotImplementedError
#             #     group_imp.append(local_norm)

#             # # Attention
#             # elif prune_fn == hf_attention_pruner.prune_out_channels:
#             #     local_norm = 0
#             #     for sub_layer in [layer.o_proj]: #linear out channel, first dim in linear.weight
#             #         if self.taylor == 'vectorize':
#             #             local_norm += salience[sub_layer].sum(1).abs()
#             #         elif 'param' in self.taylor: 
#             #             local_norm += salience[sub_layer].abs().sum(1)   
#             #         else:
#             #             raise NotImplementedError                
                
#             #     for sub_layer in [layer.q_proj, layer.k_proj, layer.v_proj]: # linear in channel, second dim in linear.weight
#             #         if self.taylor == 'vectorize':
#             #             local_norm += salience[sub_layer].sum(0).abs() 
#             #         elif 'param' in self.taylor == 'param':
#             #             local_norm += salience[sub_layer].abs().sum(0)
#             #         else:
#             #             raise NotImplementedError
#             #     group_imp.append(local_norm)

#         if len(group_imp)==0:
#             return None

#         min_imp_size = min([len(imp) for imp in group_imp])
#         aligned_group_imp = []
#         for imp in group_imp:
#             if len(imp)>min_imp_size and len(imp)%min_imp_size==0:
#                 imp = imp.view(len(imp) // min_imp_size, min_imp_size).sum(0)
#                 aligned_group_imp.append(imp)
#             elif len(imp)==min_imp_size:
#                 aligned_group_imp.append(imp)
#         group_imp = torch.stack(aligned_group_imp, dim=0)
#         group_imp = self._reduce(group_imp)
#         if self.normalizer is not None:
#             group_imp = self.normalizer(group, group_imp)
#         return group_imp
    
    
# class RandomImportance(Importance):
#     @torch.no_grad()
#     def __call__(self, group, **kwargs):
#         _, idxs = group[0]
#         return torch.rand(len(idxs))
    

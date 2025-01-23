import torch
import argparse

from load_model import load_model
from transformers import GPT2TokenizerFast
import torch.nn.functional as F
import sampling
import pdb

import numpy as np
import os
import datasets
from model import utils as mutils
from losses import get_loss_fn, get_loss_fn_conditional
from tqdm import tqdm

EOS = 50256

def generate_span_mask(seq_length, mask_prob=0.5, fixed_infilling=False, mean_span_length=None, max_num_spans=None):
    if mean_span_length is not None:
        num_spans = seq_length // mean_span_length
    else:
        assert max_num_spans is not None, "Either num_spans or max_num_spans should be provided."
        num_spans = np.random.randint(1, max_num_spans)

    # assume all samples in the batch use the same span mask
    num_to_mask = max(1, int(round(mask_prob * seq_length)))
    num_masked = 0
    blank_ids = []
    spans = []

    if fixed_infilling:
        start = seq_length // 4
        end = seq_length * 3 // 4
        spans.append((start, end))
        blank_ids.extend(range(start, end))
        num_masked += len(range(start, end))
   
    
    while num_masked < num_to_mask:
        start = np.random.randint(0, seq_length)
        # check if the start index is already part of a span
        if any(start >= span[0] and start < span[1] for span in spans):
            continue
        # sample span length
        span_length = np.random.geometric(1.0 / mean_span_length)
        end = min(start + span_length, seq_length)
        # check if the span overlaps with any existing span
        if any(idx in blank_ids for idx in range(start, end)):
            continue
        spans.append((start, end))
        blank_ids.extend(range(start, end))
        num_masked += len(range(start, end))
    
    # if the number of masked tokens is more than the target number, remove the excess for the last span
    if num_masked > num_to_mask:
        excess = num_masked - num_to_mask
        last_span_start, last_span_end = spans[-1]
        spans[-1] = (last_span_start, last_span_end - excess)
        blank_ids = blank_ids[:-excess]

    infill_mask = torch.zeros(seq_length).to(torch.long)
    infill_mask[blank_ids] = 1
    
    return infill_mask

def generate_uniform_mask(seq_length, mask_prob=0.5):
    mask = (torch.rand(seq_length) < mask_prob).long()
    return mask

def set_seed(seed):
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)

def get_dataset(dataset_name, mean_span_length, mask_prob, cache_dir='/nfs-shared-2/anji/data', block_size=1024, mask_type='span'):

	_path = os.path.join(cache_dir, dataset_name)

	filename = None
	if dataset_name == 'wikitext':
		filename = 'test.bin'
	elif dataset_name == 'wikitext2':
		filename = 'train.bin'
	elif dataset_name == 'ptb':
		filename = 'test.bin'
	elif dataset_name == '1bw':
		filename = 'test.bin'
	elif dataset_name == 'lambada':
		filename = 'test.bin'
	elif dataset_name == 'openwebtext':
		filename = 'val.bin'

	dataset = np.memmap(os.path.join(_path, filename), dtype=np.uint16, mode='r')
	dataset = torch.from_numpy(dataset.astype(np.int32))

	total_length = len(dataset)
	total_length = (total_length // block_size) * block_size

	chunked_dataset = {'input_ids': [], 'infill_mask': []}
	for i in range(0, total_length, block_size):

		chunk = dataset[i:i + block_size]
		chunked_dataset['input_ids'].append(chunk)

		# Infill mask - places where mask is 1 will be masked i.e. needs to be predicted
		if mask_type == 'span':
			assert isinstance(mean_span_length, int)
			infill_mask = generate_span_mask(block_size, mask_prob=mask_prob, mean_span_length=mean_span_length)
		elif mask_type == 'front_half':
			infill_mask = torch.ones(block_size).to(torch.long)
			infill_mask[:block_size//2] = 0
		elif mask_type == 'prefix':
			infill_mask = torch.ones(block_size).to(torch.long)
			prefix_len = int(block_size * (1 - mask_prob))
			infill_mask[:prefix_len] = 0
		chunked_dataset['infill_mask'].append(infill_mask)

	chunked_dataset = datasets.Dataset.from_dict(chunked_dataset)

	return chunked_dataset.with_format('torch')

def log_args(args, file_handler):
	for arg in vars(args):
		print(f"> {arg}: {getattr(args, arg)}", file=file_handler, flush=True)

def main(args):
	# Configurations
	set_seed(args.seed)
	log_dir = f'logs/{args.model_size}/{args.dataset_name}/{args.block_size}/'
	os.makedirs(log_dir, exist_ok=True)
	if args.include_eos:
		if args.mask_type == 'span':
			log_filename = log_dir + f'{args.mean_span_length}_{args.mask_prob}_with_eos.txt'
		elif args.mask_type == 'front_half':
			log_filename = log_dir + f'front_half_with_eos.txt'
		elif args.mask_type == 'prefix':
			log_filename = log_dir + f'prefix_{args.mask_prob}_with_eos.txt'
		print("*********With EOS*********", flush=True)
	else:
		if args.mask_type == 'span':
			log_filename = log_dir + f'{args.mean_span_length}_{args.mask_prob}_without_eos.txt'
		elif args.mask_type == 'front_half':
			log_filename = log_dir + f'front_half_without_eos.txt'
		elif args.mask_type == 'prefix':
			log_filename = log_dir + f'prefix_{args.mask_prob}_without_eos.txt'
		print("*********Without EOS*********", flush=True)
	file_handler = open(log_filename, 'w')
	print(f"Currently evaluating: {args.dataset_name} with mean mask length {args.mean_span_length} and mask prob {args.mask_prob}", flush=True)
	log_args(args, file_handler)

	# Evaluation
	device = torch.device('cuda')
	model, graph, noise = load_model('louaaron/sedd-' + args.model_size, device)
	dataset = get_dataset(args.dataset_name, args.mean_span_length, args.mask_prob, mask_type=args.mask_type, block_size=args.block_size)
	dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

	nlls = []
	loss_fn = get_loss_fn_conditional(noise, graph, train=False)
	for batch in tqdm(dataloader):
		input_ids = batch['input_ids'].to(device)
		infill_mask = batch['infill_mask'].to(device)
		for _ in range(args.repeat_rounds):
			loss = loss_fn(model, input_ids, infill_mask)

			valid_mask = infill_mask
			if not args.include_eos:
				valid_mask[input_ids == EOS] = 0
			loss = (loss * valid_mask).sum(dim=-1)
			valid_cnt = valid_mask.sum(dim=-1)
			loss = loss / valid_cnt

			nlls.extend(loss.tolist())

	nll = np.mean(nlls)
	ppl = np.exp(nll)

	# Logging
	print(f"> NLL: {nll}\n> PPL: {ppl}\n", file=file_handler, flush=True)
	file_handler.close()

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Evaluate Conditional PPL')
	parser.add_argument('--seed', type=int, default=6171, help='Random seed')
	parser.add_argument('--model_size', type=str, default='small', help='Model size')
	parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
	parser.add_argument('--repeat_rounds', type=int, default=100, help='Number of repeat rounds')
	parser.add_argument('--include_eos', action='store_true', help='Whether to include EOS tokens in PPL calculation')
	parser.add_argument('--mean_span_length', type=int, help='Infill mask mean span length')
	parser.add_argument('--mask_prob', type=float, help='Probability of mask')
	parser.add_argument('--dataset_name', type=str, help='Dataset name')
	parser.add_argument('--mask_type', type=str, default='span', help='Type of mask')
	parser.add_argument('--block_size', type=int, default=1024, help='Block size')
	args = parser.parse_args()
	main(args)
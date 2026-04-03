import copy
import os
from typing import Dict

import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence

from src.dataset.utils import normalize_masked_tp, split_and_patch_batch, setup_seed



class USHCN(object):
    """
    variables:
    "SNOW","SNWD","PRCP","TMAX","TMIN"
    """
    def __init__(self, root, n_samples = None, device = torch.device("cpu")):

        self.root = root
        self.device = device

        self.process()

        if device == torch.device("cpu"):
            self.data = torch.load(os.path.join(self.processed_folder, 'ushcn.pt'), map_location='cpu')
        else:
            self.data = torch.load(os.path.join(self.processed_folder, 'ushcn.pt'))

        if n_samples is not None:
            print('Total records:', len(self.data))
            self.data = self.data[:n_samples]

    def process(self):
        if self._check_exists():
            return
        
        filename = os.path.join(self.raw_folder, 'small_chunked_sporadic.csv')
        
        os.makedirs(self.processed_folder, exist_ok=True)

        print('Processing {}...'.format(filename))

        full_data = pd.read_csv(filename, index_col=0)
        full_data.index = full_data.index.astype('int32')

        entities = []
        value_cols = [c.startswith('Value') for c in full_data.columns]
        value_cols = list(full_data.columns[value_cols])
        mask_cols = [('Mask' + x[5:]) for x in value_cols]
        # print(value_cols)
        # print(mask_cols)
        data_gp = full_data.groupby(level=0) # group by index
        for record_id, data in data_gp:
            tt = torch.tensor(data['Time'].values).to(self.device).float() * (48./200)
            sorted_inds = tt.argsort() # sort over time
            vals = torch.tensor(data[value_cols].values).to(self.device).float()
            mask = torch.tensor(data[mask_cols].values).to(self.device).float()
            entities.append((record_id, tt[sorted_inds], vals[sorted_inds], mask[sorted_inds]))

        torch.save(
            entities,
            os.path.join(self.processed_folder, 'ushcn.pt')
        )

        print('Total records:', len(entities))

        print('Done!')

    def _check_exists(self):

        if not os.path.exists(os.path.join(self.processed_folder, 'ushcn.pt')):
            return False
        
        return True

    @property
    def raw_folder(self):
        return os.path.join(self.root, 'raw')

    @property
    def processed_folder(self):
        return os.path.join(self.root, 'processed')
        
    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
    


def USHCN_time_chunk(data, args, device):
	chunk_data = []
	zero_indices_elem = 0
	mask_zeros = 0
	for b, (record_id, tt, vals, mask) in enumerate(data):
		for st in range(0, args.n_months - args.history - args.pred_window + 1, args.pred_window):
			et = st + args.history + args.pred_window
			if(et == args.n_months):
				indices = torch.where((tt >= st) & (tt <= et))[0]
			else:
				indices = torch.where((tt >= st) & (tt < et))[0]

			t_bias = torch.tensor(st).to(device)

			if len(indices) == 0:
				zero_indices_elem += 1
			elif mask[indices].sum() == 0:
				mask_zeros += 1
			else:
				chunk_data.append((record_id, tt[indices] - t_bias, vals[indices], mask[indices], t_bias))

	print(f'Elems with zero indices: {zero_indices_elem}; elem with mask equal to zeros: {mask_zeros}, Total: {zero_indices_elem+mask_zeros}')
	return chunk_data


def USHCN_patch_variable_time_collate_fn(batch_, args, device=torch.device("cpu"), data_type="train",
                                         data_min=None, data_max=None, time_max=None):
	"""
	Expects a batch of time series data in the form of (record_id, tt, vals, mask) where
		- record_id is a patient id
		- tt is a (T, ) tensor containing T time values of observations.
		- vals is a (T, D) tensor containing observed values for D variables.
		- mask is a (T, D) tensor containing 1 where values were observed and 0 otherwise.
	Returns:
	Data form as input:
		batch_tt: (B, M, L_in, D) the batch contains a maximal L_in time values of observations among M patches.
		batch_vals: (B, M, L_in, D) tensor containing the observed values.
		batch_mask: (B, M, L_in, D) tensor containing 1 where values were observed and 0 otherwise.
	Data form to predict:
		flat_tt: (L_out) the batch contains a maximal L_out time values of observations.
		flat_vals: (B, L_out, D) tensor containing the observed values.
		flat_mask: (B, L_out, D) tensor containing 1 where values were observed and 0 otherwise.
	"""
	batch = list()
	for sample in batch_:
		elem_tuple = list()
		for elem in sample:
			if isinstance(elem, torch.Tensor):
				elem_tuple.append(elem.to(device))
			else:
				elem_tuple.append(elem)
		batch.append(tuple(elem_tuple))

	D = batch[0][2].shape[1]
	combined_tt, inverse_indices = torch.unique(torch.cat([ex[1] for ex in batch]), sorted=True, return_inverse=True)
	n_observed_tp = torch.lt(combined_tt, args.history).sum()

	offset = 0
	combined_vals = torch.zeros([len(batch), len(combined_tt), D]).to(device)
	combined_mask = torch.zeros([len(batch), len(combined_tt), D]).to(device)
	predicted_tp = []
	predicted_data = []
	predicted_mask = []
	batch_t_bias = []
	for b, (record_id, tt, vals, mask, t_bias) in enumerate(batch):
		batch_t_bias.append(t_bias)

		indices = inverse_indices[offset:offset + len(tt)]
		offset += len(tt)
		combined_vals[b, indices] = vals
		combined_mask[b, indices] = mask
		# if combined_mask.sum() == 0.:
		# 	print('hey')

		tmp_n_observed_tp = torch.lt(tt, args.history).sum()
		predicted_tp.append(tt[tmp_n_observed_tp:])
		predicted_data.append(vals[tmp_n_observed_tp:])
		predicted_mask.append(mask[tmp_n_observed_tp:])

	combined_tt = combined_tt[:n_observed_tp]  # (T_o, )
	combined_vals = combined_vals[:, :n_observed_tp]
	combined_mask = combined_mask[:, :n_observed_tp]  # qui sta il problema, se dopo questo slice la parte prima ha tutti zeri ho il problema
	predicted_tp = pad_sequence(predicted_tp, batch_first=True)
	predicted_data = pad_sequence(predicted_data, batch_first=True)
	predicted_mask = pad_sequence(predicted_mask, batch_first=True)

	combined_tt = normalize_masked_tp(combined_tt, att_min=0, att_max=time_max)
	predicted_tp = normalize_masked_tp(predicted_tp, att_min=0, att_max=time_max)
	batch_t_bias = torch.stack(batch_t_bias)  # (n_batch, )
	batch_t_bias = normalize_masked_tp(batch_t_bias, att_min=0, att_max=time_max)

	data_dict = {
		"data": combined_vals,  # (n_batch, T_o, D)
		"time_steps": combined_tt,  # (T_o, )
		"mask": combined_mask,  # (n_batch, T_o, D)
		"data_to_predict": predicted_data,  # (n_batch, T, D)
		"tp_to_predict": predicted_tp,  # (B, T)
		"mask_predicted_data": predicted_mask,  # (n_batch, T, D)
	}

	split_dict = {"tp_to_predict": data_dict["tp_to_predict"].clone(),
	              "data_to_predict": data_dict["data_to_predict"].clone(),
	              "mask_predicted_data": data_dict["mask_predicted_data"].clone()
	              }

	observed_tp = data_dict["time_steps"].clone()  # (n_observed_tp, )
	observed_data = data_dict["data"].clone()  # (bs, n_observed_tp, D)
	observed_mask = data_dict["mask"].clone()  # (bs, n_observed_tp, D)

	n_batch, n_tp, n_dim = observed_data.shape
	observed_tp_patches = observed_tp.view(1, -1, 1).repeat(n_batch, 1, n_dim)
	observed_data_patches = observed_data
	observed_mask_patches = observed_mask
	max_patch_len = int(observed_mask.sum(dim=1).max().item())

	patch_indices_final = torch.full((n_batch, max_patch_len, n_dim), n_tp).to(device)  # n_batch, npacth, max_patch_len, n_dim
	aux_tensor = torch.arange(max_patch_len).view(1, max_patch_len, 1).repeat(n_batch, 1, n_dim).to(device)

	observed_mask_patches_fill = observed_mask
	L = observed_mask.sum(dim=1, keepdim=True)  # (bs, 1, D)
	observed_mask_patches_fill_reindex = (aux_tensor < L)  # let first L[i] to be True

	# Return a indices tuple like ([...], [...], [...])
	mask_inds = torch.nonzero(observed_mask_patches_fill_reindex.permute(0, 2, 1), as_tuple=True)  # reset indices
	ind_values = torch.nonzero(observed_mask_patches_fill.permute(0, 2, 1), as_tuple=True)[-1]  # original indices of dimension 2

	# Fill n_tp if the number of observed points are less than max_patch_len
	patch_indices_final.index_put_((mask_inds[0], mask_inds[2], mask_inds[1]), ind_values)

	pad_zeros_data = torch.zeros([n_batch, 1, n_dim]).to(device)
	observed_tp_patches = torch.cat([observed_tp_patches, pad_zeros_data], dim=1).gather(1,
	                                                                                     patch_indices_final)  # (n_batch, max_patch_len, n_dim)
	observed_data_patches = torch.cat([observed_data_patches, pad_zeros_data], dim=1).gather(1, patch_indices_final)
	observed_mask_patches = torch.cat([observed_mask_patches, pad_zeros_data], dim=1).gather(1, patch_indices_final)
	# if (observed_mask_patches.sum(dim=(1, 2)) == 0).any():
	# 	print('Hey!')

	split_dict["observed_tp"] = observed_tp_patches
	split_dict["observed_data"] = observed_data_patches
	split_dict["observed_mask"] = observed_mask_patches

	split_dict["observed_tp"] = split_dict["observed_tp"] + batch_t_bias.view(len(batch_t_bias), 1, 1)
	split_dict["tp_to_predict"] = split_dict["tp_to_predict"] + batch_t_bias.view(len(batch_t_bias), 1)
	split_dict["tp_to_predict"][split_dict["mask_predicted_data"].sum(dim=-1) < 1e-8] = 0

	split_dict = check_data_integrity(split_dict)
	return split_dict

def check_data_integrity(split_dict: Dict):
	corrupted_id_set = set()
	new_split_dict = dict()
	batch_id_set = set(torch.arange(split_dict["tp_to_predict"].shape[0]).tolist())
	backup_split_dict = copy.deepcopy(split_dict)  # debug

	# Controlla se dopo il processing della dimensione temporale (past-future values) i tensori di input o output siano
	# non totalmente pieni di soli zero o dimensioni 0
	for data_type in split_dict:
		if 'tp' in data_type:
			continue
		elem = split_dict[data_type]

		# Check if there are zeros samples
		if (elem.sum(dim=(1, 2)) == 0).any():
			corrupted_id = torch.where(elem.sum(dim=(1, 2)) == 0)
			corrupted_id_set.update(corrupted_id[0].tolist())

	if len(corrupted_id_set) > 0:
		# print('Sample with all zeros!')
		non_corrupted_id = batch_id_set - corrupted_id_set
		for data_type in split_dict:
			new_split_dict[data_type] = split_dict[data_type][[list(non_corrupted_id)]]

		split_dict = new_split_dict.copy()
		return split_dict  # ricordati che qui puoi lasciare cosi e ottieni batch piu piccole (però forse in evaluation.py, r:43,47 devi mettere mean

	for elem in split_dict:
		if 0 in split_dict[elem].shape:
			# print('Batch with 0 timestep!')
			return None

	return split_dict

def USHCN_patch_variable_time_collate_fn_old(batch, args, device = torch.device("cpu"), data_type = "train",
	data_min = None, data_max = None, time_max = None):
	"""
	Expects a batch of time series data in the form of (record_id, tt, vals, mask) where
		- record_id is a patient id
		- tt is a (T, ) tensor containing T time values of observations.
		- vals is a (T, D) tensor containing observed values for D variables.
		- mask is a (T, D) tensor containing 1 where values were observed and 0 otherwise.
	Returns:
	Data form as input:
		batch_tt: (B, M, L_in, D) the batch contains a maximal L_in time values of observations among M patches.
		batch_vals: (B, M, L_in, D) tensor containing the observed values.
		batch_mask: (B, M, L_in, D) tensor containing 1 where values were observed and 0 otherwise.
	Data form to predict:
		flat_tt: (L_out) the batch contains a maximal L_out time values of observations.
		flat_vals: (B, L_out, D) tensor containing the observed values.
		flat_mask: (B, L_out, D) tensor containing 1 where values were observed and 0 otherwise.
	"""

	D = batch[0][2].shape[1]
	# combined_tt shape is (T_o, )
	combined_tt, inverse_indices = torch.unique(torch.cat([ex[1] for ex in batch]), sorted=True, return_inverse=True)
	# print(combined_tt.max(), combined_tt.min())
	# print(inverse_indices.shape, np.sum([len(ex[1]) for ex in batch]), inverse_indices.max())
	# print(inverse_indices)

	# the number of observed time points 
	n_observed_tp = torch.lt(combined_tt, args.history).sum()
	observed_tp = combined_tt[:n_observed_tp] # (n_observed_tp, )
	# print(n_observed_tp, len(combined_tt)-n_observed_tp)
	# print(combined_tt[:n_observed_tp])
	# print(combined_tt[n_observed_tp:])

	patch_indices = []
	st, ed = 0, args.patch_size
	for i in range(args.npatch):
		if(i == args.npatch-1):
			inds = torch.where((observed_tp >= st) & (observed_tp <= ed))[0]
		else:
			inds = torch.where((observed_tp >= st) & (observed_tp < ed))[0]
		patch_indices.append(inds)
		# print(st, ed, observed_tp[inds[0]: inds[-1]+1])

		st += args.stride
		ed += args.stride

	offset = 0
	combined_vals = torch.zeros([len(batch), len(combined_tt), D]).to(device)
	combined_mask = torch.zeros([len(batch), len(combined_tt), D]).to(device)
	predicted_tp = []
	predicted_data = []
	predicted_mask = [] 
	batch_t_bias = []
	for b, (record_id, tt, vals, mask, t_bias) in enumerate(batch):
		batch_t_bias.append(t_bias)

		indices = inverse_indices[offset:offset+len(tt)]
		offset += len(tt)
		combined_vals[b, indices] = vals
		combined_mask[b, indices] = mask

		tmp_n_observed_tp = torch.lt(tt, args.history).sum()
		predicted_tp.append(tt[tmp_n_observed_tp:])
		predicted_data.append(vals[tmp_n_observed_tp:])
		predicted_mask.append(mask[tmp_n_observed_tp:])

	combined_tt = combined_tt[:n_observed_tp] # (T_o, )
	combined_vals = combined_vals[:, :n_observed_tp]
	combined_mask = combined_mask[:, :n_observed_tp]
	predicted_tp = pad_sequence(predicted_tp, batch_first=True)
	predicted_data = pad_sequence(predicted_data, batch_first=True)
	predicted_mask = pad_sequence(predicted_mask, batch_first=True)


	combined_tt = normalize_masked_tp(combined_tt, att_min = 0, att_max = time_max)
	predicted_tp = normalize_masked_tp(predicted_tp, att_min = 0, att_max = time_max)
	# print(predicted_data.sum(), predicted_tp.sum())
	batch_t_bias = torch.stack(batch_t_bias) # (n_batch, )
	batch_t_bias = normalize_masked_tp(batch_t_bias, att_min = 0, att_max = time_max)
		
	data_dict = {
		"data": combined_vals, # (n_batch, T_o, D)
		"time_steps": combined_tt, # (T_o, )
		"mask": combined_mask, # (n_batch, T_o, D)
		"data_to_predict": predicted_data, # (n_batch, T, D)
		"tp_to_predict": predicted_tp, # (B, T)
		"mask_predicted_data": predicted_mask, # (n_batch, T, D)
		}

	data_dict = split_and_patch_batch(data_dict, args, n_observed_tp, patch_indices)
	# print("patchdata:", data_dict["data_to_predict"].sum(), data_dict["mask_predicted_data"].sum())

	# print(batch_t_bias.shape, data_dict["observed_tp"].shape, data_dict["tp_to_predict"].shape)
	data_dict["observed_tp"] = data_dict["observed_tp"] + batch_t_bias.view(len(batch_t_bias), 1, 1, 1)
	# data_dict["observed_tp"] = data_dict["observed_tp"] * (data_dict["mask_predicted_data"].sum(dim=-1)>1e-8)

	data_dict["tp_to_predict"] = data_dict["tp_to_predict"] + batch_t_bias.view(len(batch_t_bias), 1)
	data_dict["tp_to_predict"][data_dict["mask_predicted_data"].sum(dim=-1)<1e-8] = 0
	# delta = data_dict["tp_to_predict"].view(len(batch_t_bias),-1).max(dim=-1)[0] - data_dict["observed_tp"].view(len(batch_t_bias),-1).min(dim=-1)[0]
	# delta = data_dict["tp_to_predict"].view(len(batch_t_bias),-1).min(dim=-1)[0] - data_dict["observed_tp"].view(len(batch_t_bias),-1).max(dim=-1)[0]
	# print((delta*48).max(), (delta*48).min())
	return data_dict


def USHCN_variable_time_collate_fn(batch, args, device = torch.device("cpu"), data_type = "train", 
	data_min = None, data_max = None, time_max = None):
	"""
	Expects a batch of time series data in the form of (record_id, tt, vals, mask) where
		- record_id is a patient id
		- tt is a (T, ) tensor containing T time values of observations.
		- vals is a (T, D) tensor containing observed values for D variables.
		- mask is a (T, D) tensor containing 1 where values were observed and 0 otherwise.
	Returns:
		batch_tt: (B, L) the batch contains a maximal L time values of observations.
		batch_vals: (B, L, D) tensor containing the observed values.
		batch_mask: (B, L, D) tensor containing 1 where values were observed and 0 otherwise.
	"""

	# n_observed_tps = []
	observed_tp = []
	observed_data = []
	observed_mask = [] 
	predicted_tp = []
	predicted_data = []
	predicted_mask = [] 
	# batch_t_bias = []

	for b, (record_id, tt, vals, mask, t_bias) in enumerate(batch):
		# batch_t_bias.append(t_bias)
		n_observed_tp = torch.lt(tt, args.history).sum()
		# print(len(tt), n_observed_tp)
		# n_observed_tps.append(n_observed_tp)
		tt = tt + t_bias
		observed_tp.append(tt[:n_observed_tp])
		observed_data.append(vals[:n_observed_tp])
		observed_mask.append(mask[:n_observed_tp])
		
		predicted_tp.append(tt[n_observed_tp:])
		predicted_data.append(vals[n_observed_tp:])
		predicted_mask.append(mask[n_observed_tp:])  # anche qui problema, se non ci sono valori (prima o dopo), non va considerato
		# aggiungi qui taglio dati corrotti

	observed_tp = pad_sequence(observed_tp, batch_first=True)
	observed_data = pad_sequence(observed_data, batch_first=True)
	observed_mask = pad_sequence(observed_mask, batch_first=True)
	predicted_tp = pad_sequence(predicted_tp, batch_first=True)
	predicted_data = pad_sequence(predicted_data, batch_first=True)
	predicted_mask = pad_sequence(predicted_mask, batch_first=True)
	# print(observed_tp.shape, observed_data.shape, observed_mask.shape,\
	#     predicted_tp.shape, predicted_data.shape, predicted_mask.shape)
	
	observed_tp = normalize_masked_tp(observed_tp, att_min = 0, att_max = time_max)
	predicted_tp = normalize_masked_tp(predicted_tp, att_min = 0, att_max = time_max)
	# print(predicted_data.sum(), predicted_tp.sum())
	# batch_t_bias = torch.stack(batch_t_bias) # (n_batch, )
	# batch_t_bias = utils.normalize_masked_tp(batch_t_bias, att_min = 0, att_max = time_max)

	# print(observed_tp.max())
	# print(predicted_tp.max())

	# print(batch_t_bias.shape, observed_tp.shape, predicted_tp.shape)
	# observed_tp = observed_tp + batch_t_bias.view(len(batch_t_bias), 1)
	# observed_tp[observed_mask.sum(dim=-1)<1e-8] = 0
	# predicted_tp = predicted_tp + batch_t_bias.view(len(batch_t_bias), 1)
	# predicted_tp[predicted_mask.sum(dim=-1)<1e-8] = 0
		
	data_dict = {"observed_data": observed_data,
			"observed_tp": observed_tp,
			"observed_mask": observed_mask,
			"data_to_predict": predicted_data,
			"tp_to_predict": predicted_tp,
			"mask_predicted_data": predicted_mask,
			}
	# print("vecdata:", data_dict["data_to_predict"].sum(), data_dict["mask_predicted_data"].sum())
	data_dict = check_data_integrity(data_dict)
	return data_dict


def USHCN_get_seq_length(args, records):
	
	max_input_len = 0
	max_pred_len = 0
	lens = []
	for b, (record_id, tt, vals, mask, t_bias) in enumerate(records):
		n_observed_tp = torch.lt(tt, args.history).sum()
		max_input_len = max(max_input_len, n_observed_tp)
		max_pred_len = max(max_pred_len, len(tt) - n_observed_tp)
		lens.append(n_observed_tp)
	lens = torch.stack(lens, dim=0)
	median_len = lens.median()

	return max_input_len, max_pred_len, median_len



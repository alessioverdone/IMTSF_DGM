from src.dataset.utils import inf_generator
from src.dataset.physionet import *
from src.dataset.ushcn import *
from src.dataset.mimic import MIMIC
from src.dataset.person_activity import *
from sklearn import model_selection

def parse_datasets(args, patch_ts=False, length_stat=False):
	### PhysioNet and MIMIC dataset ###
	if args.dataset_name in ["physionet", "mimic"]:
		args.pred_window = 48 - args.history
		### list of tuples (record_id, tt, vals, mask) ###
		if args.dataset_name == "physionet":
			total_dataset = PhysioNet(os.path.join(args.data_dir, args.dataset_name),
									  quantization=args.quantization,
									  download=True,
									  n_samples=args.n,
									  device=args.device)
			# total_dataset = total_dataset[:1000]
		elif args.dataset_name == "mimic":
			total_dataset = MIMIC(os.path.join(args.data_dir, args.dataset_name),
								  n_samples=args.n,
								  device=args.device)

		seen_data, test_data = model_selection.train_test_split(total_dataset,
																train_size= 0.8,
																random_state = 42,
																shuffle = True)
		train_data, val_data = model_selection.train_test_split(seen_data,
																train_size= 0.75,
																random_state = 42,
																shuffle = False)
		print("Dataset n_samples:", len(total_dataset), len(train_data), len(val_data), len(test_data))
		test_record_ids = [record_id for record_id, tt, vals, mask in test_data]
		print("Test record ids (first 20):", test_record_ids[:20])
		print("Test record ids (last 20):", test_record_ids[-20:])

		record_id, tt, vals, mask = train_data[0]
		n_samples = len(total_dataset)
		input_dim = vals.size(-1)

		batch_size = min(min(len(seen_data), args.batch_size), args.n)
		print(f'Batch size = {batch_size}')
		data_min, data_max, time_max = get_data_min_max(seen_data, device='cpu') # (n_dim,), (n_dim,)
		x_mean = None

		if(patch_ts):
			collate_fn = patch_variable_time_collate_fn
		else:
			collate_fn = variable_time_collate_fn

		train_dataloader = DataLoader(train_data,
									  batch_size= batch_size,
									  shuffle=True,
									  num_workers=4,
									  collate_fn= lambda batch: collate_fn(batch,
																		   args,
																		   'cpu',
																		   data_type = "train",
																		   data_min = data_min,
																		   data_max = data_max,
																		   time_max = time_max))

		val_dataloader = DataLoader(val_data,
									batch_size= batch_size,
									shuffle=False,
									num_workers=4,
									collate_fn= lambda batch: collate_fn(batch,
																		 args,
																		 'cpu',
																		 data_type = "val",
																		 data_min = data_min,
																		 data_max = data_max,
																		 time_max = time_max))
		test_dataloader = DataLoader(test_data,
									 batch_size = batch_size,
									 shuffle=False,
									 num_workers=4,
									 collate_fn= lambda batch: collate_fn(batch,
																		  args,
																		  'cpu',
																		  data_type = "test",
																		  data_min = data_min,
																		  data_max = data_max,
																		  time_max = time_max))

		data_objects = {
					"train_dataloader": inf_generator(train_dataloader),
					"val_dataloader": inf_generator(val_dataloader),
					"test_dataloader": inf_generator(test_dataloader),
					"input_dim": input_dim,
					"n_train_batches": len(train_dataloader),
					"n_val_batches": len(val_dataloader),
					"n_test_batches": len(test_dataloader),
					"data_max": data_max, #optional
					"data_min": data_min,
					"time_max": time_max,
					'x_mean': x_mean
					} #optional

		if(length_stat):
			max_input_len, max_pred_len, median_len = get_seq_length(args, total_dataset)
			data_objects["max_input_len"] = max_input_len.item()
			data_objects["max_pred_len"] = max_pred_len.item()
			data_objects["median_len"] = median_len.item()
			print(data_objects["max_input_len"], data_objects["max_pred_len"], data_objects["median_len"])

		return data_objects

	##################################################################
	### USHCN dataset ###
	elif args.dataset_name == "ushcn":
		args.n_months = 48 # 48 months
		# args.pred_window = 1 # predict future one month

		### list of tuples (record_id, tt, vals, mask) ###
		total_dataset = USHCN(os.path.join(args.data_dir, args.dataset_name),
							  n_samples = args.n,
							  device = 'cpu')

		seen_data, test_data = model_selection.train_test_split(total_dataset,
																train_size= 0.8,
																random_state = 42,
																shuffle = True)
		train_data, val_data = model_selection.train_test_split(seen_data,
																train_size= 0.75,
																random_state = 42,
																shuffle = False)
		print("Dataset n_samples:", len(total_dataset), len(train_data), len(val_data), len(test_data))
		test_record_ids = [record_id for record_id, tt, vals, mask in test_data]
		print("Test record ids (first 20):", test_record_ids[:20])
		print("Test record ids (last 20):", test_record_ids[-20:])

		record_id, tt, vals, mask = train_data[0]
		n_samples = len(total_dataset)
		input_dim = vals.size(-1)
		data_min, data_max, time_max = get_data_min_max(seen_data, 'cpu')  # (n_dim,), (n_dim,)
		x_mean = None

		if(patch_ts):
			collate_fn = USHCN_patch_variable_time_collate_fn
		else:
			collate_fn = USHCN_variable_time_collate_fn

		train_data = USHCN_time_chunk(train_data, args, 'cpu')
		val_data = USHCN_time_chunk(val_data, args, 'cpu')
		test_data = USHCN_time_chunk(test_data, args, 'cpu')
		batch_size = args.batch_size
		print("Dataset n_samples after time split:", len(train_data)+len(val_data)+len(test_data),\
			len(train_data), len(val_data), len(test_data))
		train_dataloader = DataLoader(train_data,
									  batch_size= batch_size,
									  shuffle=True,
									  collate_fn= lambda batch: collate_fn(batch,
																		   args,
																		   'cpu',
																		   time_max = time_max,
																		   data_max=data_max,
																		   data_min=data_min))
		val_dataloader = DataLoader(val_data,
									batch_size= batch_size,
									shuffle=False,
									collate_fn= lambda batch: collate_fn(batch,
																		 args,
																		 'cpu',
																		 time_max = time_max,
																		 data_max=data_max,
																		 data_min=data_min))
		test_dataloader = DataLoader(test_data,
									 batch_size = batch_size,
									 shuffle=False,
									 collate_fn= lambda batch: collate_fn(batch,
																		  args,
																		  'cpu',
																		  time_max = time_max,
																		  data_max=data_max,
																		  data_min=data_min))

		data_objects = {
					"train_dataloader": inf_generator(train_dataloader),
					"val_dataloader": inf_generator(val_dataloader),
					"test_dataloader": inf_generator(test_dataloader),
					"input_dim": input_dim,
					"n_train_batches": len(train_dataloader),
					"n_val_batches": len(val_dataloader),
					"n_test_batches": len(test_dataloader),
					"data_max": data_max, #optional
					"data_min": data_min,
					"time_max": time_max,
					'x_mean': x_mean
		} #optional

		if(length_stat):
			max_input_len, max_pred_len, median_len = USHCN_get_seq_length(args, train_data+val_data+test_data)
			data_objects["max_input_len"] = max_input_len.item()
			data_objects["max_pred_len"] = max_pred_len.item()
			data_objects["median_len"] = median_len.item()
			print(data_objects["max_input_len"], data_objects["max_pred_len"], data_objects["median_len"])

		return data_objects
		

	##################################################################
	### Activity dataset ###
	elif args.dataset_name == "activity":
		args.pred_window = 4000 - args.history # predict future 1000 ms

		total_dataset = PersonActivity(os.path.join(args.data_dir, args.dataset_name),
									   n_samples = args.n,
									   download=True,
									   device = 'cpu')

		# Shuffle and split
		seen_data, test_data = model_selection.train_test_split(total_dataset,
																train_size= 0.8,
																random_state = 42,
																shuffle = True)
		train_data, val_data = model_selection.train_test_split(seen_data,
																train_size= 0.75,
																random_state = 42,
																shuffle = False)
		print("Dataset n_samples:", len(total_dataset), len(train_data), len(val_data), len(test_data))
		test_record_ids = [record_id for record_id, tt, vals, mask in test_data]
		print("Test record ids (first 20):", test_record_ids[:20])
		print("Test record ids (last 20):", test_record_ids[-20:])

		record_id, tt, vals, mask = train_data[0]

		n_samples = len(total_dataset)
		input_dim = vals.size(-1)

		batch_size = min(min(len(seen_data), args.batch_size), args.n)
		data_min, data_max, _ = get_data_min_max(seen_data, 'cpu')  # (n_dim,), (n_dim,)
		x_mean = None

		time_max = torch.tensor(args.history + args.pred_window)
		print('manual set time_max:', time_max)

		if(patch_ts):
			collate_fn = patch_variable_time_collate_fn
		else:
			collate_fn = variable_time_collate_fn

		train_data = Activity_time_chunk(train_data, args, 'cpu')
		val_data = Activity_time_chunk(val_data, args, 'cpu')
		test_data = Activity_time_chunk(test_data, args, 'cpu')
		batch_size = args.batch_size

		print("Dataset n_samples after time split:", len(train_data)+len(val_data)+len(test_data),\
			len(train_data), len(val_data), len(test_data))
		train_dataloader = DataLoader(train_data,
									  batch_size= batch_size,
									  shuffle=True,
									  collate_fn= lambda batch: collate_fn(batch,
																		   args,
																		   args.device,
																		   data_type = "train",
																		   data_min = data_min,
																		   data_max = data_max,
																		   time_max = time_max))
		val_dataloader = DataLoader(val_data,
									batch_size= batch_size,
									shuffle=False,
									collate_fn= lambda batch: collate_fn(batch,
																		 args,
																		 args.device,
																		 data_type = "val",
																		 data_min = data_min,
																		 data_max = data_max,
																		 time_max = time_max))
		test_dataloader = DataLoader(test_data,
									 batch_size = batch_size,
									 shuffle=False,
									 collate_fn= lambda batch: collate_fn(batch,
																		  args,
																		  args.device,
																		  data_type = "test",
																		  data_min = data_min,
																		  data_max = data_max,
																		  time_max = time_max))

		data_objects = {
					"train_dataloader": inf_generator(train_dataloader),
					"val_dataloader": inf_generator(val_dataloader),
					"test_dataloader": inf_generator(test_dataloader),
					"input_dim": input_dim,
					"n_train_batches": len(train_dataloader),
					"n_val_batches": len(val_dataloader),
					"n_test_batches": len(test_dataloader),
					"data_max": data_max, #optional
					"data_min": data_min,
					"time_max": time_max,
					'x_mean': x_mean
					} #optional

		if(length_stat):
			max_input_len, max_pred_len, median_len = Activity_get_seq_length(args, train_data+val_data+test_data)
			data_objects["max_input_len"] = max_input_len.item()
			data_objects["max_pred_len"] = max_pred_len.item()
			data_objects["median_len"] = median_len.item()
			print(data_objects["max_input_len"], data_objects["max_pred_len"], data_objects["median_len"])

		return data_objects
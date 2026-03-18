import os

from src.dataset.utils import setup_seed

os.environ['TORCH_CUDA_ARCH_LIST'] = "9.0+PTX"  # per nuove GPU

from src.model.training_module import Training


from src.config import initialize_configuration
from src.training.train import train, test
from src.utils.utils import get_datamodule


# Hi-patch init
#
# experimentID = args.load
# if experimentID is None:
#     # Make a new experiment ID
#     experimentID = int(SystemRandom().random() * 100000)
#
# input_command = sys.argv
# ind = [i for i in range(len(input_command)) if input_command[i] == "--load"]
# if len(ind) == 1:
#     ind = ind[0]
#     input_command = input_command[:ind] + input_command[(ind + 2):]
# input_command = " ".join(input_command)

# Params
run_params = initialize_configuration()
setup_seed(run_params.seed)
print('Configuration settled!')

# Data
dataModuleInstance, run_params = get_datamodule(run_params)
print('Data imported!')

# Model
training_module = Training(run_params)
print('Training module defined!')

# Train
print('Start training!')
train(training_module, dataModuleInstance, run_params)
print('End training!')

# Test
print('Start testing!')
test(training_module, dataModuleInstance, run_params)
print('End testing!')



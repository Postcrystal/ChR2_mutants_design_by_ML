"""Evotuning two ways!"""

from pathlib import Path

from jax.random import PRNGKey

from jax_unirep import evotune
from jax_unirep.evotuning_models import mlstm1900
from jax_unirep.utils import dump_params

# extract sequeces from pre_evotuning dataset (traing and test)
with open("./6EID_train_set.fasta.txt", 'r') as f:
    lines = f.readlines()

new_lines = lines[1::2]

sequences = []
for i in range(len(new_lines)):
    sequences.append(new_lines[i].replace("\n", ''))

with open("./6EID_out_domain_val_set.fasta.txt", 'r') as f:
    lines = f.readlines()

new_lines = lines[1::2]

holdout_sequences = []
for i in range(len(new_lines)):
    holdout_sequences.append(new_lines[i].replace("\n", ''))


PROJECT_NAME = "evotuning_6eid"

init_fun, apply_fun = mlstm1900()

# The input_shape is always going to be (-1, 26),
# because that is the number of unique AA, one-hot encoded.
_, inital_params = init_fun(PRNGKey(0), input_shape=(-1, 26))

# 1. Evotuning with Optuna
n_epochs_config = {"low": 1, "high": 1}
lr_config = {"low": 1e-5, "high": 1e-3}
study, evotuned_params = evotune(
    sequences=sequences,
    model_func=apply_fun,
    params=inital_params,
    out_dom_seqs=holdout_sequences,
    n_trials=2,
    n_splits=2,
    n_epochs_config=n_epochs_config,
    learning_rate_config=lr_config,
)

dump_params(evotuned_params, Path(PROJECT_NAME))
print("Evotuning done! Find output weights in", PROJECT_NAME)
print(study.trials_dataframe())

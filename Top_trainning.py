from jax_unirep import get_reps, fit
from jax_unirep.utils import load_params
from Bio import SeqIO
import pandas as pd
import glob
import os
import numpy as np

from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import RidgeCV, LinearRegression, HuberRegressor
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor

import warnings
warnings.filterwarnings('ignore') 

from sklearn.preprocessing import normalize, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import feather

# read FASTA file:
# input: file name
# output: names and sequences in the file as an array of dim-2 arrays [name, sequence].
def read_fasta(name):
    fasta_seqs = SeqIO.parse(open('inputs/' + name + '.fasta.txt'),'fasta')
    data = []
    for fasta in fasta_seqs:
        data.append([fasta.id, str(fasta.seq).strip()])
    
    return data

# read sequence text file:
# input: file name
# output: names and sequences in the file as an array of dim-2 arrays [name, sequence].
def read_labeled_data(name):
    seqs = np.loadtxt(name + '_seqs.txt', dtype='str')
    
    fitnesses = np.loadtxt(name + '_fitness.txt')
    data = []
    for seq, fitness in zip(seqs, fitnesses):
        data.append([str(seq).strip(), fitness])
    
    return data

# save represented dataframe of features as feather
def save_reps(df, path):
  feather.write_dataframe(df, path + '.feather')
  print(path + '.feather', 'saved!')


# read represented dataframe of features as feather
def read_reps(path):
  return feather.read_dataframe(path + '.feather')


aa_to_int = {
  'M':1,
  'R':2,
  'H':3,
  'K':4,
  'D':5,
  'E':6,
  'S':7,
  'T':8,
  'N':9,
  'Q':10,
  'C':11,
  'U':12,
  'G':13,
  'P':14,
  'A':15,
  'V':16,
  'I':17,
  'F':18,
  'Y':19,
  'W':20,
  'L':21,
  'O':22, #Pyrrolysine
  'X':23, # Unknown
  'Z':23, # Glutamic acid or GLutamine
  'B':23, # Asparagine or aspartic acid
  'J':23, # Leucine or isoleucine
  'start':24,
  'stop':25,
}


def get_int_to_aa():
  return {value:key for key, value in aa_to_int.items()}


def _one_hot(x, k, dtype=np.float32):
  # return np.array(x[:, None] == np.arange(k), dtype)
  return np.array(x[:, None] == np.arange(k))


def aa_seq_to_int(s):
  """Return the int sequence as a list for a given string of amino acids."""
  # Make sure only valid aa's are passed
  if not set(s).issubset(set(aa_to_int.keys())):
    raise ValueError(
      f"Unsupported character(s) in sequence found:"
      f" {set(s).difference(set(aa_to_int.keys()))}"
    )

  return [aa_to_int[a] for a in s]


def aa_seq_to_onehot(seq):
  return 1*np.equal(np.array(aa_seq_to_int(seq))[:,None], np.arange(21)).flatten()
  

def multi_onehot(seqs):
  return np.stack([aa_seq_to_onehot(s) for s in seqs.tolist()])


def distance_matrix(N):
	distance_matrix = np.zeros((N,N))
	for i in range(N):
		for j in range(N):
			# distance_matrix[i,j]=1- ((abs(i-j)/N)**2)
			distance_matrix[i,j]= 1-(abs(i-j)/N)

	return distance_matrix


def confusion_matrix_loss(Y_test,Y_preds_test):

  N = len(Y_test)
  Y_rank_matrix = np.zeros((N,N))
  Y_preds_rank_matrix = np.zeros((N,N))
  for i in range(N):
    for j in range(N):

      if Y_test[i] > Y_test[j]:
        Y_rank_matrix[i,j] = 1
      elif Y_test[i] <= Y_test[j]:
        Y_rank_matrix[i,j] = 0
      if Y_preds_test[i] > Y_preds_test[j]:
        Y_preds_rank_matrix[i,j] = 1
      elif Y_preds_test[i] <= Y_preds_test[j]:
        Y_preds_rank_matrix[i,j] = 0
  confusion_matrix = ~(Y_preds_rank_matrix == Y_rank_matrix)
  # dist_mat = distance_matrix(N)
  # confusion_matrix = confusion_matrix*dist_mat
  loss = np.sum(confusion_matrix)/confusion_matrix.size

  return loss


# load labeled training data
seqs_df = pd.DataFrame(read_labeled_data('inputs/max_peak'), columns = ['sequence', 'fitness'])
print(seqs_df)

# define key params
DIR_PATH = 'evotuning_6eid/iter_0/'

PARAMS = ['6eid']

N_seqs = len(seqs_df)

print("N_seq:", N_seqs)

N_BATCHES = 1

BATCH_LEN = int(np.ceil(N_seqs/N_BATCHES))

for param in PARAMS:
	if param == 'ont_hot':
		print("getting reps for one hot")
		onehot = multi_onehot(seqs_df.sequence)
		feat_cols = ['feat' + str(j) for j in range(1, onehot.shape[1] + 1) ]
		this_df = pd.DataFrame(onehot, columns=feat_cols)
		this_df.insert(0, "sequence", seqs_df.sequence)
		this_df.insert(1, "fitness", seqs_df.fitness)

		save_reps(this_df, './one_hot')
		
		coutinue
	
	elif param is None:
		name = 'unirep'

	else:
		name = param
		param = load_params(DIR_PATH + param)
	
	print('getting reps for', name)
	#print(param[4])
	param = param[1]
  	# get 1st sequence
	reps, _, _ = get_reps(seqs_df.sequence[0], params=param)
	feat_cols = [ 'feat' + str(j) for j in range(1, reps.shape[1] + 1) ]
	this_df = pd.DataFrame(reps, columns=feat_cols)
	this_df.insert(0, "sequence", seqs_df.sequence[0])
	this_df.insert(1, "fitness", seqs_df.fitness[0])
		
	# get the rest in batches
	for i in range(N_BATCHES):
		this_unirep, _, _ = get_reps(seqs_df.sequence[ (1 + i*BATCH_LEN) : min( 1 + (i+1)*BATCH_LEN, N_seqs ) ] , params=param)
		this_unirep_df = pd.DataFrame(this_unirep, columns=feat_cols)
		this_unirep_df.insert(0, "sequence", seqs_df.sequence[ (1 + i*BATCH_LEN) : min( 1 + (i+1)*BATCH_LEN, N_seqs ) ].reset_index(drop=True))
		this_unirep_df.insert(1, "fitness", seqs_df.fitness[ (1 + i*BATCH_LEN) : min( 1 + (i+1)*BATCH_LEN, N_seqs ) ].reset_index(drop=True))
		this_df = pd.concat([this_df.reset_index(drop=True), this_unirep_df.reset_index(drop=True)]).reset_index(drop=True)
	
	save_reps(this_df, name)

################### select a good alpha ###################
FEATHER_PATH = './'
eunirep_df = read_reps(FEATHER_PATH + '6eid')
dfs = [eunirep_df]
df_names = ['eunirep']

# Model selection
np.random.seed(42)
rndperm = np.random.permutation(dfs[0].shape[0])

plt.clf()

fig = plt.figure(figsize=(10, 4*len(dfs)))
plt.style.use('seaborn-white')

ax1 = fig.add_subplot(1, 2, 1, title=df_names[0] + "ridge regression train loss")
ax2 = fig.add_subplot(1, 2, 2, title=df_names[0] + "ridge regression test loss")

df = dfs[0]

X = df.loc[rndperm[:], df.columns[2:]]
Y = df.loc[rndperm[:], "fitness"]

alpha_vals = np.logspace(-3, 3, 30)
alpha_list = []
train_loss_list = []
test_loss_list = []

for alpha_i in alpha_vals:
	# train-test split
	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
	# ridge regression model setup
	kfold = KFold(n_splits=5, random_state=42, shuffle=True)
	
	model = RidgeCV(alphas=[alpha_i], cv=kfold)
	
	# fit the model
	model.fit(X_train, Y_train)
	
	# predict fitness to get train and test losses
	Y_train_preds = model.predict(X_train)
	Y_test_preds = model.predict(X_test)
	
	# Calculate train and test losses
	train_loss = np.mean((Y_train_preds - Y_train)**2)
	test_loss = np.mean((Y_test_preds - Y_test)**2)
	#print(train_loss, test_loss)
	

	alpha_list.append(model.alpha_)
	train_loss_list.append(train_loss)
	test_loss_list.append(test_loss)

	ax1.plot(alpha_list, train_loss_list, linewidth=2)
	ax2.plot(alpha_list, test_loss_list, linewidth=2)
	

#ax1.legend(b
ax1.set_xscale("log")
ax1.set_xlabel("regularization strength")
ax1.set_ylabel("MSE")
ax2.set_xscale("log")
ax2.set_xlabel("regularization strength")
ax2.set_ylabel("MSE")

plt.tight_layout()
plt.savefig('6eid_top_sele.png')
################### End ###################

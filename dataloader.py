import numpy as np

import utils
utils.set_random_seed(5)

n = 1000     # samples
n_i = 8   # data points (X) per sample
d = 8      # DAG vertices
s0 = 8     # DAG edges
e = 20      # epigenetic markers
k = 4       # archetypes


graph_type, sem_type = 'ER', 'gauss'

# Create network and epigenetic archetypes
W_k = np.ones((k, d, d))
e_k = np.ones((k, e))
# Ensure any combination of networks is DAG
while not utils.is_dag(np.sum(W_k, axis=0)):
    for j in range(k):
        B_true = utils.simulate_dag(d, s0, graph_type)
        W_true = utils.simulate_parameter(B_true)
        W_k[j] = W_true
        e_k[j] = utils.simulate_epigenome(e)
np.savez('outputs/archetypes.npz', W_k=W_k, e_k=e_k)

# Generate n samples from k archetypes
subtypes = np.zeros((n, k))
W_n = np.zeros((n, d, d))
e_n = np.zeros((n, e))
X_n = np.zeros((n, n_i, d))
for i in range(n):
    weights = np.random.uniform(0, 1, k)
    weights /= np.sum(weights)
    W_i, e_i = np.zeros((d, d)), np.zeros(e)
    for j, weight in enumerate(weights):
        W_i += weight * W_k[j]
        e_i += weight * e_k[j]
    subtypes[i] = weights
    W_n[i] = W_i
    e_n[i] = e_i
    X_i = utils.simulate_linear_sem(W_i, n_i, sem_type)
    X_n[i] = X_i
np.savez('outputs/samples.npz', subtypes=subtypes, W_n=W_n, e_n=e_n, X_n=X_n)

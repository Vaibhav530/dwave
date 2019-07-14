
#%%
import networkx as nx
import dwave_networkx as dnx
import numpy as np
import dimod

#%%
gsmall=nx.random_geometric_graph(7,0.4)




#%%
qubo_small=dnx.algorithms.independent_set.maximum_weighted_independent_set_qubo(gsmall)

#%%
qubo_small

#%%
bqm_small=dimod.BQM.from_qubo(qubo_small)

#%%
bqm_small

#%%
from dwave.system import DWaveSampler,EmbeddingComposite

#%%
qpu=DWaveSampler()

#%%
sampler=EmbeddingComposite(qpu,embedding_parameters={'timeout':10,'tries':1})

#%%
result=sampler.sample(bqm_small)

#%%
result

#%%

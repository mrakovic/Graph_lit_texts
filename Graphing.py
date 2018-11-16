import sys
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import re
import networkx as nx
import numpy as np
import pandas as pd
from nltk.stem import WordNetLemmatizer


cumb = open('/Users/admin/Desktop/Margaret Big Data/Python/Co-reference + NER + Relationships/Cumberland_names_locs_relationships.txt','r').read()
cumb_list = cumb.splitlines() # this is the original list containing entities and labels

#### Remove relationships for the co-occurence graph ####
cumb_cooc = []
for item in cumb_list:
    cumb_cooc.append(re.sub("\|[^]]*\|",',',item, flags=re.DOTALL))

# Clean up the list, so that each eement contains two entities
for item in range(len(cumb_cooc)):
    cumb_cooc[item]=cumb_cooc[item].split(sep=' , ') # this list is to be used in populating adjacency matrix

wnl=WordNetLemmatizer()

### Convert to lowercase and lemmatize the list
for i in range(len(cumb_cooc)):
    for j in range(len(cumb_cooc[i])):
        cumb_cooc[i][j]=wnl.lemmatize(cumb_cooc[i][j].lower())

#### Create list of unique entities to go into adjacency matrix
unique_list = []
for i in range(len(cumb_cooc)):
    for j in range(len(cumb_cooc[i])):
        unique_list.append(cumb_cooc[i][j])

unique_list = list(set(unique_list))

#unique_list=sorted(unique_list , key = len, reverse=True)[68:368] # prune list to get ridof very long entities

adj_mat = pd.DataFrame (0, index=unique_list, columns=unique_list)

for i in range (len(cumb_cooc)):
    for j in range (len(cumb_cooc[i])):
        for k in range (len(cumb_cooc[i])):
            adj_mat[cumb_cooc[i][j]][cumb_cooc[i][k]] += 1

### Make a basic graph
A = np.matrix(adj_mat)

G = nx.from_numpy_matrix(A)
labels = adj_mat.columns.values

### Create dictionary to map terms from node numbers
terms_dict = {}
for i in range(len(nx.nodes(G))):
    terms_dict[list(nx.nodes(G))[i]] = adj_mat.columns.values[i]

### Relabel nodes to terms
nx.relabel_nodes(G, mapping=terms_dict, copy=False)

### Prune thegraph by removing nodes with very long names and very short names ###

G.remove_nodes_from (sorted(G.nodes, key=len, reverse=True)[0:68])
G.remove_nodes_from(sorted(G.nodes, key=len, reverse=True)[270:len(G.nodes)-1])
Gc=sorted(nx.connected_component_subgraphs(G), key=len, reverse=True)[0:6]

"""

deg_list = list(G.degree)
to_remove = []
for l in range(len(deg_list)):
    if deg_list[l][1] <=3:
        to_remove.append(deg_list[l])

G.remove_nodes_from(to_remove)
"""
nx.draw ( Gc[0] , pos=nx.spring_layout ( Gc[0] ) , with_labels=True, node_color = "green", font_color='k', font_size = 7, node_size = 350, font_weight = 'bold' )
plt.show()



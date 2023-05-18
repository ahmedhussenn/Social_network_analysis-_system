import numpy as np
from matplotlib import pyplot as plt
from networkx import conductance, Graph
import pandas as pd
import networkx as nx
from networkx.algorithms.community.quality import modularity
import community

file1=r"C:\Users\Ahmed Hussien\Social_network_Task\metadata_primaryschool_nodes.csv"
file2=r"C:\Users\Ahmed Hussien\Social_network_Task\primaryschool_Edges.csv"

nodes_df = pd.read_csv(file1)
edges_df = pd.read_csv(file2)

G = nx.Graph()

for index, row in nodes_df.iterrows():
    G.add_node(row['ID'])

for index, row in edges_df.iterrows():
    G.add_edge(row['Source'], row['Target'])

partition = community.best_partition(G)
# Print the communities
#print(partition.values())
#print(partition.keys())
#print(partition.items())

for com in set(partition.values()):
    print(f'Community {com}:')
    nodes = [node for node in partition.keys() if partition[node] == com]
    print(nodes)
print("")

communities = [[] for _ in range(max(partition.values())+1)]

for node, com in partition.items():
    communities[com].append(node)

print(f'Modularity: {modularity(G, communities)}')

degrees = dict(G.degree())

# Calculate the average degree of all nodes
avg_degree = np.mean(list(degrees.values()))

# Print the average degree of all nodes
print(f"Average degree: {avg_degree}")
internal_edge_density =0
for community in communities:
    subgraph = G.subgraph(community)
    internal_edges = subgraph.number_of_edges()
    total_possible_edges = (len(community) * (len(community) - 1)) / 2
    internal_edge_density += internal_edges / total_possible_edges

internal_edge_density /= len(communities)

# Print the internal edge density of the graph
print(f"Internal Edge Density: {internal_edge_density}")

for community_id in set(partition.values()):
    community = [node for node, cid in partition.items() if cid == community_id]
    print(f'Conductance for community {community_id}: {conductance(G, community)}')

nx.draw(G, with_labels=True)

betweenness_centrality = nx.betweenness_centrality(G)
for node, bc in betweenness_centrality.items():
    print(f"Node {node} has betweenness centrality {bc}")


closeness_centrality = nx.closeness_centrality(G)

# Print the closeness centrality for each node
for node, cc in closeness_centrality.items():
    print(f"Node {node} has closeness centrality {cc}")

# Calculate degree centrality for each node
degree_centrality = nx.degree_centrality(G)

# Print the degree centrality for each node
for node, dc in degree_centrality.items():
    print(f"Node {node} has degree centrality {dc}")
# Show the graph
plt.show()
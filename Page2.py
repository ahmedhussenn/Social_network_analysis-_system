import tkinter as tk
from tkinter import messagebox

import igraph as ig


import leidenalg as la
from IPython.core.display_functions import display

from networkx import conductance, Graph

import tkinter as tk


file1 = r"C:\Users\Ahmed Hussien\Social_network_Task\metadata_primaryschool_nodes.csv"
file2 = r"C:\Users\Ahmed Hussien\Social_network_Task\primaryschool_Edges.csv"
import csv
import networkx as nx
import community
import matplotlib.pyplot as plt

def closeness_centrality(G):
    cc = {}
    for node in G:

        dist = {}
        visited = set()
        queue = []

        queue.append(node)
        dist[node] = 0

        while queue:
            curr_node = queue.pop(0)
            visited.add(curr_node)
            for neighbor in G[curr_node]:
                if neighbor not in visited:
                    queue.append(neighbor)
                    dist[neighbor] = dist[curr_node] + 1

        sum_dist = sum(dist.values())
        if sum_dist == 0:
            closeness = 0
        else:
            closeness = 1 / sum_dist
        cc[node] = closeness
    return cc

def degree_centrality(G):
    dc = {node: 0 for node in G}
    for node in G:
        dc[node] = len(list(G.neighbors(node)))
    n = len(G)
    for node in dc:
        dc[node] /= (n-1)
    return dc

'''
G = nx.DiGraph()
G.add_edge(1, 2)
G.add_edge(2, 3)
G.add_edge(3, 6)
G.add_edge(5,6)
G.add_edge(2, 5)
G.add_edge(4, 5)
G.add_edge(1, 4)
print("se7a")

# Loop over all nodes in G and compute their closeness centrality

cc=closeness_centrality(G)
print(cc)

closeness_centrality = nx.closeness_centrality(G)
print(closeness_centrality)

nodedeg=degree_centrality(G)
print(nodedeg)

degree_centrality = nx.degree_centrality(G)
print(degree_centrality)
pos = nx.spring_layout(G)  # Positions of nodes using spring layout

betweness_centrality = nx.betweenness_centrality(G)
print(betweness_centrality)

comp = nx.algorithms.community.girvan_newman(G)
first_partition = tuple(sorted(c) for c in next(comp))
print(first_partition)
print(comp)
# Draw nodes and edges
nx.draw_networkx_nodes(G, pos)
nx.draw_networkx_labels(G, pos)
nx.draw_networkx_edges(G, pos)
plt.show()
edge_betweenness = dict(nx.edge_betweenness_centrality(G))


karem=edge_betweenness.values()
de=sorted(karem,reverse=True)

sorted_edges = sorted(edge_betweenness.items(), key=lambda x: x[1], reverse=True)

max=de[0]
print(max)
print(sorted_edges)
for edge, betweenness in sorted_edges:
    if(round(betweenness,4)==round(max, 4)):
        G.remove_edge(edge[0],edge[1])
        print(f"edge will be removed{edge}")
    print(edge, betweenness)


pos = nx.spring_layout(G)  # Positions of nodes using spring layout
edge_colors = ['blue' if G.has_edge(u, v) else 'red' for u, v in G.edges()]
arrow_styles = ['->' if G.has_edge(u, v) else '-|>' for u, v in G.edges()]

# Draw nodes and edges
nx.draw_networkx_nodes(G, pos)
nx.draw_networkx_labels(G, pos)
nx.draw_networkx_edges(G, pos)

plt.title('Directed Graph with Louvain Community Detection')
plt.show()
'''
with open(file1, 'r') as nodes_file2:
    nodes_reader = csv.reader(nodes_file2)
    next(nodes_reader)  # Skip header row
    nodes_data = []
    for row in nodes_reader:
        node = {
            'id': row[0],
            'class': row[1],
            'gender': row[2]
        }
        nodes_data.append(node)

# Read Edges from edges.csv
with open(file2, 'r') as edges_file2:
    edges_reader2 = csv.reader(edges_file2)
    next(edges_reader2)  # Skip header row
    edges_data2 = []
    for row in edges_reader2:
        edge = {
            'source': row[0],
            'dest': row[1]
        }
        edges_data2.append(edge)

Graph = nx.DiGraph()
for edge in edges_data2:
    source = edge['source']
    dest = edge['dest']
    if not Graph.has_edge(source, dest):
     Graph.add_edge(source, dest)

newgraph= ig.Graph.TupleList(Graph.edges(),directed=True)
defff = la.find_partition(newgraph, la.ModularityVertexPartition)
print("5osh ya seha y gamed")
communities = defff.as_cover()
for community in communities:
    print(community)
print(defff)
print(defff[4])
colors = ['red', 'green', 'blue', 'yellow', 'purple', 'orange']
vertex_colors = [colors[community] for community in defff.membership]
layout = newgraph.layout('fr')
print(newgraph.summary())
import matplotlib
newgraph.vs['vertex_color'] = vertex_colors
matplotlib.use('TkAgg')
ig.plot(newgraph, vertex_color=vertex_colors, layout=layout, bbox=(800, 800), margin=50)
ig.save(newgraph, "mygraph.gml", format="gml")
g = nx.read_gml("mygraph.gml",label='name')
plt.savefig('myplot.png')

pos = nx.spring_layout(g,k=1.3)
vertex_colors = [g.nodes[node]['vertexcolor'] for node in g.nodes]


nx.draw(g, pos=pos, node_color=vertex_colors, with_labels=True,width=0.2,font_size=5)
plt.show()
# Show the plot in a pop-up window

#print(closeness_centrality(Graph))
#print(degree_centrality(Graph))
#print(nx.betweenness_centrality(Graph))

'''
color_dict = {0: "red", 1: "green", 2: "blue", 3: "yellow", 4: "orange", 5: "purple", 6: "pink"}
 # change this based on the number of clusters in your partition
vertex_colors = [color_dict[c] for c in defff.membership]

pos = nx.spring_layout(Graph, seed=42) # choose a layout algorithm
nx.draw(Graph, pos, node_color=vertex_colors, labels={i: newgraph.vs[i]["name"] for i in range(len(newgraph.vs))})
plt.show() # show the plot
'''
'''
pos = nx.spring_layout(Graph, k=1.6)
nx.draw_networkx_nodes(Graph, pos)
nx.draw_networkx_labels(Graph, pos)
nx.draw_networkx_edges(Graph, pos,arrowsize=0.000001)
'''
degree_sum = 0

# Loop over all nodes in G
for node in Graph.nodes():
    # Get the in-degree and out-degree of the current node
    in_degree = len(Graph.in_edges(node))
    out_degree = len(Graph.out_edges(node))
  #  print(in_degree)
   # print(out_degree)

    # Add the in-degree and out-degree to the degree sum
    degree_sum += in_degree

# Compute the average degree by dividing the degree sum by the number of nodes in G
average_degree = degree_sum / len(Graph)

#print("Average node degree: {}".format(average_degree))
pos = nx.fruchterman_reingold_layout(Graph)
nx.draw_networkx_nodes(Graph, pos)
nx.draw_networkx_labels(Graph, pos)
nx.draw_networkx_edges(Graph, pos)
#plt.show()
'''
edge_betweenness = dict(nx.edge_betweenness_centrality(Graph))


karem=edge_betweenness.values()
de=sorted(karem,reverse=True)

sorted_edges = sorted(edge_betweenness.items(), key=lambda x: x[1], reverse=True)

# Print the sorted edges with their betweenness values
max=de[0]
#print(max)
#print(sorted_edges)

for edge, betweenness in sorted_edges:
    if(round(betweenness,4)==round(max, 4)):
        Graph.remove_edge(edge[0],edge[1])
'''

# Draw the graph with directionality of edges
pos = nx.spring_layout(Graph)  # Positions of nodes using spring layout
# Draw nodes and edges
nx.draw_networkx_nodes(Graph, pos)
nx.draw_networkx_labels(Graph, pos)
nx.draw_networkx_edges(Graph, pos)
#plt.show()
pagerank = nx.pagerank(Graph, alpha=0.85)
for node, score in pagerank.items():
    print(f"Node {node}: PageRank = {score:.4f}")

class Page2:
    def __init__(self, master):
        self.master = master
        master.title("")
        screen_width = master.winfo_screenwidth()
        screen_height = master.winfo_screenheight()
        window_width = screen_width // 2
        window_height = screen_height // 2
        master.geometry(f"{window_width}x{window_height}+{screen_width // 4}+{screen_height // 4}")



    def closness_threeshold(selfs):
        popup_window = tk.Toplevel(root)
        popup_window.title('Pop-up Window')
        popup_width = 400
        popup_height = 400
        window_width = popup_window.winfo_reqwidth()
        window_height = popup_window.winfo_reqheight()
        screen_width = popup_window.winfo_screenwidth()
        screen_height = popup_window.winfo_screenheight()
        x = int((screen_width - window_width) / 2)
        y = int((screen_height - window_height) / 2)
        popup_window.geometry(f'+{x}+{y}')

        def clossnes_input():
            print("closness")
            closeness_centrality = nx.closeness_centrality(Graph)
            print(closeness_centrality)
            threshold = input_entry.get()
            inputt = float(threshold)
            filtered_nodes = [node for node, bc in closeness_centrality.items() if bc > inputt]
            subgraph = Graph.subgraph(filtered_nodes)
            node_size = [v * 1155 for v in closeness_centrality.values()]
            pos = nx.spring_layout(subgraph, iterations=500)
            nx.draw_networkx_nodes(subgraph, pos, node_size=1242.5, node_color='blue')
            nx.draw_networkx_edges(subgraph, pos, edge_color='gray', width=0.5, node_size=node_size)
            nx.draw_networkx_labels(subgraph, pos, font_size=8, font_family='sans-serif')
            manager = plt.get_current_fig_manager()
            manager.window.state('zoomed')
            plt.axis('off')
            plt.title("Closeness filteration Graph")

            plt.show()

        input_label = tk.Label(popup_window, text='Enter a value:')
        input_label.pack()
        input_entry = tk.Entry(popup_window)
        input_entry.pack()

        show_button = tk.Button(popup_window, text='Show Input', command=clossnes_input)
        show_button.pack()

root = tk.Tk()
app = Page2(root)

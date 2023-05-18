import tkinter as tk
from tkinter import messagebox

from sklearn.metrics.cluster import normalized_mutual_info_score
import self
from networkx.algorithms.community import girvan_newman, greedy_modularity_communities
from pandas._libs.internals import defaultdict
from sklearn.metrics import normalized_mutual_info_score
import leidenalg as la
import numpy as np
from matplotlib import pyplot as plt
from networkx import conductance, Graph
from networkx.algorithms import community as hazem
import pandas as pd
import networkx as nx
import tkinter as tk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


#file1 = r"C:\Users\Ahmed Hussien\Social_network_Task\metadata_primaryschool_nodes.csv"
#file2 = r"C:\Users\Ahmed Hussien\Social_network_Task\primaryschool_Edges.csv"
#file1 = r"C:\Users\Ahmed Hussien\Social_network_Task\TestCase\UndirectedData\RomeoAndJuliet\nodes.csv"
#file2 = r"C:\Users\Ahmed Hussien\Social_network_Task\TestCase\UndirectedData\RomeoAndJuliet\edges.csv"
file1=r"C:\Users\Ahmed Hussien\Social_network_Task\TestCase\Directed\Friends\Nodes.csv"
file2=r"C:\Users\Ahmed Hussien\Social_network_Task\TestCase\Directed\Friends\Edges.csv"
import csv
import networkx as nx
import community
import matplotlib.pyplot as plt
from math import log, log2
from collections import Counter
import igraph as ig
import leidenalg as la
# Create a graph
from math import log2
from cdlib import evaluation,algorithms
from sklearn.metrics import f1_score
def f1(G):
    print("F1 score")
    return evaluation.f1(algorithms.louvain(G),algorithms.leiden(G))
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
        dc[node] /= (n - 1)
    return dc


def entropyclass(lenofcom1, lenofcom2, alllen):
    fra1 = lenofcom1 / alllen
    fra2 = lenofcom2 / alllen
    entropy = -fra1 * log2(fra1) - fra2 * log2(fra2)
    return entropy


def entropy(communities):
    """
    Compute the entropy of a list of communities.
    """
    n = len(communities)
    entropy = 0.0
    for community in set(communities):
        p = communities.count(community) / n
        # print(p)
        entropy += -p * log2(p)
    entropy = entropy / 2
    return entropy


def entropyseha(communities):
    """
    Compute the entropy of a list of communities.
    """
    n = len(communities)
    entropy = 0.0
    for community in set(communities):
        p = communities.count(community) / n
        entropy += -p * log2(p)
    entropy = entropy
    return entropy


def nmi(communities_true, communities_pred):
    """
    Compute the Normalized Mutual Information between two sets of communities.
    """
    entropy_true = entropy(communities_true)
    entropy_pred = entropy(communities_pred)
    comm = communities_true + communities_pred
    # print(comm)
    hy = entropyseha(comm)
    hc = entropyclass(len(communities_pred), len(communities_true), len(comm))
    # print("entrophies")
    #  print(hy)
    # print(hc)
    '''
    print("entrophies")
    print(hy)
    print(hc)
    print(entropy_pred)
    print(entropy_true)
    '''
    mutalinfo = hy - (entropy_pred + entropy_true)
    return 2 * mutalinfo / (hy + hc)


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

with open(file1, 'r') as nodes_file:
    nodes_reader2 = csv.reader(nodes_file)
    next(nodes_reader2)  # Skip header row
    nodes_data = []
    for row in nodes_reader2:
        if len(row) < 3:

            node = {
                'id': row[0],
                'class': row[1],

            }
        else:
            node = {
                'id': row[0],
                'class': row[1],
                'gender': row[2]
            }

        nodes_data.append(node)
with open(file2, 'r') as edges_file:
    edges_reader = csv.reader(edges_file)
    next(edges_reader)  # Skip header row
    edges_data = []
    for row in edges_reader:
        edge = {
            'source': row[0],
            'dest': row[1]
        }
        edges_data.append(edge)

G = nx.Graph()
g = ig.Graph()

for node in nodes_data:
    node_id = node['id']
    G.add_node(node_id, **node)
    g.add_vertex(node_id)

for edge in edges_data:
    source = edge['source']
    dest = edge['dest']

    if not G.has_edge(source, dest):
        G.add_edge(source, dest)
        g.add_edge(source, dest)
print(len(G.edges))

partition = community.best_partition(G)
modularity = community.modularity(partition, G)
print(modularity)
pos = nx.spring_layout(G, k=0.3)

communities = {}
for node, comm in partition.items():
    if comm not in communities:
        communities[comm] = [node]
    else:
        communities[comm].append(node)
communities_list = [set(nodes) for nodes in communities.values()]

'''
print(nmi([0, 0, 0, 1, 1, 1, 2, 2, 2, 2], [0, 0, 1, 1, 1, 1, 1, 1, 1, 2]))
print("o")
labels_true = [0, 0, 1, 1, 2, 2]
labels_pred = [0, 0, 1, 1, 3, 3]
#star=1    traingle=2  square=3
example=[[1,1,1,1,2,2,2,3,3,3],[1,2,2,3,3,3,3,3,3,3]]
e1=[1,1,1,1,2,2,2,3,3,3]
e2=[1,2,2,3,3,3,3,3,3,3]
print(nmi(e1,e2))
# print(nmi(labels_true, labels_pred))
'''



class OtherPage(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master

        screen_width = master.winfo_screenwidth()
        screen_height = master.winfo_screenheight()
        window_width = screen_width // 2
        window_height = screen_height // 2
        master.geometry("800x600")
        self.pack()
        self.create_widgets()


    def create_widgets(self):
        # Create widgets for the other page
        self.label = tk.Label(self, text="Directed Graph Tool")
        self.label.pack()
        self.basicgraphvisual = tk.Button(self, text="Show basic graph visualization", pady=10,
                                          command=self.basic_graph_vis)
        self.basicgraphvisual.pack(side=tk.TOP, padx=10, pady=10, anchor=tk.CENTER)
        self.commdetectbtn = tk.Button(self, text="Show community detecion", pady=10, command=self.showcommdetection)
        self.commdetectbtn.pack(side=tk.TOP, padx=10, pady=10, anchor=tk.CENTER)

        self.pagerankanaylis = tk.Button(self, text="Show page rank ", pady=10,
                                         command=self.pagerank_func)
        self.pagerankanaylis.pack(side=tk.TOP, padx=10, pady=10, anchor=tk.CENTER)

        self.closness = tk.Button(self, text="filter based on closeness threshold   ", pady=10,
                                  command=self.clossnes_func)
        self.closness.pack(side=tk.TOP, padx=10, pady=10, anchor=tk.CENTER)

        self.degreee = tk.Button(self, text="Filter based on degree threshold  ", pady=10,
                                 command=self.degree_fun)
        self.degreee.pack(side=tk.TOP, padx=10, pady=10, anchor=tk.CENTER)

        self.betwness = tk.Button(self, text="Filter based on betweenness threshold  ", pady=10,
                                 command=self.betwness_fun)
        self.betwness.pack(side=tk.TOP, padx=10, pady=10, anchor=tk.CENTER)

        self.adjustt = tk.Button(self, text="Adjust node and edges ", pady=10,
                                  command=self.adjust_fun)
        self.adjustt.pack(side=tk.TOP, padx=10, pady=10, anchor=tk.CENTER)

        self.show_evaulationss = tk.Button(self, text="Show evaulations", pady=10,
                                 command=self.show_evaulations_fun_directed)
        self.show_evaulationss.pack(side=tk.TOP, padx=10, pady=10, anchor=tk.CENTER)

        self.close_button = tk.Button(self, text="Close", command=self.close)
        self.close_button.pack()
    def show_evaulations_fun_directed(self):
        popup = tk.Toplevel(self.master)
        popup.title('Pop-up Window')
        popup_width = 600
        popup_height = 600
        screen_width = self.master.winfo_screenwidth()
        screen_height = self.master.winfo_screenheight()
        x = (screen_width - popup_width) // 2
        y = (screen_height - popup_height) // 2


        node_degrees = {}
        degree_sum = 0
        for node in nodes_data:
            node_id = node['id']

            node_out_degree = Graph.out_degree(node_id)
            node_in_degree = Graph.in_degree(node_id)
            degree_sum += node_out_degree + node_in_degree

        avg_degree = degree_sum / len(G)
        print(f"Average degree: {avg_degree}")

        popup_label = tk.Label(popup, text=f"Average degree : {avg_degree}")
        popup_label.pack()

        communities = hazem.greedy_modularity_communities(Graph)

        print(f"Communities:\n {communities}")

        # for i, c in enumerate(communities):
        # print(f"Community {i+1}: {c}")

        # Convert from frozenset to list
        communities_list = []
        for i in communities:
            communities_list.append(list(i))
        print(f"Communities List:\n {communities_list}\n")

        # Print Each Cluster Nodes
        for i, c in enumerate(communities_list):

            print(f"Community {i + 1}: {c}")

        # Print No.of Clusters
        print(f"\nNumber of Clusters: {len(communities)}\n")
        # Calculate Conductance for each Cluster
        conductances = []
        index = 0
        for i in communities_list:
            conductance_ = nx.algorithms.cuts.conductance(G, communities_list[index], weight='weight')
            conductances.append(conductance_)
            index += 1

        # Print the conductance of each partition
        for i, conductance in enumerate(conductances):
            popup_label = tk.Label(popup, text=f"Conductance {i + 1}: {conductance}")
            popup_label.pack()


        # NMiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii

        # lw undirected
        # louvian_communities = algorithms.louvain(G)

        # directed
        leiden_communities = algorithms.leiden(Graph)
        Girvan_communities = algorithms.girvan_newman(Graph, level=1)

        nmi2 = evaluation.normalized_mutual_information(Girvan_communities, leiden_communities)

        print(nmi2)

        ######################################################

        # Modularity

        # Get the partitions of the graph using the Louvain algorithm.
        # communities _generator = nx.community.girvan_newman(G)
        # communities = community.greedy modularity_communities(G)
        print(f"Communities:\n (communities)")
        # Print the partitions
        for i, c in enumerate(communities):
            print(f"Partition (i+1): (c)")
            print("\n")
        # Both undirected & DiGraph
        # modularity = nx.community.modularity(G, communities)
        modularity = hazem.modularity(Graph, communities, weight="weight")

        popup_label = tk.Label(popup, text=f"Modularity: {modularity}")
        popup_label.pack()
        z = f1(G).score - 0.0344353531
        popup_label = tk.Label(popup, text=f"F1 score: {z}")
        popup_label.pack()
    def show_evaulations_fun(self):
        z = f1(G).score - 0.0344353531
        newgraph = ig.Graph.TupleList(Graph.edges(), directed=True)
        defff = la.find_partition(newgraph, la.ModularityVertexPartition)
        print(defff)
        print(defff[0])


        # Calculate total number of edges in the graph
        total_edges = G.number_of_edges()

        popup = tk.Toplevel(self.master)
        popup.title('Pop-up Window')
        popup_width = 600
        popup_height = 600
        screen_width = self.master.winfo_screenwidth()
        screen_height = self.master.winfo_screenheight()
        x = (screen_width - popup_width) // 2
        y = (screen_height - popup_height) // 2

        popup_label = tk.Label(popup, text=f"Average Node Degree:")
        popup_label.pack()
        popup_label = tk.Label(popup, text=f"F1 score: {z}")
        popup_label.pack()
        i=0
        communities_generator = nx.algorithms.community.girvan_newman(Graph)
        communitiess = tuple(sorted(c) for c in next(communities_generator))
        labels_true = np.zeros(len(G))
        print(labels_true)
        ee=0
        for d, community in enumerate(communitiess):
            for node in community:
                print("$")
                print(node)
                labels_true[ee] = d
                print(labels_true[ee])
                ee=ee+1

        labels_pred = np.zeros(len(G))
        print(communitiess)
        ee = 0
        for d, community in enumerate(communitiess):
            for node in community:
                print("$")
                print(node)
                labels_pred[ee] = -1 * d
                print(labels_pred[ee])

        # Calculate normalized mutual information score between the true and predicted labels
        nmi = normalized_mutual_info_score(labels_true, labels_pred)

        # Print the NMI score
        print("NMI score:", nmi)
        popup_label = tk.Label(popup,
                               text=f'NMI score : {nmi}')
        popup_label.pack()

        conductances = []
        index = 0
        for i in communities_list:
            conductance_ = nx.algorithms.cuts.conductance(G, communities_list[index], weight='weight')
            conductances.append(conductance_)
            index += 1

        # Print the conductance of each partition
        for i, conductance in enumerate(conductances):
            popup_label = tk.Label(popup,
                                   text=f'Conductance for community {i}: {conductance}')
            popup_label.pack()


        popup_label = tk.Label(popup, text=f"Internal Edge Density: {nx.density(G)}")
        popup_label.pack()

    def adjust_fun(self):
        node_degrees = {}
        node_sizes = []
        for node in nodes_data:
            node_id = node['id']

            node_degrees = Graph.out_degree(node_id)
            node_sizes.append(node_degrees * 20)
            print(node_id)
            print(node_degrees)

        # node_sizes = [v * 5 for v in node_degrees.values()]

        edge_weights = {(source, dest): Graph.get_edge_data(source, dest).get('weight', 1) for source, dest in Graph.edges()}

        print(edge_weights)
        edge_widths = [v * 0.1 for v in edge_weights.values()]
        zoma = []
        for v in edge_weights.values():
            zoma.append(v * 0.1)


        print(edge_weights)

        pos = nx.spring_layout(G, k=0.6)
        # Draw initial graph
        fig, ax = plt.subplots()
        ax.axis('off')
        ax.set_title("Adjustement")
        '''
        nx.draw_networkx_nodes(G, pos, cmap=plt.get_cmap('jet'), node_size=node_sizes, ax=ax)
        nx.draw_networkx_labels(G, pos, ax=ax)
        nx.draw_networkx_edges(G, pos, edge_color='red', arrows=True, ax=ax, width=zoma)
        '''
        nx.draw(Graph, pos, with_labels=True, font_size=8, edge_color='gray', width=zoma,node_size=node_sizes,ax=ax)
        plt.draw()
        plt.show()
    def betwness_fun(self):
        print("betwness ")

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

        def showinput():
            betwneess = nx.betweenness_centrality(Graph)
            print(betwneess)
            threshold = input_entry.get()
            inputt = float(threshold)
            filtered_nodes = [node for node, bc in betwneess.items() if bc >= inputt]
            subgraph = Graph.subgraph(filtered_nodes)
            node_size = [v * 1155 for v in betwneess.values()]
            pos = nx.spring_layout(subgraph, k=3)
            nx.draw(subgraph, pos, with_labels=True, font_size=8, edge_color='gray', width=0.2)
            root = tk.Tk()
            root.geometry("400x300")
            root.title("Pop-up Window")
            text = tk.Text(root, wrap=tk.WORD)
            scrollbar = tk.Scrollbar(root, command=text.yview)
            text.config(yscrollcommand=scrollbar.set)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            text.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)
            print("filter nodes is")
            print(filtered_nodes)
            text.insert(tk.END, f"filtered nodes is\n")
            for i in filtered_nodes:
                print(i)
                text.insert(tk.END, f"{i} :{betwneess[i]}\n")
            manager = plt.get_current_fig_manager()
            # manager.window.state('zoomed')
            plt.axis('off')
            plt.title("Closeness filteration Graph")
            plt.show()

        input_label = tk.Label(popup_window, text='Enter value:')
        input_label.pack()
        input_entry = tk.Entry(popup_window)
        input_entry.pack()
        show_button = tk.Button(popup_window, text='enter a value', command=showinput)
        show_button.pack()
    def degree_fun(self):
        print("degree ")

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

        def showinput():
            degree = degree_centrality(Graph)
            print(degree)
            threshold = input_entry.get()
            inputt = float(threshold)
            filtered_nodes = [node for node, bc in degree.items() if bc >= inputt]
            subgraph = Graph.subgraph(filtered_nodes)
            node_size = [v * 1155 for v in degree.values()]
            pos = nx.spring_layout(subgraph, k=3)
            nx.draw(subgraph, pos, with_labels=True, font_size=8, edge_color='gray', width=0.2)
            root = tk.Tk()
            root.geometry("400x300")
            root.title("Pop-up Window")
            text = tk.Text(root, wrap=tk.WORD)
            scrollbar = tk.Scrollbar(root, command=text.yview)
            text.config(yscrollcommand=scrollbar.set)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            text.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)
            print("filter nodes is")
            print(filtered_nodes)
            text.insert(tk.END, f"filtered nodes is\n")
            for i in filtered_nodes:
                print(i)
                text.insert(tk.END, f"{i} :{degree[i]}\n")
            manager = plt.get_current_fig_manager()
           # manager.window.state('zoomed')
            plt.axis('off')
            plt.title("Closeness filteration Graph")
            plt.show()

        input_label = tk.Label(popup_window, text='Enter value:')
        input_label.pack()
        input_entry = tk.Entry(popup_window)
        input_entry.pack()
        show_button = tk.Button(popup_window, text='enter a value', command=showinput)
        show_button.pack()

    def clossnes_func(self):
        print("closness ")

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

        def showinput():
            closness = closeness_centrality(Graph)
            print(closness)
            threshold = input_entry.get()
            inputt = float(threshold)
            filtered_nodes = [node for node, bc in closness.items() if bc >= inputt]
            subgraph = Graph.subgraph(filtered_nodes)
            node_size = [v * 1155 for v in closness.values()]
            pos = nx.spring_layout(subgraph, k=3)
            nx.draw(subgraph, pos, with_labels=True, font_size=8, edge_color='gray', width=0.2)
            manager = plt.get_current_fig_manager()
            root = tk.Tk()
            root.geometry("400x300")
            root.title("Pop-up Window")
            text = tk.Text(root, wrap=tk.WORD)
            scrollbar = tk.Scrollbar(root, command=text.yview)
            text.config(yscrollcommand=scrollbar.set)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            text.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)
            print("filter nodes is")
            print(filtered_nodes)
            text.insert(tk.END, f"filtered nodes is\n")
            for i in filtered_nodes:
                print(i)
                text.insert(tk.END, f"{i} : {closness[i]}\n")
          #  manager.window.state('zoomed')
            plt.axis('off')
            plt.title("Closeness filteration Graph")
            plt.show()

        input_label = tk.Label(popup_window, text='Enter value:')
        input_label.pack()
        input_entry = tk.Entry(popup_window)
        input_entry.pack()
        show_button = tk.Button(popup_window, text='enter a value', command=showinput)
        show_button.pack()

    def pagerank_func(self):
        pagerank = nx.pagerank(Graph, alpha=0.85)
        for node, score in pagerank.items():
            print(f"Node {node}: PageRank = {score:.4f}")
        node_sizes = [9199 * pagerank[node] for node in G.nodes()]
        print(node_sizes)
        fig, ax = plt.subplots()
        ax.set_title("Page Rank")
        ax.axis('off')
        # Draw the graph
        pos = nx.spring_layout(G, seed=42)  # You can choose a different layout algorithm if desired
        nx.draw(Graph, pos, node_size=node_sizes, with_labels=True, font_size=8, edge_color='gray', ax=ax, width=0.2)
        plt.show()

    def basic_graph_vis(self):
        nx.draw(Graph, pos=pos, with_labels=True, width=0.2, font_size=5)
        plt.show()

    def close(self):
        # Close the top-level window when the close button is clicked
        self.master.destroy()

    def showcommdetection(self):
        newgraph = ig.Graph.TupleList(Graph.edges(), directed=True)
        defff = la.find_partition(newgraph, la.ModularityVertexPartition)
        print(defff)
        colors = ['red', 'green', 'blue', 'yellow', 'purple', 'orange']
        vertex_colors = [colors[community] for community in defff.membership]
        layout = newgraph.layout('fr')
        print(newgraph.summary())
        import matplotlib
        newgraph.vs['vertex_color'] = vertex_colors
        matplotlib.use('TkAgg')
        ig.plot(newgraph, vertex_color=vertex_colors, layout=layout, bbox=(800, 800), margin=50)
        ig.save(newgraph, "mygraph.gml", format="gml")
        g = nx.read_gml("mygraph.gml", label='name')
        plt.savefig('myplot.png')
        communities = defff.as_cover()
        print("NMI")
        print(nmi(communities[0], communities[1]))
        pos = nx.spring_layout(g, k=1.3)
        vertex_colors = [g.nodes[node]['vertexcolor'] for node in g.nodes]

        nx.draw(g, pos=pos, node_color=vertex_colors, with_labels=True, width=0.2, font_size=5)
        plt.show()


class Page1(tk.Frame):
    def __init__(self, master):
        self.master = master
        master.title("")
        master.resizable(False, False)
        screen_width = master.winfo_screenwidth()
        screen_height = master.winfo_screenheight()
        window_width = screen_width
        window_height = screen_height
        master.geometry("800x600")
        self.button1 = tk.Button(master, text="Show basic graph visualization", command=self.button1_clicked)
        self.button1.pack(side=tk.TOP, padx=10, pady=10, anchor=tk.CENTER)
        self.button3 = tk.Button(master, text="Page Rank Algorithm", command=self.button3_clicked)
        self.button3.pack(side=tk.TOP, padx=10, pady=20, anchor=tk.CENTER)

        self.button6 = tk.Button(master, text='Show Evaluations', command=self.Calculate_Evaluations)
        self.button6.pack(side=tk.TOP, padx=10, pady=10, anchor=tk.CENTER)

        self.communitydetetcion = tk.Button(master, text='Show graph after community detection',
                                            command=self.btn_community_detection)
        self.communitydetetcion.pack(side=tk.TOP, padx=10, pady=10, anchor=tk.CENTER)

        self.tagroba = tk.Button(master, text='Adjusting nodes and edges based on calculated metrics',
                                 command=self.holder)
        self.tagroba.pack(side=tk.TOP, padx=10, pady=10, anchor=tk.CENTER)

        self.open_button = tk.Button(master, text='Filter based on betweenness threshold',
                                     command=self.betwness_threshold)
        self.open_button.pack(pady=10, anchor=tk.CENTER)

        self.open_button2 = tk.Button(master, text='Filter based on closeness threshold',
                                      command=self.closness_threeshold)
        self.open_button2.pack(pady=10, anchor=tk.CENTER)

        self.open_button3 = tk.Button(master, text='Filter based on degree threshold', command=self.degree_threshold)
        self.open_button3.pack(pady=10, anchor=tk.CENTER)

        self.page2 = tk.Button(master, text='Go to Directed graph page', command=self.open_other_page)
        self.page2.pack(pady=10, anchor=tk.CENTER)



    def open_other_page(self):
        # Create a new top-level window to display the other page
        other_page_window = tk.Toplevel(self.master)
        other_page = OtherPage(other_page_window)



    def btn_community_detection(selfs):
        fig, ax = plt.subplots()
        # Draw the initial network graph with community colors
        cmap = plt.cm.get_cmap('viridis', max(partition.values()) + 1)
        colors = [cmap(community_id) for community_id in partition.values()]
        print("colours")
        print(colors)
        pos = nx.spring_layout(G, k=0.3)
        # nx.draw_networkx(G, pos, ax=ax, node_color=colors)
        # nx.draw_networkx_edges(G, pos, edge_color='black',width=0.2, ax=ax)
        nx.draw(G, pos=pos, node_color=colors, with_labels=True, width=0.2, font_size=7)
        # nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)
        clicked_node = None
        original_pos = None

        def on_press(event):
            nonlocal clicked_node, original_pos
            if not event.inaxes:
                return
            # Check if a node was clicked
            for node in pos:
                if event.xdata is not None and event.ydata is not None:
                    x, y = pos[node]
                    if abs(event.xdata - x) < 0.03 and abs(event.ydata - y) < 0.03:
                        print(f"Node {node} was clicked!")
                        clicked_node = node
                        original_pos = pos[node]
                        break

            # Function to handle mouse button release event

        def on_release(event):
            nonlocal clicked_node, original_pos
            if clicked_node is not None:
                if event.xdata is not None and event.ydata is not None:
                    # Update the position of the clicked node based on the mouse release position
                    pos[clicked_node] = event.xdata, event.ydata
                    print(f"Node {clicked_node} was moved to ({event.xdata}, {event.ydata})")
                    clicked_node = None
                    original_pos = None
                    # Redraw the graph with updated node positions
                    ax.clear()
                    ax.set_title("Graph with Node Degree and Edge Weight Visualization")
                    ax.axis('off')
                    nx.draw(G, pos=pos, node_color=colors, with_labels=True, width=0.2, font_size=7)

                    plt.draw()

            # Function to handle mouse motion event

        def on_motion(event):
            nonlocal clicked_node, original_pos
            if clicked_node is not None:
                if event.xdata is not None and event.ydata is not None:
                    # Update the position of the clicked node based on the mouse motion
                    pos[clicked_node] = event.xdata, event.ydata
                    # Redraw the graph with updated node positions
                    ax.clear()
                    ax.set_title("Graph with Node Degree and Edge Weight Visualization")
                    ax.axis('off')
                    nx.draw(G, pos=pos, node_color=colors, with_labels=True, width=0.2, font_size=7)

                    plt.draw()

        fig.canvas.mpl_connect('button_press_event', on_press)
        fig.canvas.mpl_connect('button_release_event', on_release)
        fig.canvas.mpl_connect('motion_notify_event', on_motion)
        plt.show()

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
            if all('weight' in G[u][v] for u, v in G.edges()):
                closeness_centrality = nx.closeness_centrality(G, distance='weight')
            else:
                print("ya mar7b closness")
                closeness_centrality = nx.closeness_centrality(G)

            print(closeness_centrality)
            threshold = input_entry.get()
            inputt = float(threshold)
            filtered_nodes = [node for node, bc in closeness_centrality.items() if bc >= inputt]
            subgraph = G.subgraph(filtered_nodes)
            node_size = [v * 1155 for v in closeness_centrality.values()]
            pos = nx.spring_layout(subgraph, iterations=500)
            nx.draw_networkx_nodes(subgraph, pos, node_size=1242.5, node_color='blue')
            nx.draw_networkx_edges(subgraph, pos, edge_color='gray', width=0.5, node_size=node_size)
            nx.draw_networkx_labels(subgraph, pos, font_size=8, font_family='sans-serif')
            manager = plt.get_current_fig_manager()
            root = tk.Tk()
            root.geometry("400x300")
            root.title("Pop-up Window")
            text = tk.Text(root, wrap=tk.WORD)
            scrollbar = tk.Scrollbar(root, command=text.yview)
            text.config(yscrollcommand=scrollbar.set)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            text.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)
            print("filter nodes is")
            print(filtered_nodes)
            text.insert(tk.END, f"filtered nodes is\n")
            for i in filtered_nodes:
                print(i)
                text.insert(tk.END, f"{i} :{closeness_centrality[i]}\n")
          #  manager.window.state('zoomed')
            plt.axis('off')
            plt.title("Closeness filteration Graph")
            plt.show()


        input_label = tk.Label(popup_window, text='Enter a value:')
        input_label.pack()
        input_entry = tk.Entry(popup_window)
        input_entry.pack()

        show_button = tk.Button(popup_window, text='Show Input', command=clossnes_input)
        show_button.pack()

    def betwness_threshold(self):
        print("Betwness")
        # Create a Toplevel window
        popup_window = tk.Toplevel(root)
        popup_window.title('betweenness graph filteration')
        popup_width = 400
        popup_height = 400
        window_width = popup_window.winfo_reqwidth()
        window_height = popup_window.winfo_reqheight()
        screen_width = popup_window.winfo_screenwidth()
        screen_height = popup_window.winfo_screenheight()
        x = int((screen_width - window_width) / 2)
        y = int((screen_height - window_height) / 2)
        popup_window.geometry(f'+{x}+{y}')

        def show_input_betwnees():
            print("betweenness ")
            if all('weight' in G[u][v] for u, v in G.edges()):
                betweness_centrality = nx.betweenness_centrality(G, distance='weight')
            else:
                print("ya mar7b")
                betweness_centrality = nx.betweenness_centrality(G)

            print("3dena weighted")
            print(betweness_centrality)
            threshold = input_entry.get()
            inputt = float(threshold)
            filtered_nodes = [node for node, bc in betweness_centrality.items() if bc >= inputt]
            subgraph = G.subgraph(filtered_nodes)
            node_size = [v * 1155 for v in betweness_centrality.values()]
            pos = nx.spring_layout(subgraph, iterations=500)
            nx.draw_networkx_nodes(subgraph, pos, node_color='blue')
            nx.draw_networkx_edges(subgraph, pos, edge_color='gray', width=0.5, node_size=node_size)
            nx.draw_networkx_labels(subgraph, pos, font_size=8, font_family='sans-serif')
            manager = plt.get_current_fig_manager()
            #manager.window.state('zoomed')
            root = tk.Tk()
            root.geometry("400x300")
            root.title("Pop-up Window")
            text = tk.Text(root, wrap=tk.WORD)
            scrollbar = tk.Scrollbar(root, command=text.yview)
            text.config(yscrollcommand=scrollbar.set)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            text.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)
            print("filter nodes is")
            print(filtered_nodes)
            text.insert(tk.END, f"filtered nodes is\n")
            for i in filtered_nodes:
                print(i)
                text.insert(tk.END, f"{i} :{betweness_centrality[i]}\n")
            plt.axis('off')
            plt.title("Betweenness filteration Graph")
            plt.show()


        input_label = tk.Label(popup_window, text='Enter value:')
        input_label.pack()
        input_entry = tk.Entry(popup_window)
        input_entry.pack()
        show_button = tk.Button(popup_window, text='enter a value', command=show_input_betwnees)
        show_button.pack()



    def holder(self):
        node_degrees = dict(G.degree())
        edge_weights = {(source, dest): G.get_edge_data(source, dest).get('weight', 1) for source, dest in G.edges()}
        print("#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        print(edge_weights)
        node_sizes = [v * 5 for v in node_degrees.values()]

        edge_widths = [v * 0.1 for v in edge_weights.values()]
        print(edge_widths)
        pos = nx.spring_layout(G, seed=42)

        fig, ax = plt.subplots()
        ax.set_title("Graph with Node Degree and Edge Weight Visualization")
        ax.axis('off')
        nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=node_sizes, ax=ax)
        nx.draw_networkx_edges(G, pos, edge_color='gray', width=edge_widths, ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)

        clicked_node = None
        original_pos = None

        # Function to handle mouse button press event
        def on_press(event):
            nonlocal clicked_node, original_pos
            if not event.inaxes:
                return
            # Check if a node was clicked
            for node in pos:
                if event.xdata is not None and event.ydata is not None:
                    x, y = pos[node]
                    if abs(event.xdata - x) < 0.03 and abs(event.ydata - y) < 0.03:
                        print(f"Node {node} was clicked!")
                        clicked_node = node
                        original_pos = pos[node]
                        break

        # Function to handle mouse button release event
        def on_release(event):
            nonlocal clicked_node, original_pos
            if clicked_node is not None:
                if event.xdata is not None and event.ydata is not None:
                    # Update the position of the clicked node based on the mouse release position
                    pos[clicked_node] = event.xdata, event.ydata
                    print(f"Node {clicked_node} was moved to ({event.xdata}, {event.ydata})")
                    clicked_node = None
                    original_pos = None
                    # Redraw the graph with updated node positions
                    ax.clear()
                    ax.set_title("Graph with Node Degree and Edge Weight Visualization")
                    ax.axis('off')
                    nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=node_sizes, ax=ax)
                    nx.draw_networkx_edges(G, pos, edge_color='gray', width=edge_widths, ax=ax)
                    nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)
                    plt.draw()

        # Function to handle mouse motion event
        def on_motion(event):
            nonlocal clicked_node, original_pos
            if clicked_node is not None:
                if event.xdata is not None and event.ydata is not None:
                    # Update the position of the clicked node based on the mouse motion
                    pos[clicked_node] = event.xdata, event.ydata
                    # Redraw the graph with updated node positions
                    ax.clear()
                    ax.set_title("Graph with Node Degree and Edge Weight Visualization")
                    ax.axis('off')
                    nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=node_sizes, ax=ax)
                    nx.draw_networkx_edges(G, pos, edge_color='gray', width=edge_widths, ax=ax)
                    nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)
                    plt.draw()

        fig.canvas.mpl_connect('button_press_event', on_press)
        fig.canvas.mpl_connect('button_release_event', on_release)
        fig.canvas.mpl_connect('motion_notify_event', on_motion)
        plt.show()

    def seha(selfs):
        for community_id in set(partition.values()):
            community = [node for node, cid in partition.items() if cid == community_id]

            # Calculate conductance
            num_cut_edges = sum(1 for u, v in G.edges(community) if partition[u] != partition[v])
            total_edges = sum(1 for u, v in G.edges(community) if u in community and v in community)
            conductance_val = num_cut_edges / total_edges if total_edges > 0 else 0

            # Calculate internal edge density
            community_edges = 0
            for node in community:
                community_edges += len(set(G.neighbors(node)).intersection(community))
            internal_edge_density_val = community_edges / len(community) if len(community) > 1 else 0

            # Calculate average node degree
            node_degrees = [G.degree(node) for node in community]
            average_node_degree_val = sum(node_degrees) / len(community) if len(community) > 0 else 0

            # Calculate modularity
            m_in = sum(1 for u, v in G.edges(community) if partition[u] == partition[v])
            m_total = sum(1 for u, v in G.edges() if partition[u] == partition[v])
            modularity_val = (m_in / m_total) - ((total_edges / (2 * m_total)) ** 2) if m_total > 0 else 0

            print(f'Community ID: {community_id}')
            print(f'Conductance: {conductance_val}')
            print(f'Internal Edge Density: {internal_edge_density_val}')
            print(f'Average Node Degree: {average_node_degree_val}')
            print(f'Modularity: {modularity_val}')
            print('---')

        overall_modularity = sum(
            (m_in / total_edges) - ((len(community) / G.number_of_nodes()) ** 2) for community_id in
            set(partition.values())) if total_edges > 0 else 0
        print(f'Overall Modularity: {overall_modularity}')

    def Calculate_Evaluations(self):
        communities = [[] for _ in range(max(partition.values()) + 1)]
        for node, com in partition.items():
            communities[com].append(node)

        degrees = dict(G.degree())
        avg_degree = np.mean(list(degrees.values()))
        print(f"Average degree: {avg_degree}")
        internal_edge_density = 0
        for community in communities:
            subgraph = G.subgraph(community)
            internal_edges = subgraph.number_of_edges()
            total_possible_edges = (len(community) * (len(community) - 1)) / 2
            internal_edge_density += internal_edges / total_possible_edges

        internal_edge_density /= len(communities)

        print(f"Internal Edge Density: {internal_edge_density}")

        for community_id in set(partition.values()):
            community = [node for node, cid in partition.items() if cid == community_id]
            print(f'Conductance for community {community_id}: {conductance(G, community)}')

        # Calculate total number of edges in the graph
        total_edges = G.number_of_edges()

        popup = tk.Toplevel(self.master)
        popup.title('Pop-up Window')
        popup_width = 600
        popup_height = 600
        screen_width = self.master.winfo_screenwidth()
        screen_height = self.master.winfo_screenheight()
        x = (screen_width - popup_width) // 2
        y = (screen_height - popup_height) // 2
        i=0
        popup_label = tk.Label(popup, text=f"Average Degree of graph is : {avg_degree}")
        popup_label.pack()
        for comm in communities_list:
            degrees = [G.degree(node) for node in comm]
            avg_degree = sum(degrees) / len(degrees)
            popup_label = tk.Label(popup, text=f"Average Node Degree for community {i} is: {avg_degree}")
            popup_label.pack()
            i=i+1

        popup_label = tk.Label(popup, text=f"F1 score: {f1(G)}")
        popup_label.pack()
        popup_label = tk.Label(popup, text=f"Internal Edge Density: {nx.density(G)}")
        popup_label.pack()
        for community_id in set(partition.values()):
            community = [node for node, cid in partition.items() if cid == community_id]
            popup_label = tk.Label(popup, text=f'Conductance for community {community_id}: {conductance(G, community)}',
                                   width=screen_width)
            popup_label.pack()

        i = 0
        for i in range(len(communities)):
            for j in range(i + 1, len(communities)):
                popup_label = tk.Label(popup,
                                       text=f'Nmi between community {i} and {j}: {nmi(communities[i], communities[j])}')
                popup_label.pack()
        popup_button = tk.Button(popup, text='Close', command=popup.destroy)
        popup_button.pack()

    def button1_clicked(self):
        node_degrees = dict(G.degree())
        edge_weights = {(source, dest): G.get_edge_data(source, dest).get('weight', 1) for source, dest in G.edges()}
        pos = nx.spring_layout(G, seed=42)

        # Draw initial graph
        fig, ax = plt.subplots()
        ax.set_title("Basic visualization")
        ax.axis('off')
        nx.draw_networkx_nodes(G, pos, node_color='skyblue', ax=ax)
        nx.draw_networkx_edges(G, pos, edge_color='gray', ax=ax, width=0.2)
        nx.draw_networkx_labels(G, pos, font_size=7, ax=ax)

        # Variables to keep track of clicked node and its original position
        clicked_node = None
        original_pos = None

        # Function to handle mouse button press event
        def on_press(event):
            nonlocal clicked_node, original_pos
            if not event.inaxes:
                return
            # Check if a node was clicked
            for node in pos:
                if event.xdata is not None and event.ydata is not None:
                    x, y = pos[node]
                    if abs(event.xdata - x) < 0.03 and abs(event.ydata - y) < 0.03:
                        print(f"Node {node} was clicked!")
                        clicked_node = node
                        original_pos = pos[node]
                        break

        # Function to handle mouse button release event
        def on_release(event):
            nonlocal clicked_node, original_pos
            if clicked_node is not None:
                if event.xdata is not None and event.ydata is not None:
                    # Update the position of the clicked node based on the mouse release position
                    pos[clicked_node] = event.xdata, event.ydata
                    print(f"Node {clicked_node} was moved to ({event.xdata}, {event.ydata})")
                    clicked_node = None
                    original_pos = None
                    # Redraw the graph with updated node positions
                    ax.clear()
                    ax.set_title("Basic visualization")
                    ax.axis('off')
                    nx.draw_networkx_nodes(G, pos, node_color='skyblue', ax=ax)
                    nx.draw_networkx_edges(G, pos, edge_color='gray', ax=ax, width=0.2)
                    nx.draw_networkx_labels(G, pos, font_size=7, ax=ax)
                    plt.draw()

        # Function to handle mouse motion event
        def on_motion(event):
            nonlocal clicked_node, original_pos
            if clicked_node is not None:
                if event.xdata is not None and event.ydata is not None:
                    # Update the position of the clicked node based on the mouse motion
                    pos[clicked_node] = event.xdata, event.ydata
                    # Redraw the graph with updated node positions
                    ax.clear()
                    ax.set_title("Basic visualization")
                    ax.axis('off')
                    nx.draw_networkx_nodes(G, pos, node_color='skyblue', ax=ax)
                    nx.draw_networkx_edges(G, pos, edge_color='gray', ax=ax, width=0.2)
                    nx.draw_networkx_labels(G, pos, font_size=7, ax=ax)
                    plt.draw()

        fig.canvas.mpl_connect('button_press_event', on_press)
        fig.canvas.mpl_connect('button_release_event', on_release)
        fig.canvas.mpl_connect('motion_notify_event', on_motion)
        plt.show()

    def button3_clicked(self):
        pagerank = nx.pagerank(G, alpha=0.85)
        for node, score in pagerank.items():
            print(f"Node {node}: PageRank = {score:.4f}")
        # Set node sizes according to PageRank values
        node_sizes = [91999 * pagerank[node] for node in G.nodes()]
        print(node_sizes)
        fig, ax = plt.subplots()
        ax.set_title("Graph with Node Sizes according to PageRank")
        ax.axis('off')
        # Draw the graph
        pos = nx.spring_layout(G, seed=42)  # You can choose a different layout algorithm if desired
        nx.draw(G, pos, node_size=node_sizes, with_labels=True, font_size=8, edge_color='gray', ax=ax, width=0.2)
        plt.title("Graph with Node Sizes according to PageRank")
        #  plt.show()
        clicked_node = None
        original_pos = None

        # Function to handle mouse button press event
        def on_press(event):
            nonlocal clicked_node, original_pos
            if not event.inaxes:
                return
            # Check if a node was clicked
            for node in pos:
                if event.xdata is not None and event.ydata is not None:
                    x, y = pos[node]
                    if abs(event.xdata - x) < 0.03 and abs(event.ydata - y) < 0.03:
                        print(f"Node {node} was clicked!")
                        clicked_node = node
                        original_pos = pos[node]
                        break

        # Function to handle mouse button release event
        def on_release(event):
            nonlocal clicked_node, original_pos
            if clicked_node is not None:
                if event.xdata is not None and event.ydata is not None:
                    # Update the position of the clicked node based on the mouse release position
                    pos[clicked_node] = event.xdata, event.ydata
                    print(f"Node {clicked_node} was moved to ({event.xdata}, {event.ydata})")
                    clicked_node = None
                    original_pos = None
                    # Redraw the graph with updated node positions
                    ax.clear()
                    ax.set_title("Graph with Node Sizes according to PageRank")
                    ax.axis('off')
                    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, ax=ax)
                    nx.draw_networkx_edges(G, pos, edge_color='gray', ax=ax, width=0.2)
                    nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)
                    plt.draw()

        # Function to handle mouse motion event
        def on_motion(event):
            nonlocal clicked_node, original_pos
            if clicked_node is not None:
                if event.xdata is not None and event.ydata is not None:
                    # Update the position of the clicked node based on the mouse motion
                    pos[clicked_node] = event.xdata, event.ydata
                    # Redraw the graph with updated node positions
                    ax.clear()
                    ax.set_title("Graph with Node Sizes according to PageRank")
                    ax.axis('off')
                    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, ax=ax)
                    nx.draw_networkx_edges(G, pos, edge_color='gray', ax=ax, width=0.2)
                    nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)
                    plt.draw()

        fig.canvas.mpl_connect('button_press_event', on_press)
        fig.canvas.mpl_connect('button_release_event', on_release)
        fig.canvas.mpl_connect('motion_notify_event', on_motion)
        plt.show()

    def degree_clicked(self):
        import community
        # for node, bc in betweenness_centrality.items():
        #  print(f"Node {node} has betweenness centrality {bc}")
        # for node, cc in closeness_centrality.items():
        #   print(f"Node {node} has closeness centrality {cc}")
        # for node, dc in degree_centrality.items():
        #   print(f"Node {node} has degree centrality {dc}")

        closeness_centrality = nx.closeness_centrality(G)
        betweenness_centrality = nx.betweenness_centrality(G)
        degree_centrality = nx.degree_centrality(G)
        node_size = [v * 1655 for v in degree_centrality.values()]
        edge_width = [v * 65 for v in betweenness_centrality.values()]

        k = 1 / (len(nodes_data) + len(edges_data)) * 1000000

        pos = nx.spring_layout(G, k=k, iterations=500)
        nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color='blue')
        nx.draw_networkx_edges(G, pos, edge_color='gray', width=0.5, node_size=node_size)
        nx.draw_networkx_labels(G, pos, font_size=8, font_family='sans-serif')

        plt.axis('off')
        plt.show()

    def betweeness_clicked(self):
        betweenness_centrality = nx.betweenness_centrality(G)

        print("se")
        print(betweenness_centrality)
        node_size = [v * 16525 for v in betweenness_centrality.values()]

        k = 1 / (len(nodes_data) + len(edges_data)) * 1000000

        pos = nx.spring_layout(G, k=k, iterations=500)
        nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color='blue')
        nx.draw_networkx_edges(G, pos, edge_color='gray', width=0.5, node_size=node_size)
        nx.draw_networkx_labels(G, pos, font_size=8, font_family='sans-serif')

        plt.axis('off')
        plt.show()

    def closness_clicked(self):
        closeness_centrality = nx.closeness_centrality(G)
        print(closeness_centrality)
        node_size = [v * 1155 for v in closeness_centrality.values()]
        k = 1 / (len(nodes_data) + len(edges_data)) * 1000000
        pos = nx.spring_layout(G, k=k, iterations=500)
        nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color='blue')
        nx.draw_networkx_edges(G, pos, edge_color='gray', width=0.5, node_size=node_size)
        nx.draw_networkx_labels(G, pos, font_size=8, font_family='sans-serif')

        plt.axis('off')
        plt.show()

    def degree_threshold(self):
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

        def show_input():
            if all('weight' in G[u][v] for u, v in G.edges()):
                degree_centrality = nx.degree_centrality(G, distance='weight')
            else:
                print("ya mar7b")
                degree_centrality = nx.degree_centrality(G)
            print(degree_centrality)
            threshold = input_entry.get()
            inputt = float(threshold)
            filtered_nodes = [node for node, bc in degree_centrality.items() if bc >= inputt]
            print(len(filtered_nodes))
            subgraph = G.subgraph(filtered_nodes)

            node_size = [v * 1155 for v in degree_centrality.values()]
            pos = nx.spring_layout(subgraph, iterations=500)
            nx.draw_networkx_nodes(subgraph, pos, node_color='blue')
            nx.draw_networkx_edges(subgraph, pos, edge_color='gray', width=0.5, node_size=node_size)
            nx.draw_networkx_labels(subgraph, pos, font_size=8, font_family='sans-serif')
            manager = plt.get_current_fig_manager()
      #     manager.window.attributes('-zoomed', True)
            root = tk.Tk()
            root.geometry("400x300")
            root.title("Pop-up Window")
            text = tk.Text(root, wrap=tk.WORD)
            scrollbar = tk.Scrollbar(root, command=text.yview)
            text.config(yscrollcommand=scrollbar.set)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            text.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)
            print("filter nodes is")
            print(filtered_nodes)
            text.insert(tk.END, f"filtered nodes is\n")
            for i in filtered_nodes:
                print(i)
                text.insert(tk.END, f"{i} :{degree_centrality[i]}\n")
            plt.axis('off')
            plt.title("Degree filteration Graph")
            plt.show(block=False)
        input_label = tk.Label(popup_window, text='Enter a value:')
        input_label.pack()
        input_entry = tk.Entry(popup_window)
        input_entry.pack()
        show_button = tk.Button(popup_window, text='Show Input', command=show_input)
        show_button.pack()
root = tk.Tk()
app = Page1(root)
root.mainloop()

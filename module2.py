import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Load dataset
file_path = 'wine 2.csv' 
df = pd.read_csv(file_path, usecols=['varietal', 'category', 'appellation', 'rating'])

df.dropna(subset=['varietal', 'category', 'rating'], inplace=True)

# Create a graph
G = nx.Graph()

# Adding nodes and edges
for _, row in df.iterrows():
    varietal = row['varietal']
    category = row['category']
    appellation = row['appellation']
    rating = row['rating']
    
    G.add_node(varietal, type="varietal", rating=rating)
    G.add_node(category, type="category")
    G.add_edge(varietal, category, label="belongs to")  

    if pd.notna(appellation):
        G.add_node(appellation, type="appellation")
        G.add_edge(category, appellation, label="appellation")  

# Ensuring that the graph is not empty
if len(G.nodes) == 0:
    print("Graph is empty. Check your dataset.")
else:
    # Get the top 50 varietals by rating
    top_varietals = sorted(
        [n for n, d in G.nodes(data=True) if d["type"] == "varietal"],  
        key=lambda x: G.nodes[x]["rating"],  
        reverse=True
    )[:50]  

    # Get all connected nodes (categories & appellations)
    selected_nodes = set(top_varietals)  
    for varietal in top_varietals:
        selected_nodes.update(G.neighbors(varietal))  # Add connected nodes

    H = G.subgraph(selected_nodes)

    rating_values = [H.nodes[node]["rating"] for node in top_varietals]
    scaler = MinMaxScaler(feature_range=(100, 1000))
    scaled_sizes = scaler.fit_transform(np.array(rating_values).reshape(-1, 1)).flatten()

    node_colors = [
        "green" if H.nodes[n]["type"] == "category" else 
        "yellow" if H.nodes[n]["type"] == "varietal" else 
        "blue" for n in H.nodes
    ]

    # Assign node sizes (only for varietals, others are smaller)
    node_sizes = [
        scaled_sizes[top_varietals.index(n)] if n in top_varietals else 300  
        for n in H.nodes
    ]




# Compute Degree Centrality (importance based on direct connections)
degree_centrality = nx.degree_centrality(G)
top_degree = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
print("\nTop 5 Wine types Degree Centrality:")
for node, score in top_degree:
    print(f"{node}: {score:.4f}")

# Compute Betweenness Centrality (importance based on bridging different parts of the network)
betweenness_centrality = nx.betweenness_centrality(G)
top_betweenness = sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
print("\nTop 5 Wine types Betweenness Centrality:")
for node, score in top_betweenness:
    print(f"{node}: {score:.4f}")

# Compute Eigenvector Centrality (importance based on being connected to other important nodes)
try:
    eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000)
    top_eigenvector = sorted(eigenvector_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
    print("\nTop 5 Wine types Eigenvector Centrality:")
    for node, score in top_eigenvector:
        print(f"{node}: {score:.4f}")
except nx.NetworkXError as e:
    print("\nEigenvector Centrality failed:", str(e))




#Wine type + Edge count table
node_types = ["varietal", "category", "appellation"] 
edge_counts = {}
for node in G.nodes:
    node_type = G.nodes[node].get("type") 
    if node_type in node_types: 
        edge_counts[node] = G.degree(node) 

edge_counts_df = pd.DataFrame(list(edge_counts.items()), columns=["Node", "Number of Edges"])
edge_counts_df = edge_counts_df.sort_values(by="Number of Edges", ascending=False).head(5)
print(edge_counts_df)




# Graph for all Wine types 
pos = nx.spring_layout(H, seed=42)  # Generate layout for nodes
plt.figure(figsize=(12, 8))
nx.draw(
    H,
    pos,
    with_labels=True,
    node_size=node_sizes,  
    font_size=7,          
    alpha=0.7,            
    node_color=node_colors,  
    edge_color="gray",    
    width=0.7             
)
plt.title("Top 50 Wine Varietals & Their Categories/Appellations")
#no overlap 
plt.tight_layout()
plt.show()


#Graph for side by side Red and White for popular vote
target_nodes = ['Red', 'White']
selected_nodes = set(target_nodes)
for node in target_nodes:
    selected_nodes.update(H.neighbors(node)) 

H_subgraph = H.subgraph(selected_nodes)
node_sizes = [H_subgraph.degree(node) * 100 for node in H_subgraph.nodes] 

node_colors = ['red' if node == 'Red' else 'white' if node == 'White' else 'lightgray' for node in H_subgraph.nodes]
pos = nx.spring_layout(H_subgraph, seed=42)

#plot
plt.figure(figsize=(12, 8))
nx.draw(
    H_subgraph, pos, with_labels=True, node_size=node_sizes, font_size=10, alpha=0.7,
    node_color=node_colors, edge_color="gray", width=0.7
)

plt.title('Comparison of Red and White Wine Nodes and Their Network Relationships')
plt.tight_layout()
plt.show()

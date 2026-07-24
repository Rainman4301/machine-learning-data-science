# GNN Task Types: Node Classification vs. Graph Classification vs. Link Prediction

The core question that separates these three tasks is always the same:

> **"What am I putting a label on?"**

Everything else follows from the answer.

---

## 1. Node Classification — label a *node*

- **Input:** one graph (or a few), with some nodes labeled and others not.
- **Question:** "What is *this specific node*?"
- **Example (`gnn_node_classification.ipynb`):** 2,708 papers connected by citations. Some papers' subjects are known, most aren't. Predict: what subject is *this* paper?
- **Role of graph structure:** context. Who a paper cites is a clue about its topic, but the *node itself* is what receives the prediction.
- **Real-world examples:** Is *this* user a bot? What category does *this* product belong to?

## 2. Graph Classification — label the *whole graph*

- **Input:** many separate, smaller graphs, each with its own label.
- **Question:** "What is *this entire structure*, taken as one unit?"
- **Example (`gnn_graph_classification.ipynb`):** each graph is its own self-contained world (20–60 nodes). Predict: which random-graph family generated *this whole graph*?
- **Role of graph structure:** everything. No individual node gets a label — all node embeddings get pooled into one summary vector per graph before the final prediction.
- **Real-world examples:** Is *this molecule* toxic? Does *this whole social-media cluster* look like a bot network?

## 3. Link Prediction — label a *pair of nodes* (specifically, whether they're connected)

- **Input:** one graph, with some real edges hidden and mixed in with fake ones.
- **Question:** "Should *these two specific nodes* be connected?"
- **Example (`gnn_link_prediction.ipynb`):** same Cora graph as notebook 1, but instead of asking about one paper's identity, it asks about a *relationship* between two papers — does A cite B?
- **Role of graph structure:** the target itself. Neither node gets a label; the *edge* (or its absence) is what's being predicted.
- **Real-world examples:** Will *this user* follow *that user*? Is *this transaction* part of a laundering ring with *that account*?

---

## Side-by-Side Summary

| | What's the "unit" being labeled? | How many predictions per graph? | Pooling? |
|---|---|---|---|
| **Node classification** | a node | one label per node | none — each node keeps its own output |
| **Graph classification** | the whole graph | one label per graph | yes — all node embeddings collapse into one vector |
| **Link prediction** | a pair of nodes | one label per node-pair | none — but the *decoder* combines two node embeddings instead of reading one |

---

## What Stays the Same

The GCN **encoder** — stacking `GCNConv` layers to build node embeddings from neighbors — is **identical across all three tasks**. The only thing that changes is the *last step*, i.e. what you do with the node embeddings once you have them:

- **Node classification:** keep every node's embedding as-is, feed it to a classifier.
- **Graph classification:** average all node embeddings into one vector (pooling), feed *that* to a classifier.
- **Link prediction:** take two node embeddings and combine them (e.g. dot product) into a single compatibility score.

Same engine (the GCN encoder), different "what do I do with the output" step bolted on the end.

---

## Mapping to Real-World Use Cases

| Domain | Nodes | Edges | What the GNN Predicts | Task Type |
|---|---|---|---|---|
| Social Media | Users | Friendships / Follows | Product recommendations or fake account detection | Node classification (fake account) / Link prediction (recommendation) |
| Drug Discovery | Atoms | Chemical Bonds | Molecular properties or drug effectiveness | Graph classification |
| Navigation | Intersections | Roads | Traffic speeds and ETA | Edge-level regression (a cousin of link prediction) |
| Financial Fraud | Accounts | Transactions | Fraudulent money-laundering rings | Node or link prediction |
| Knowledge Graphs | Concepts / Facts | Logical Relationships | Question answering and search engines | Link prediction |

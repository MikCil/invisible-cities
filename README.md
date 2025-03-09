# Invisible Cities: A Semantic Network Analysis

This repository contains the code and data used in the paper "Evoking the Invisible: A semantic network analysis of concrete and abstract imagery in Calvino's Le Città Invisibili", submitted for review to the Journal of Computational Literary Studies (JCLS).

## Repository Contents

### Data
For copyright reasons, it is not possible to share the dataset containing all city descriptions from Le Città Invisibili, organized by overall ID, Category, Category ID and Chapter. However, all network data is stored in the directory `network_data/`.

### Code
- `main.py`: Processes the JSON corpus and generates CSV files containing nodes and edges for:
  - The main corpus network
  - Individual category networks
  
The textual preprocessing, lemma extraction, and network construction methodology are detailed in:
- The paper's Methodology section
- Code comments within `main.py`

### Output
- `network_data/`: Directory containing all edges and nodes data visualized using Gephi

# invisible-cities
A repository containing the code and data used in the paper ""

The invisible_cities_corpus.json file contains all the city descriptions from Le Città Invisibili, categorized by overall ID, category, category ID and chapter.

The main.py file takes the JSON corpus as an input and creates a directory of CSV files storing nodes and edges of the main corpus network and of the networks of each single category. The textual preprocessing and lemma extraction, as well as the network construction strategy, is detailed in the Methodology section of the paper and in the file comments.

The network_data directory contains all edges and nodes data that were obtained and visualized via Gephi.

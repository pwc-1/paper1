# The Framework CLUE
MindSpore implementation for CLUE (WWW'24 Jointly Canonicalizing and Linking Open Knowledge Base via Unified Embedding Learning)

## Environment

Computational platform: PyTorch 1.4.0, NVIDIA Geforce RTX 3090 (GPU), Inter i9-10900X (CPU), CUDA Toolkit 10.0

Development language: Python 3.7
       
Liabraries are listed as follow, which can be installed via the command `pip install -r requirements.txt`.
```
numpy, scipy, tqdm, scikit-learn, nltk, os, sys, collections, itertools, gensim, logging, argparse, subprocess, pickle, cudatoolkit=10.0, mindspore==2.2.11
```
Please download the crawl-300d-2M.vec.zip from https://fasttext.cc/docs/en/english-vectors.html.

## Data sets
We provide both the data sets (ReVerb45K, OPIEC59K) in the folder `data/`. 
### ReVerb45K (obtained from [1])  
triples: 45k; noun phrases: 15.5k   
used by the previous work: [CESI](https://dl.acm.org/doi/abs/10.1145/3178876.3186030) [SIST](https://ieeexplore.ieee.org/abstract/document/8731346) [JOCL](https://dl.acm.org/doi/abs/10.1145/3448016.3452776)   

### OPIEC59K (obtained from [2])   
triples: 59k; noun phrases: 22.8k      
used by the previous work: [CMVC](https://dl.acm.org/doi/abs/10.1145/3534678.3539449)   

## Reproduce
### Run CLUE on the ReVerb45K data set:
    python CLUE_main_reverb.py
### Run CLUE on the OPIEC59K data set:
    python CLUE_main_opiec.py


[1] Vashishth S, Jain P, Talukdar P. Cesi: Canonicalizing open knowledge bases using embeddings and side information. WWW'2018, 1317-1327.   

[2] Shen W, Yang Y, Liu Y. Multi-View Clustering for Open Knowledge Base Canonicalization. SIGKDD'2022, 1578-1588.

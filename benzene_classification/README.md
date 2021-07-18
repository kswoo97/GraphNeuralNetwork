# Classification of Benzene shape in a molecule

### Source Paper : https://proceedings.neurips.cc/paper/2020/file/417fbbf2e9d5a28a855a11894b2e795a-Paper.pdf
### Data : https://github.com/google-research/graph-attribution/tree/main/data/benzene
### Result is in Graph_GNN file

Indentifying whether a certain molecular structure contains a benzen shape.  
특정 분자 구조에 벤젠 형태가 포함되어 있는지 확인하는 task 입니다.  

- GNN structure : NNConv(Neural message passing) / (Gilmer et al. 17.)
- Pooling : Summation
- Readout : Single layer MLP with sigmoid activation

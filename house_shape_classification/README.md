# Classification of house shape in a graph

### Source Paper : https://arxiv.org/abs/1903.03894 (GNN-Explainer)
### Data : https://arxiv.org/abs/1903.03894 (GNN-Explainer)
### Result is in House_shape_generator.ipynb

Indentifying whether a certain graph contains a house shape 
특정 그래프에 집 모양의 subgraph가 포함되어 있는지 확인하는 task입니다.

- GNN Structure : GraphSAGE / (Hamilton et al. 17.)
- Pooling : SUmmation
- Readout : Single layer MLP with sigmoid activation

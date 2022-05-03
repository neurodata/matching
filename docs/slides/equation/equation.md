---
marp: true
theme: slides
size: 16:9
paginate: false
---

# Graph matching for neuroscience

$\min_{P \in \mathcal{P}_s} \left [ \sum_{i=1}^K \|A_{LL}^{(i)} - P A_{RR}^{(i)} P^T\|_F^2 + \sum_{i=1}^K \|A_{LR}^{(i)} P^T - P A_{RL}^{(i)} \|_F^2 + tr(S P^T) \right ]$

where:
- $\mathcal{P}_s$ is the set of permutation matrices *which respect fixed seeds, $s$,* where the seeds give a known correspondence between some of the neurons.
- $A_{LL}$ and $A_{RR}$ are the within-hemisphere (ipsilateral) subgraphs.
- $A_{LR}$ and $A_{RL}$ are the between-hemisphere (contralateral) subgraphs.
- $S$ is a matrix of similarity scores between neurons, e.g. morphology (NBLAST).
- $K$ is the number of *layers* or edge types e.g. axo-axonic, axo-dendritic, etc.
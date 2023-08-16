# interpretable meta neural ordinary differential equation (iMODE)
This repository contains the code for the paper:
- [Metalearning Generalizable Dynamics from Trajectories](https://doi.org/10.1103/PhysRevLett.131.067301)

Find the paper story coverage at:
- [AI Learns to Play with a Slinky](https://physics.aps.org/articles/v16/s119)

<!---describing the work--->
In this work, we propose a physics-informed deep learning approach to build reduced-order models of physical systems. We use Slinky as a demonstration. The approach introduces a **Euclidena symmetric neural network architecture (ESNN)**, trained under the **neural ordinary differential equation** framework. The ESNN implements a physics-guided architecture that simultaneously preserves energy invariance and force equivariance on Euclidean transformations of the input, including translation, rotation, and reflection. We demonstrate that the ESNN approach is able to accelerate simulation by roughly **60** times compared to traditional numerical methods and achieve a superior generalization performance, i.e., the neural network, trained on a single demonstration case, predicts accurately on unseen cases with different Slinky configurations and boundary conditions.

## Requirements

## Tentative Notes


## Citation
If you use this code for part of your project or paper, or get inspired by the associated paper, please cite:  

    @article{Li2023Metalearning,
        title = {Metalearning Generalizable Dynamics from Trajectories},
        author = {Li, Qiaofeng and Wang, Tianyi and Roychowdhury, Vwani and Jawed, M. K.},
        journal = {Phys. Rev. Lett.},
        volume = {131},
        issue = {6},
        pages = {067301},
        numpages = {6},
        year = {2023},
        month = {Aug},
        publisher = {American Physical Society},
        doi = {10.1103/PhysRevLett.131.067301},
        url = {https://link.aps.org/doi/10.1103/PhysRevLett.131.067301}
    }

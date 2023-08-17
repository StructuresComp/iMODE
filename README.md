# interpretable meta neural ordinary differential equation (iMODE)
This repository contains the code for the paper:
- [Metalearning Generalizable Dynamics from Trajectories](https://doi.org/10.1103/PhysRevLett.131.067301)

Find the paper story coverage at:
- [AI Learns to Play with a Slinky](https://physics.aps.org/articles/v16/s119)

## short summary
In this work, we propose a *meta-learning for physics* algorithm, named **interpretable meta neural ordinary differential equation (iMODE)**. The iMODE algorithm, instead of training a neural network as a surrogate prediction model for a specic dynamic system as normally seen in *deep learning for physics* literature, captures the common or meta knowledge across a family of dynamical systems and thus is capable of fast adaptation to unseen systems. 

## long summary
We call a dynamical system with a specific set of physical parameters "a system instance" and different system instances with the same form of dynamics but different physical parameters "a system family". iMODE takes trajectories from different system instances within a family as the training dataset. The iMODE algorithm
1. constructs a neural network taking the system state and latent adaptation parameters as input, generates derivatives (i.e. acceleration for second-order dynamical systems) as output;
2. calculates prediction trajectories from given initial conditions under the **neural ordinary differential equation (NODE)** framework;
3. with a bi-level optimization, finds the optimal weights of the neural network with which a few example trajectories can adapt the common adaptation parameters to suitable values for each individual system instance in a few (e.g. 5) gradient descent steps.

We demonstrate that 
1. by learning the meta knowledge (in the form of the neural network weights), i.e. the shared dynamics form across system instances, the iMODE neural network is capable of fast adaptation to unseen systems (and acting as an accurate surrogate prediction model);
2. thanks to the model agnostic nature of iMODE, we can embed different inductive biases into the neural network, such as the Euclidean symmetry of energy field and induced force field ([Euclidean symmetric neural network](https://github.com/StructuresComp/slinky-is-sliding)).
3. the latent parameter space is physical meaningful. The optimal dimension of the latent space can be determined via principal component analysis and is shown to equal to the dimension of the true physical parameter space across cases;
4. *neural gauge*: a diffeomorphism can be constructed from the latent parameter space to the true physical parameter space. This facilitates fast measurement of the physical parameters of an unseen system.

## Requirements
- PyTorch (1.11.0)
- [torchdiffeq](https://github.com/rtqichen/torchdiffeq)

## Inspirations
This work significantly benefits from the following:
- [neural ordinary differential equations](https://github.com/rtqichen/torchdiffeq)
- [Model-Agnostic Meta-Learning](https://github.com/cbfinn/maml)
- [Euclidean symmetric neural network](https://github.com/StructuresComp/slinky-is-sliding)

## Tentative Notes
The repo currently contains the fully functional but unformatted code for the pendulum case. The neural network architecture, training procedure, and code structure are pretty similar to the rest of the cases. We will upload the rest soon.
### todos
- [ ] formatting the codes
- [ ] adding other cases
  
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

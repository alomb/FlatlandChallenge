# Flatland Challenge

> The Flatland challenge aims to address the problem of train scheduling and rescheduling by providing a simple grid world environment and allowing for diverse experimental approaches.
> This is the second edition of this challenge. In the first one, participants mainly used solutions from the operations research field. In this second edition we are encouraging participants to use solutions which leverage the recent progress in reinforcement learning.

For more information visit the [official page](https://www.aicrowd.com/challenges/neurips-2020-flatland-challenge).

The following work is related to the second edition of this challenge started in Summer 2020. 
In the first edition, which took place in 2019, the majority of the participants delivered solutions from the operations research field, this edition tries to encourage participants to elaborate Reinforcement Learning solutions, indeed the challenge has been selected to be featured in the NeurIPS 2020 Accepted competitions. 
In this work we propose and describe the implementation of some Deep Reinforcement Learning solutions, leveraging the most recent outcomes in this vibrant and exciting field of study.
A detailed explanation and analysis of our strategies can be read in the report.

Project structure
---
    .
    ├── models (contains PyTorch neural network saved parameters of our final solution)
    ├── modules (contains Git submodules)
    │   ├── MARL-Papers (list of MARL papers)
    │   └── neurips2020-flatland-starter-kit (contains files useful for submitting solutions to the Challenge)
    ├── report (contains resources and source files to generate the Latex report)
    ├── single (contains preliminary works on single agent setting performed on previous versions of Flatland)
    └── src
        ├── common (contains source code in common with all the approaches)
        ├── curriculum (contains files related with the curriculum approach)
        ├── d3qn (contains files related with the D3QN approach)
        │   ├── hyperparameters
        │   │   └── server.py (code to run Sweeps)
        │   ├── d3qn_flatland.py (main loop)
        │   ├── d3qn_main.py (hyperparameter definition and starting point)
        │   ├── eval_d3qn.py (code to evaluate the policy)
        │   ├── memory.py (experience replay)
        │   ├── model.py (the network architecture)
        │   └── policy.py (the )
        └──  psppo (contains files related with the parameter sharing PPO approach)

Team members
---
Alessandro Lombardi - University of Bologna - [98.alessandro.lombardi@gmail.com](mailto:98.alessandro.lombardi@gmail.com) - https://github.com/AlessandroLombardi

Fiorenzo Parascandolo - University of Bologna - [fiorenzo.parascandolo@gmail.com](mailto:fiorenzo.parascandolo@gmail.com) - https://github.com/FiorenzoParascandolo1
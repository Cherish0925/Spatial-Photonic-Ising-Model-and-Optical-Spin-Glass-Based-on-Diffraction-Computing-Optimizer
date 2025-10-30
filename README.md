# Spatial-Photonic-Ising-Model-and-Optical-Spin-Glass-Based-on-Diffraction-Computing-Optimizer
Programming language is Python, based on torch and munpy. It precisely models the light field propagation between multiple planes, enabling the research of spatial light Ising machines and spin glasses. It assists the Markov Monte Carlo algorithm by optimizing the Hamiltonian of the QUBO problem through lens-based Fourier transformation.

# Core library functions
- numpy 1.24.4
- torch 2.4.1+cu118
- scipy 1.10.1
- matplotlib 3.7.5

# Computing Env
- Python 3.8.20 | packaged by conda-forge | (default, Sep 30 2024, 17:44:03) [MSC v.1929 64 bit (AMD64)] on win32
- NVIDIA GeForce RTX 3050 4096MiB NVIDIA-SMI 546.30                 Driver Version: 546.30       CUDA Version: 12.3
- SKU	LENOVO_MT_82JW_BU_idea_FM_Legion R70002021 	AMD Ryzen 5 5600H with Radeon Graphics，3301 Mhz

# Project File Description
- "General_Optical_Ising_Optimizer.py" The file serves as the entry point for solving QUBO problems. By simply modifying the coupling interaction matrix J of the Ising problem within this file, it can be adapted to the problems you wish to optimize through spatial optical computing.
- "Bandlimited_ASM.py" is a file that uses PyTorch to calculate the light field propagation process between multiple planes, and it includes options for whether to use a lens or not.
- "decompose_J_coupling_matrix.py" The file is adapted for the spatial light Ising calculation system, which can only optimize the coupling interaction problem with a rank of 1 in a single step. Therefore, by leveraging the tensor calculation capabilities of the graphics card, the originally high-rank matrix is decomposed and then stacked into a three-channel dimensional tensor. 
- "datas_process.py" The file is for data visualization and processing.

# An example of visualizing the results of the project operation
*The illustrative examples available for the operation of the project include the optical path corresponding to the simulation system, the definition of the Ising model, the solution of the Ising problem, as well as the solution and research of more complex spin glass systems.*
## Fig 1
**The following diagram clearly illustrates our calculation process: light waves propagate from the liquid crystal spatial light modulator to the lens, and then continue to propagate from the lens to the camera's photoelectric conversion plane.**
![image](https://github.com/Cherish0925/Spatial-Photonic-Ising-Model-and-Optical-Spin-Glass-Based-on-Diffraction-Computing-Optimizer/blob/main/images/Experimental_architecture_diagram.png?raw=true)
## Fig 2
**The following figure clearly illustrates the definition of an Ising problem where all coupling interactions are equal to 1, and the rank of the coupling interaction matrix is 1.**
![image](https://github.com/Cherish0925/Spatial-Photonic-Ising-Model-and-Optical-Spin-Glass-Based-on-Diffraction-Computing-Optimizer/blob/main/images/Ising_Model_Problem_Definition.png?raw=true)
## Fig 3
**The following figure clearly illustrates the solution process of the aforementioned Ising model. The final optimized result demonstrates that the Hamiltonian has finally converged to the global optimum.**
![image](https://github.com/Cherish0925/Spatial-Photonic-Ising-Model-and-Optical-Spin-Glass-Based-on-Diffraction-Computing-Optimizer/blob/main/images/Ising_Model_Problem_Optimization.png?raw=true)
## Fig 4
**The following figure clearly illustrates the definition and optimization of a more complex two-dimensional spin glass system. It also observes a series of key phenomena of the autocorrelation function of spin configuration in different configurations.**
![image](https://github.com/Cherish0925/Spatial-Photonic-Ising-Model-and-Optical-Spin-Glass-Based-on-Diffraction-Computing-Optimizer/blob/main/images/Spin_Glass_Optical_Computing_Sim.png?raw=true)

# Simple instructions for use
- If you want to use the aforementioned multi-plane spatial light computing system to obtain the Hamiltonian of a QUBO Ising problem and optimize the Hamiltonian using the simulated annealing method, then you only need to replace the number of spins and the coupling interaction matrix of the problem you need to solve with the corresponding parts in the corresponding files.
- If you want to use the aforementioned multi-plane spatial light computing system to observe the key parameters and phenomena in some spatial light spin glass systems, then carry out a second DIY based on the existing files in the project.

# Detail
**If you want to have a deeper understanding of this project, you can visit the author's personal blog on Zhihu to check it out.**
https://zhuanlan.zhihu.com/p/1966619744913298117

# deep_residual_voxel_autoencoder (EfficientNet3D)

Repository for EfficientNets extended to 3D classification and the implementation of an EfficientNet Autoencoder in Tensorflow.

Code follows after paper was accepted to the CIRP Design 2022 Conference

In the domain of computer vision, deep residual neural networks like EfficientNet have set new standards in terms of robustness and accuracy. In this work, we present a deep residual 3D autoencoder based on the EfficientNet architecture for transfer learning. For this purpose, we adopted EfficientNet to 3D problems like voxel models derived from a STEP file. 

Paper URL: https://arxiv.org/abs/2202.10099

Cite: R. Schönhof, J. Elstner, R. Manea, S. Tauber, R. Awad, M. F. Huber, Simplified Learning of CAD Features Leveraging a Deep Residual Autoencoder, CIRP Design 2022.

Project Summary:

- efficientnets: the Efficientnet architecture, currently using the bugfix version
- lrp: script for testing heatmap generation, not relevant for the repo
- small_test_scripts: scripts to test ae, not relevant for the repo
- tensorboards: Saved weights and models from previous training sessions
- training.py: autonencoder training
- training_transferlearning: Using the trained encoder, proceed to transferlearning
- vizualitzation.py: vizualize autoencoder results

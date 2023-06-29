# Bachelor End Project - Statistical Model of Deformations 
By Marcus Vroemen, student number: 1582038, email: m.j.vroemen@student.tue.nl <br>
This repository contains the code and documentation for the BEP: _Realistic data augmentation using a statistical deformation model for deep learning-based registration on respiratory CT_<br>

## Introduction

In this project, the statistical deformation model by Corral Acero et al. [1] was extended to generate artificial thoracic exhaled and inhaled imaged, incorperating the training data's population and respiration variations. The model was evaluated on a multi-step ViT developed by I.D. Kolenbrander, and compared with Gryds* as baseline augmentation method. 

## Folder Structure

The repository is organized into the following folders:

- **`/multi-step-ViT`**: Contains the scripts used to generate the results. augmentations.py contains the SMOD and Gryds* augmentations. Other files are used to train the registration model.
- **`/4DCT`**: Contains the datasets used in the project, including any pre-processed data.
- **`/visualization`**: Contains some scripts which were used to generate the figures from the papar

## Augmentation method
The file augmentations.py contains the proposed SMOD and baseline augmentation methods. Two versions of the model can be called: SMOD_full with fully augmented image pairs and SMOD_insp with only augmented inspiration images. Both methods can be combined with the real training images. Additionally, augmented images can be generated every epoch (on-the-fly) and only in the first epoch.
## Installation

To run or contribute to this project, please follow these steps:

1. Clone the repository:

   ```shell
   git clone https://github.com/MarcusVroemen/BEP_MIA_SMOD.git

2. Install the required packages
    ```shell
    pip install -r requirements.txt

## References
[1] https://link.springer.com/chapter/10.1007/978-3-030-21949-9_39

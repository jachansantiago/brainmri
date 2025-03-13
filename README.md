# Brain MRI Segmentation

This repository contains code for Tumor area segmentation brain MRI images using deep learning techniques.


**Clone the repository:**

```bash
git clone https://github.com/jachansantiago/brainmri.git
cd brainmri
```
## Dataset

A subset of the data from the 2016 and 2017 Brain Tumour Image Segmentation (BraTS) challenges is used for this project.

The dataset can be downloaded from the following link: [Task01_BrainTumour.tar](https://drive.google.com/drive/folders/1HqEgzS8BV2c7xYNrZdEAnrHk7osJJ--2).  
**Simply download and extract the data into the root folder of this repository.**

# Training
To train the model, run:

```bash
python train.py --epochs num_epochs --batch_size batch_size
```

This script will create an experiment folder with a 5-fold cross-validation.

# Evaluation
To evaluate a trained model on the test set, run:
```bash
python eval.py --resume experiement_folder
```


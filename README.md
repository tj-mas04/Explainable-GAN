# Explainable GAN-based Brain MRI Tumor Project

This repository contains code for training a CNN classifier and a customized GAN
on a dataset of brain MRI scans, some containing tumors and some without. The
goal is to generate realistic MRI images with controllable tumor presence, and
to analyze GAN behavior using counterfactuals and explainability methods such as
Grad-CAM.

The core focus of this project is **explainable generative adversarial networks**. 
By splitting the latent vector into separate "structure" and "pathology"
dimensions, and integrating a frozen tumor classifier into the generator loss,
we force the GAN to encode tumor-specific features in a controllable way. This
allows us to generate counterfactual image pairs (same anatomy with/without
tumor) and to inspect which pixels the generator alters when toggling the
pathology component. Additionally, Grad-CAM is applied to both the classifier
and discriminator to visualize where each model attends when making decisions.

The novelty lies in combining a disentangled latent space with classifier-
guided training to produce interpretable tumor manipulations, offering a more
transparent understanding of what the GAN has learned compared to a standard
black-box model.

## Structure

- `xgan.ipynb` – Main Jupyter notebook with all data loading, preprocessing,
  classifier training, GAN training, evaluation, Grad-CAM visualizations, and
  counterfactual generation.
- `dataset/` – Folder containing two subfolders:
  - `tumor/` – MRI scans labeled as containing tumors.
  - `no_tumor/` – MRI scans without tumors.
- `train_idx.npy`, `val_idx.npy` – Saved indices for deterministic train/validation split.
- `tumor_classifier_axial.h5` – Saved CNN classifier model.
- `classifier_history.json` – Training history for the classifier.
- `gan_checkpoints_V2_h5/` – Saved generator and discriminator H5 checkpoints every 10 epochs.
- `gan_generated_images_V2_h5/` – Generated images from the GAN during training.
- `outputs_gradcam/` – Grad-CAM visualizations for classifier and discriminator.

## Requirements

Create a virtual environment and install dependencies (e.g.):

```bash
python -m venv venv
.\nvenv\Scripts\activate
pip install -r requirements.txt
```

### Example `requirements.txt`
```
tensorflow
numpy
opencv-python
matplotlib
scikit-learn
seaborn
scikit-image
```

## Usage

Open `xgan.ipynb` in Jupyter Notebook or JupyterLab. Run cells sequentially to:

1. Load and preprocess the dataset.
2. Train the CNN classifier and save the model.
3. Visualize classifier performance and Grad-CAM explanations.
4. Define and train the GAN with separate latent structure and pathology
   components, integrating the frozen classifier to guide pathology encoding.
5. Generate and inspect counterfactual image pairs, Grad-CAM on generated
   samples, and calculate similarity metrics.

## License & Copyright

```
Copyright (c) 2026 Sam T James

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Notes

- Modify latent dimensions and training parameters inside the notebook as
  needed.
- The classifier can be reused for other MRI tasks with minimal changes.

Happy experimenting!

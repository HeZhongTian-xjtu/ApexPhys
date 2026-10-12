# ApexPhys

Official implementation of **“Anchoring Physiological Invariance, Expanding Domain Plasticity: Prior-Stabilized Dynamic Adaptation for Continual rPPG Measurement”** (NeurIPS 2026).

ApexPhys is a continual remote photoplethysmography (rPPG) framework. It combines Backbone Consistency Representation Preservation (BCRP), Hierarchical Expansion Strategy (HES), and Prior-Guided Pathfinding (PGP) to preserve physiological representations while adapting to changing domains.

## Setup

Create a Python environment with CUDA-enabled PyTorch, then install the project dependencies:

```bash
pip install torch numpy scipy opencv-python h5py Pillow tqdm dlib face-alignment einops scikit-image
```

Prepare the datasets and configure their root directories in `utils/engine.py`. Set the face landmark predictor path in `utils/util.py` if face preprocessing is needed.

The training data should follow the format expected by the dataset loaders. For VIPL, each sample is stored as `sample.hdf5` with `video_data` shaped `(C, T, H, W)` and `ecg_data` shaped `(T,)`.

## Train

Run continual learning on the paper's domain sequence:

```bash
python trainer.py \
  --gpu 0 \
  --dataset VIPL1to4_VIPL5_PURE_UBFC_BUAA_MMPD \
  --save_path ./runs \
  --batch_size 8 \
  --epochs 25_15 \
  --lr 1e-4
```

Change `--gpu` to the CUDA device ID available on your machine. Use `--dataset` to specify the domain order; separate dataset names with underscores.

Training logs, checkpoints, and other run outputs are written under the directory passed to `--save_path`. The trainer evaluates the model during training and after the final domain.

## Evaluation

The training run includes evaluation on the configured domains. Results are reported using heart-rate MAE, RMSE, standard deviation, and Pearson correlation.

## Citation

If you use ApexPhys in your research, please cite the NeurIPS 2026 paper:

> Anchoring Physiological Invariance, Expanding Domain Plasticity: Prior-Stabilized Dynamic Adaptation for Continual rPPG Measurement.

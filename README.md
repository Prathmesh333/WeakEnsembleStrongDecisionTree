# TreeStack-CNN

TreeStack-CNN tests a focused question: can a shallow decision tree combine three small CNNs better than fixed voting rules? Each CNN supplies its full class-probability vector. The tree then learns class-specific and confidence-dependent fusion rules without seeing the original image.

The code is built around one non-negotiable rule. A CNN never predicts a sample for the tree if that sample helped train the CNN. The official training set is divided into a base partition and a held-out meta partition using the configured 60:20 ratio. Ten percent of the base partition is reserved for CNN early stopping. The official test set remains untouched until final evaluation.

This is stacked generalization, not a new ensemble family. The research contribution is narrower: a small, interpretable nonlinear combiner for heterogeneous lightweight CNNs, measured against individual models, majority voting, probability averaging, accuracy-weighted averaging and logistic-regression stacking.

## What is included

- Three compact models: a spatial CNN, a depthwise-separable CNN and a residual CNN.
- Soft, hard and confidence-augmented features for the decision tree.
- Cross-validated tree selection over depth, leaf size and split criterion. The unrestricted tree appears only in the depth ablation.
- Majority, soft, weighted-soft, logistic, random-forest and histogram-gradient-boosting baselines.
- Hard-versus-soft, depth, two-versus-three-model and confidence-feature ablations.
- Accuracy, macro F1, per-class scores, confusion matrices, parameter counts, inference time, tree depth and leaf count.
- Three-seed aggregation with a paper-ready `mean ± standard deviation` accuracy table.
- Cached checkpoints and prediction arrays. A hash of the run settings and model version prevents incompatible artifacts from being mixed.

## Set up the project

Use Python 3.10 or newer. A CUDA-enabled PyTorch install will shorten the CNN training runs, but the code also runs on CPU.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Run the automated checks before downloading either dataset:

```powershell
pytest
```

## Run on Kaggle's T4 x2 accelerator

Open [the Kaggle/Colab notebook](notebooks/treestack_kaggle_colab.ipynb) and select Kaggle's `GPU T4 x2` accelerator. The notebook clones this repository, checks both CUDA devices and runs the full leakage-free pipeline.

Run the notebook cells in order. The activation cell adds the repository's `src` directory to the current kernel and to each child process. It does not require an editable package installation or a kernel restart. If an older notebook reports `No module named 'treestack_cnn'`, open the latest notebook revision and rerun from the activation cell.

Two CNNs train concurrently, one per GPU. When either finishes, the third takes its place. This is independent-model parallelism rather than data parallelism: each network keeps its own optimizer, random seed and checkpoint. Automatic mixed precision reduces T4 memory use and speeds up convolution training.

The ensemble is deliberately heterogeneous. Its members use different convolution types, activation functions, pooling layouts, depths, widths, dropout rates, learning rates, weight decay, label smoothing and initialization seeds. The notebook reports pairwise prediction disagreement, double-fault rates and oracle accuracy to check whether those differences produce complementary errors. Diversity does not prevent overfitting, so every CNN still uses an isolated validation split, early stopping and train-versus-validation convergence checks.

The V1 models underfit Fashion-MNIST: two training accuracies remained below 81%, and the strongest CNN reached 89.4% test accuracy. V2 preserves spatial information, increases model capacity and lowers the batch size from 512 to 128. The default notebook now runs Fashion-MNIST for 40 epochs with seed 42. Treat it as a convergence check. Change the seeds to `17, 42, 73` only after all three base models clear the diagnostic floor.

The exact V1 result table, diagnosis and the role of an optional FMNN baseline are recorded in [the V1 result audit](docs/v1_result_audit.md).

The same runner works from a terminal:

```bash
treestack-cuda --dataset fashion_mnist --seeds 42 --epochs 40 --batch-size 128 --max-gpus 2
```

With one GPU, the runner queues all three CNNs on that device. On a CPU-only machine it stops with a clear error instead of silently running a long experiment.

## Run an experiment

Start with the quick configuration. It trains for only two epochs, so use it to catch environment or data-loading problems rather than to judge the method.

```powershell
treestack --config configs/quick.yaml
```

The full configuration runs both datasets with three seeds. Its 150-epoch ceiling is intended for CIFAR-10; early stopping normally ends Fashion-MNIST much sooner:

```powershell
treestack --config configs/full.yaml
```

Command-line flags can override the YAML file. This is useful for a single debugging run:

```powershell
treestack --config configs/full.yaml --datasets fashion_mnist --seeds 42 --epochs 5
```

Pass `--force` to ignore cached weights and predictions. Without it, an interrupted run resumes from artifacts whose configuration hash matches the current settings.

## Data protocol

The default runner preserves the official torchvision test set. It divides only the official training set into base and meta partitions. This makes Fashion-MNIST and CIFAR-10 results directly comparable with work that follows the standard train/test boundary. Set `use_official_test: false` only to reproduce the legacy V1 custom split.

The partitions have separate jobs:

```text
Official training set
  ├─ 75% base partition
  │  ├─ 90% of base: CNN parameter fitting
  │  └─ 10% of base: CNN early stopping and voting weights
  └─ 25% meta partition: fusion-model fitting

Official test set: final evaluation only
```

Split indices are saved in every run directory. This makes the leakage audit concrete: the stored arrays can be checked for overlap and reused to trace any reported sample.

## Outputs

Each run is written to:

```text
artifacts/<dataset>/seed_<seed>/<configuration-hash>/
```

That folder contains CNN checkpoints, cached probability arrays, fitted scikit-learn combiners, training curves, confusion matrices, a rendered tree, feature importances, the exact split indices and a JSON report. The root artifact directory also receives:

- `all_runs.csv` with one row per method, dataset and seed;
- `aggregate_results.csv` with means and standard deviations;
- `paper_accuracy_table.csv` in the layout used by the planned result table.

## Reading the ablations

`DT-Hard` one-hot encodes each CNN's predicted class. It never feeds numeric class IDs directly to the tree. `DT-Soft` concatenates all probability vectors and is the proposed method. `DT-Enhanced` adds each model's maximum confidence, entropy and pairwise agreement flags.

The primary tree search excludes unlimited depth. Depths 3, 5 and 7 compete through stratified cross-validation on the meta partition. The depth ablation holds each depth fixed while cross-validating the split criterion and leaf size, so its rows are comparable. The unrestricted tree remains an ablation rather than the headline model.

## Method references

- David H. Wolpert, [Stacked Generalization](https://doi.org/10.1016/S0893-6080(05)80023-1), *Neural Networks*, 1992.
- scikit-learn, [StackingClassifier documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingClassifier.html), for the standard out-of-sample stacking principle.

The implementation does not use `StackingClassifier` because the base predictors are trained PyTorch models and their probabilities are cached explicitly. That separation makes the leakage boundary easier to inspect.

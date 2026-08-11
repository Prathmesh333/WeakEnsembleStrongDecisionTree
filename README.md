# TreeStack-CNN

TreeStack-CNN tests a focused question: can a shallow decision tree combine three small CNNs better than fixed voting rules? Each CNN supplies its full class-probability vector. The tree then learns class-specific and confidence-dependent fusion rules without seeing the original image.

The code is built around one non-negotiable rule. A CNN never predicts a sample for the tree if that sample helped train the CNN. The complete Fashion-MNIST or CIFAR-10 dataset is split into 60% base training, 20% meta training and 20% final testing. Ten percent of the base partition is reserved for CNN early stopping. The meta and test partitions remain untouched during CNN fitting.

This is stacked generalization, not a new ensemble family. The research contribution is narrower: a small, interpretable nonlinear combiner for heterogeneous lightweight CNNs, measured against individual models, majority voting, probability averaging, accuracy-weighted averaging and logistic-regression stacking.

## What is included

- Three compact models: a shallow CNN, a wider dropout CNN and a tiny residual CNN.
- Soft, hard and confidence-augmented features for the decision tree.
- Cross-validated tree selection over depth, leaf size and split criterion. The unrestricted tree appears only in the depth ablation.
- Majority, soft, weighted-soft and logistic stacking baselines.
- Hard-versus-soft, depth, two-versus-three-model and confidence-feature ablations.
- Accuracy, macro F1, per-class scores, confusion matrices, parameter counts, inference time, tree depth and leaf count.
- Three-seed aggregation with a paper-ready `mean ± standard deviation` accuracy table.
- Cached checkpoints and prediction arrays. A hash of the run settings prevents artifacts from different configurations from being mixed.

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

The ensemble is deliberately heterogeneous. Its members differ in depth, width, residual connections, batch normalization, dropout, optimizer, learning rate, weight decay, label smoothing and initialization seed. The notebook reports pairwise prediction disagreement, double-fault rates and oracle accuracy to verify that these design differences create useful error diversity. Diversity does not guarantee freedom from overfitting, so every CNN still uses an isolated validation split, early stopping and train-versus-validation convergence checks.

The default notebook settings run Fashion-MNIST for 25 epochs with seed 42. Treat that run as a convergence check. Change the seeds to `17, 42, 73` only after the base models reach credible validation accuracy, then run CIFAR-10 in a separate saved notebook version with a larger epoch budget.

The same runner works from a terminal:

```bash
treestack-cuda --dataset fashion_mnist --seeds 42 --epochs 25 --max-gpus 2
```

With one GPU, the runner queues all three CNNs on that device. On a CPU-only machine it stops with a clear error instead of silently running a long experiment.

## Run an experiment

Start with the quick configuration. It trains for only two epochs, so use it to catch environment or data-loading problems rather than to judge the method.

```powershell
treestack --config configs/quick.yaml
```

The full configuration runs both datasets with three seeds:

```powershell
treestack --config configs/full.yaml
```

Command-line flags can override the YAML file. This is useful for a single debugging run:

```powershell
treestack --config configs/full.yaml --datasets fashion_mnist --seeds 42 --epochs 5
```

Pass `--force` to ignore cached weights and predictions. Without it, an interrupted run resumes from artifacts whose configuration hash matches the current settings.

## Data protocol

The runner joins the official training and test portions of each torchvision dataset, then makes one seeded, class-stratified 60/20/20 split. This gives the exact experimental proportions in the project brief. It also means the resulting numbers should be described as results on a custom stratified split, not compared directly with papers that report accuracy on the official test set.

The partitions have separate jobs:

```text
60% base partition
  ├─ 54% of all samples: CNN parameter fitting
  └─  6% of all samples: CNN early stopping and voting weights

20% meta partition: logistic regression and decision-tree fitting
20% test partition: final evaluation only
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

The primary tree search excludes unlimited depth. Depths 3, 5 and 7 compete through stratified cross-validation on the meta partition. A separate ablation fits those depths plus an unrestricted tree with the same leaf-size setting, which exposes the interpretability-versus-overfitting trade-off without letting an unrestricted model become the headline result.

## Method references

- David H. Wolpert, [Stacked Generalization](https://doi.org/10.1016/S0893-6080(05)80023-1), *Neural Networks*, 1992.
- scikit-learn, [StackingClassifier documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingClassifier.html), for the standard out-of-sample stacking principle.

The implementation does not use `StackingClassifier` because the base predictors are trained PyTorch models and their probabilities are cached explicitly. That separation makes the leakage boundary easier to inspect.

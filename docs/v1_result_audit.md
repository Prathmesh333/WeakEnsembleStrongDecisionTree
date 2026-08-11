# V1 result audit and V2 experiment protocol

## What the first Kaggle run showed

The Fashion-MNIST run used seed 42, 25 epochs and a batch size of 512. Its three CNNs reached 78.06%, 83.64% and 89.40% test accuracy. The decision-tree result did not improve the strongest base model: DT-Soft reached 86.52%, compared with 86.72% for soft voting, 86.99% for weighted soft voting and 89.37% for logistic stacking.

| Method | V1 accuracy |
|---|---:|
| CNN1 | 78.06% |
| CNN2 | 83.64% |
| CNN3 | 89.40% |
| Majority vote | 85.11% |
| Soft vote | 86.72% |
| Weighted soft vote | 86.99% |
| Logistic stack | 89.37% |
| DT-Hard | 75.21% |
| DT-Soft | 86.52% |
| DT-Enhanced | 86.52% |
| DT-Soft, best two CNNs | 87.10% |

This is primarily an underfitting result. At their best validation epochs, CNN1 and CNN2 had training accuracies of only 77.58% and 80.57%. CNN3 reached 89.48%. The models were not memorizing the training data; they had insufficient representation capacity and received too few optimizer updates because of the batch size of 512. Global average pooling also discarded most spatial information before classification.

The ensemble had some useful diversity. Its oracle accuracy was 92.44%, and DT-Soft corrected 849 majority-vote errors while introducing 651 new errors. The problem was that the first two models were too weak, so combining all three often diluted CNN3. The one-seed run and the legacy custom split also make the values unsuitable as publication evidence.

## What V2 changes

V2 uses three stronger but structurally different networks: a spatial CNN with a 4x4 feature map, a depthwise-separable CNN and a deeper residual CNN. They use different activation functions, widths, pooling paths, regularization settings, learning rates and initialization seeds. The default batch size is 128, with 40 Fashion-MNIST epochs and early stopping.

The default protocol now preserves the official test set. For Fashion-MNIST, 45,000 official training images form the base partition; 40,500 train the CNNs and 4,500 control early stopping. The remaining 15,000 official training images fit the fusion models. The original 10,000 test images are evaluated only after training is complete.

Random Forest and histogram-gradient-boosting stackers are included as nonlinear baselines on the same probability features. The depth ablation now fixes only depth while cross-validating criterion and leaf size, avoiding the confounded comparison in V1.

## Where FMNN fits

A fuzzy min-max neural network is a possible fusion ablation, not the first fix for these results. It would receive the same held-out CNN probability vectors as DT-Soft, Random Forest and HGB. Testing it before the CNNs clear the accuracy floor would not answer whether the fusion method is effective, because the experiment would still be dominated by weak base features.

Run V2 with seed 42 first. Continue to the three-seed experiment only if every Fashion-MNIST CNN reaches at least 90% and the learned fusion method is competitive with the strongest CNN and logistic stacking. If those checks pass, FMNN can be added as another meta-classifier under the identical split; if they fail, inspect convergence and pairwise errors before changing the fusion layer again.

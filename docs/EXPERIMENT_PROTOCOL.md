# Experiment protocol

## Claim under test

A shallow decision tree trained on out-of-sample probability predictions from diverse lightweight CNNs can learn class-specific fusion rules. The experiment asks whether those rules beat the strongest individual CNN and static voting while adding little inference cost.

The study does not claim that stacking is new. Its contribution is the use and analysis of an interpretable tree combiner for a small CNN ensemble.

## Research questions

1. Does decision-tree fusion outperform the strongest individual lightweight CNN?
2. Does it outperform majority voting and soft-probability averaging?
3. Are full probability vectors more useful than hard predicted labels?
4. How does tree depth change accuracy, tree size and interpretability?
5. Does the third CNN add complementary information?

## Leakage and selection controls

The default protocol preserves each torchvision official test set. It splits only the official training set in a class-stratified 75:25 ratio between the CNN base partition and the meta partition. CNN fitting and early stopping occur entirely inside the base partition. The frozen CNNs generate predictions for the meta partition, and those predictions train the fusion models. Official test labels are read only by reporting code after every method is frozen.

V4 further splits the meta partition into 80% search and 20% audit samples. All evolutionary candidates compete on search data. The single search winner is frozen and evaluated once on audit data. Audit labels cannot select, mutate, gate or otherwise modify the genome. The audit result is diagnostic evidence, not an extra hyperparameter search.

The target label is never a tree feature. For a ten-class dataset, the proposed three-CNN input has 30 values:

```text
[CNN-1 probability vector | CNN-2 probability vector | CNN-3 probability vector]
```

CNN class predictions are one-hot encoded in the hard-label ablation. Treating labels such as 2, 8 and 4 as numeric values would introduce a false ordering between classes.

## Comparisons

The main table reports the three CNNs, majority vote, soft vote, accuracy-weighted soft vote, logistic stacking, DT-Hard and DT-Soft. DT-Enhanced and DT-Soft with the best two CNNs answer the confidence-feature and model-count questions.

Voting weights come from the base-validation subset rather than the final test set. Tree hyperparameters are selected by stratified cross-validation inside the meta partition. The search covers depths 3, 5 and 7; minimum leaf sizes 5, 10 and 20; and Gini or entropy splits.

## Measurements

Report test accuracy and macro F1 as mean ± sample standard deviation over seeds 17, 42 and 73. Include bootstrap 95% confidence intervals. Use exact paired McNemar tests for V4 versus soft voting, logistic stacking, random-forest stacking, DT-Soft and the strongest CNN; report correction and harm counts with Holm-adjusted p-values. Keep the per-class precision, recall and F1 values in the supplementary results. Each run also records a confusion matrix, trainable CNN parameter counts, prediction time, tree depth, leaf count and split-node count.

The disagreement report separates samples into three groups: all CNNs agree, exactly two agree and all three disagree. It also counts cases where DT-Soft corrects majority voting, damages a correct majority decision or fails despite at least one correct base prediction. Those cases are more informative than a single accuracy difference.

## Result interpretation

Support for the hypothesis requires more than DT-Soft winning one run. The mean should exceed the strongest individual CNN and both unweighted voting baselines, with the seed-level variance reported beside it. If logistic stacking matches or beats the tree, the evidence favors a linear combiner. If an unrestricted tree scores well only on the meta set or varies sharply across seeds, that is evidence of overfitting rather than a stronger method.

Negative results remain useful. Highly correlated CNN errors may leave no exploitable signal for any combiner. The stored predictions allow pairwise error agreement and class-specific behavior to be examined without retraining the networks.

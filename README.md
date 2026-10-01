# GRLV — code accompanying the manuscript

This directory contains the complete code used for the experiments reported in
*"GRLV: Gradient Reversal Layer with Variance Loss for Major Depressive Disorder
Detection"*.  Everything here follows the protocol described in the manuscript
(Sections 3.4 and 4.4) and in the response to the reviewers; nothing that was
only used for exploratory analysis is included.

## Pipeline

```
dependences.py   imports, global settings and set_seed()
tools.py         GradReverseLayer, straight_model_with_grl, sample construction
                 (three facial conditions -> 825 x 256), the domain-label
                 clustering, and the metric functions
data_models.py   TVTLoader: the 90 subjects, the six subject-level groups and
                 the train / validation / test split of one fold
pl_models.py     BaselineModule and GRLModule (Lightning); the three-loss
                 rotation and the lambda ramp of Equation (11)
best_params.py   the fixed hyper-parameters of Tables 2 and 3
domains.py       builds and stores the fixed domain labels
optuna.py        nested cross-validation hyper-parameter search (Optuna/TPE)
evaluate.py      six-fold evaluation with fixed hyper-parameters
plot.py          t-SNE of the learned features with ARI/AMI, Wasserstein distance
run.sh           launches the stages above on several GPUs
```

## Order of execution

```bash
bash run.sh domains                     # writes domains.npz (fixed domain labels)
bash run.sh optuna                      # lr -> clf / var, per backbone
bash run.sh evaluate                    # six-fold evaluation, all conditions
bash run.sh plot                        # t-SNE and Wasserstein figures
```

`optuna.py` is run once per (backbone, search group): the learning rate is
searched under the baseline condition, `(grl_gamma, max_grl_lambd)` under the
condition in which only the conventional GRL is active, and
`(std_gamma, max_std_lambd)` under the condition in which only the variance loss
is active.  The learning rate of the last two groups is the value found in the
baseline condition.

## Notation

The symbols follow the manuscript: `gamma_clf` and `lambda_max,clf` scale the
gradient reversal applied to the domain-classification loss, `gamma_var` and
`lambda_max,var` scale the variance loss; `K` is the number of domains (K = 3).

## Environment

```
python >= 3.10
torch
braindecode
optuna
scikit-learn
scipy
numpy
matplotlib
```

See `requirements.txt` for the versions used for the reported results.

## Data

The MEG data are not publicly redistributable.  The pipeline expects a file
`dataset_and_info_256.pt`, a `(TensorDataset, Munch)` pair in which the dataset
holds `(inputs, labels, subjects, face_categories)` with `inputs` of shape
`(n_trials, 1, 275, 256)`.  Its location and the output directory are set with
the environment variables `GRLV_DATA_FILE` and `GRLV_OUT_DIR` (defaults:
`data/dataset_and_info_256.pt` and `output/`); no machine-specific path is
hard-coded anywhere in the code.

```bash
export GRLV_DATA_FILE=/path/to/dataset_and_info_256.pt
export GRLV_OUT_DIR=./output
bash run.sh domains
```

'''嵌套交叉验证超参搜索（论文 4.4，回复 Comment 18）。

流程：
1. 90 名被试按类别均分为 6 组，外层即 6 折嵌套交叉验证；
2. 每个外层折内，4 组训练、1 组验证，对 4 个验证组分别搜索；
3. 每个验证组内用 TPE 采样，前 5 个 trial 随机，其余由模型给出；
4. 4 个内层折各给出一组最优超参，这 4 组超参再在 4 个验证集上评估，
   得到 4×4 个准确率；同一组超参的 4 个分数取平均；
5. 取平均后最好的 2 组超参取平均，作为该外层折的搜索结果；
6. 六个外层折的结果取中位数，即最终超参。

用法：
    python optuna.py --model EEGNet --group lr  --gpu 0
    python optuna.py --model EEGNet --group clf --gpu 1
    python optuna.py --model EEGNet --group var --gpu 2
'''

import argparse
import json
import os

from tools import *
from best_params import *
from data_models import TVTLoader
from pl_models import GRLModule, BaselineModule


GROUP_KEYS = {
    'lr': (['label_opt_lr'], 'baseline'),
    'clf': (['grl_gamma', 'max_grl_lambd'], 'no_subj_grl_var'),
    'var': (['std_gamma', 'max_std_lambd'], 'no_domain_grl_bce'),
}


class SlopePruning(L.Callback):
    '''回复 Comment 18-5 的剪枝规则：
    从第 PRUNE_AFTER 个 iteration 起，若最近 PRUNE_WINDOW 个 iteration 的
    验证准确率拟合直线斜率为负，或训练准确率达到 0.99，则提前终止该 trial。'''

    def __init__(self, trial: optuna.Trial):
        super().__init__()
        self.trial = trial
        self.values = []

    def on_validation_epoch_end(self, trainer, pl_module):
        value = trainer.callback_metrics.get('valid_acc')
        if value is None:
            return
        self.values.append(float(value))
        if len(self.values) <= PRUNE_AFTER + PRUNE_WINDOW:
            return
        window = self.values[-PRUNE_WINDOW:]
        slope = np.polyfit(np.arange(len(window)), window, 1)[0]
        self.trial.report(float(value), step=len(self.values))
        if slope < 0 or self.trial.should_prune():
            raise optuna.TrialPruned()


def trainer_fit(module, datamodule, trial, max_epochs=MAX_EPOCHS):
    '''训练一个 trial，返回训练好的模块与验证准确率。'''
    callbacks = [SlopePruning(trial)] if trial is not None else []
    trainer = L.Trainer(max_epochs=max_epochs, precision='bf16-mixed',
                        callbacks=callbacks, devices=1,
                        enable_progress_bar=False, logger=False,
                        enable_model_summary=False, num_sanity_val_steps=0)
    trainer.fit(module, datamodule=datamodule)
    return trainer


def validation_accuracy(trainer):
    value = trainer.callback_metrics.get('valid_acc')
    return float(value) if value is not None else float('nan')


def run_trial(backbone, params, ablation, fold, valid_group, hp_fixed, epochs,
              n_concat=N_CONCAT):
    datamodule = TVTLoader(fold=fold, valid_group=valid_group, condition=ablation,
                           n_concat=n_concat)
    hp = dict(hp_fixed)
    hp.update(params)
    hp['model'] = backbone
    hp['ablation'] = ablation
    module = BaselineModule if ablation == 'baseline' else GRLModule
    model = module(None, hp, )
    trainer = trainer_fit(model, datamodule, trial=None, max_epochs=epochs)
    return validation_accuracy(trainer)


def objective_factory(backbone, keys, ablation, fold, valid_group, hp_fixed,
                      epochs, n_concat=N_CONCAT):
    def objective(trial: optuna.Trial):
        params = {}
        for key in keys:
            low, high, log = SEARCH_SPACE[key]
            params[key] = trial.suggest_float(key, low, high, log=log)
        datamodule = TVTLoader(fold=fold, valid_group=valid_group,
                               condition=ablation, n_concat=n_concat)
        hp = dict(hp_fixed); hp.update(params)
        hp['model'] = backbone; hp['ablation'] = ablation
        module = BaselineModule if ablation == 'baseline' else GRLModule
        model = module(trial, hp)
        trainer = trainer_fit(model, datamodule, trial, max_epochs=epochs)
        return validation_accuracy(trainer)
    return objective


def search_outer_fold(backbone, keys, ablation, fold, hp_fixed, args):
    others = [i for i in range(N_OUTER_FOLDS) if i != fold]
    rng = np.random.RandomState(SEED + fold)
    valid_groups = list(rng.choice(others, size=N_INNER_VALID, replace=False))

    best_per_inner = []
    for vg in valid_groups:
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=SEED + fold,
                                              n_startup_trials=N_STARTUP_TRIALS))
        study.optimize(objective_factory(backbone, keys, ablation, fold, vg,
                                         hp_fixed, args.epochs, args.n_concat),
                       n_trials=args.trials)
        best_per_inner.append(dict(study.best_params))
        print('  fold %d / validation group %d : best val acc %.4f'
              % (fold, vg, study.best_value), flush=True)

    # 4 组超参 × 4 个验证集
    scores = []
    for cand in best_per_inner:
        accs = [run_trial(backbone, cand, ablation, fold, vg, hp_fixed, args.epochs,
                          args.n_concat)
                for vg in valid_groups]
        scores.append((float(np.mean(accs)), cand))
    scores.sort(key=lambda t: -t[0])
    top = scores[:TOP_N_AVERAGE]
    print('  fold %d top-%d mean validation acc: %s'
          % (fold, TOP_N_AVERAGE, [round(s, 4) for s, _ in top]), flush=True)
    return {k: float(np.mean([c[k] for _, c in top])) for k in keys}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, choices=MODELS)
    parser.add_argument('--group', required=True, choices=list(GROUP_KEYS))
    parser.add_argument('--trials', type=int, default=N_TRIALS)
    parser.add_argument('--epochs', type=int, default=MAX_EPOCHS)
    parser.add_argument('--gpu', default=None, help='由 run.sh 传入')
    parser.add_argument('--max-folds', type=int, default=N_OUTER_FOLDS,
                        help='调试用：只跑前若干个外层折')
    parser.add_argument('--n-concat', type=int, default=N_CONCAT,
                        help='调试用：每个被试构造的样本数')
    args = parser.parse_args()

    set_seed(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)
    keys, ablation = GROUP_KEYS[args.group]

    hp_fixed = dict(best_param_dict['grl'])
    hp_fixed.update(dict(best_param_dict[args.model]))
    if args.group != 'lr':
        # 学习率在 baseline 条件下单独搜索，之后固定（回复 Comment 18-2）
        lr_file = os.path.join(OUT_DIR, 'search_%s_lr.json' % args.model)
        if not os.path.exists(lr_file):
            raise FileNotFoundError('先运行 --group lr：%s' % lr_file)
        hp_fixed['label_opt_lr'] = json.load(open(lr_file))['median']['label_opt_lr']
        print('learning rate fixed to %.4g' % hp_fixed['label_opt_lr'])

    results = []
    for fold in range(min(args.max_folds, N_OUTER_FOLDS)):
        print('=== %s / %s : outer fold %d ===' % (args.model, args.group, fold),
              flush=True)
        results.append(search_outer_fold(args.model, keys, ablation, fold,
                                         hp_fixed, args))

    median = {k: float(np.median([r[k] for r in results])) for k in keys}
    record = {'model': args.model, 'group': args.group, 'ablation': ablation,
              'search_keys': keys, 'per_fold': results, 'median': median,
              'trials': args.trials, 'epochs': args.epochs, 'seed': SEED}
    path = os.path.join(OUT_DIR, 'search_%s_%s.json' % (args.model, args.group))
    json.dump(record, open(path, 'w'), indent=2)
    print('median over the six outer folds:', median)
    print('wrote', path)


if __name__ == '__main__':
    main()

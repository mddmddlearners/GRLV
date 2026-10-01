'''固定超参下的六折评估（回复 Comment 18-6、19 与论文 5.1）。

超参来自 optuna.py 的搜索结果并固定不变。六个外层折依次作为测试集，
每次在其余 5 组上训练（不使用验证集），再在该折上测试；每名被试恰好被
测试一次。报告 acc、AUC、F1、precision、recall、specificity 的均值与
标准差，以及逐被试准确率的方差（论文 5.1 的异质性指标）。

    python evaluate.py --model EEGNet --condition full  --gpu 0
    python evaluate.py --model EEGNet --condition baseline --gpu 1
'''

import argparse
import json
import os

from tools import *
from best_params import *
from data_models import TVTLoader
from pl_models import GRLModule, BaselineModule


def load_searched(model: str):
    '''读取六个外层折的中位数作为固定超参。'''
    out = {}
    for group in ('lr', 'clf', 'var'):
        path = os.path.join(OUT_DIR, 'search_%s_%s.json' % (model, group))
        if not os.path.exists(path):
            print('[warn] %s 不存在，暂时使用表 2/表 3 的固定值' % path)
            continue
        out.update(json.load(open(path))['median'])
    return out


def hp_for(model: str, condition: str):
    hp = dict(best_param_dict['grl'])
    hp.update(dict(best_param_dict[model]))
    hp.update(load_searched(model))
    hp['model'] = model
    hp['ablation'] = condition
    return hp


def collect_predictions(trainer, model, datamodule):
    '''返回 (labels, probs, subjects)。'''
    loader = datamodule.test_dataloader()
    labels, probs, subjects = [], [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            input, y, s, _, _ = batch
            hedp_y, _, _ = model(input)
            labels.append(y.numpy())
            probs.append(torch.softmax(hedp_y, dim=1).numpy())
            subjects.append(s.numpy())
    return (np.concatenate(labels), np.concatenate(probs),
            np.concatenate(subjects))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, choices=MODELS)
    parser.add_argument('--condition', default='full', choices=ABLATIONS)
    parser.add_argument('--epochs', type=int, default=MAX_EPOCHS)
    parser.add_argument('--gpu', default=None)
    parser.add_argument('--max-folds', type=int, default=N_OUTER_FOLDS)
    parser.add_argument('--n-concat', type=int, default=N_CONCAT,
                        help='调试用：每个被试构造的样本数')
    args = parser.parse_args()

    set_seed(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)
    hp = hp_for(args.model, args.condition)
    print('hyper-parameters:', hp)

    folds, per_subject = [], {}
    for fold in range(min(args.max_folds, N_OUTER_FOLDS)):
        datamodule = TVTLoader(fold=fold, valid_group=None,
                               condition=args.condition, n_concat=args.n_concat)
        module = BaselineModule if args.condition == 'baseline' else GRLModule
        model = module(None, hp)
        trainer = L.Trainer(max_epochs=args.epochs, precision='bf16-mixed',
                            devices=1, enable_progress_bar=False, logger=False,
                            enable_model_summary=False, num_sanity_val_steps=0)
        trainer.fit(model, datamodule=datamodule)
        labels, probs, subjects = collect_predictions(trainer, model, datamodule)
        metrics = binary_metrics(labels, probs)
        metrics['fold'] = fold
        folds.append(metrics)
        per_subject.update(per_subject_metrics(labels, probs, subjects))
        print('fold %d  acc=%.4f  auc=%.4f  f1=%.4f'
              % (fold, metrics['acc'], metrics['auc'], metrics['f1score']),
              flush=True)

    summary = {}
    for key in ('acc', 'auc', 'f1score', 'precision', 'recall', 'specificity'):
        values = np.array([f[key] for f in folds], dtype=float)
        summary[key] = {'mean': float(np.nanmean(values)),
                        'std': float(np.nanstd(values, ddof=1)),
                        'per_fold': [float(v) for v in values]}
    acc = np.array(list(per_subject.values()))
    summary['per_subject_accuracy'] = {
        'mean': float(acc.mean()), 'std': float(acc.std(ddof=1)),
        'variance': float(acc.var(ddof=1)),
        'values': {str(k): v for k, v in per_subject.items()}}

    record = {'model': args.model, 'condition': args.condition,
              'hyperparameters': dict(hp), 'epochs': args.epochs,
              'n_folds': N_OUTER_FOLDS, 'summary': summary}
    path = os.path.join(OUT_DIR, 'eval_%s_%s.json' % (args.model, args.condition))
    json.dump(record, open(path, 'w'), indent=2)
    print('accuracy %.4f +- %.4f ; per-subject accuracy variance %.4f'
          % (summary['acc']['mean'], summary['acc']['std'],
             summary['per_subject_accuracy']['variance']))
    print('wrote', path)


if __name__ == '__main__':
    main()

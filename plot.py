'''可视化：图 5 的 t-SNE、图 6 的 Wasserstein 距离。

t-SNE 设定（回复 Comment 20）：特征为 Gf 的 50 维输出，不做任何额外降维或
标准化；sklearn t-SNE，n_components=2、perplexity=30、init='pca'、欧氏距离、
learning rate='auto'、early exaggeration=12、1000 次迭代、固定随机种子。
所有骨干、所有条件使用同一套设定。

每个子图标题只给 ARI 与 AMI：在 **50 维原始特征** 上做 KMeans(k=3) 后与域
标签比较，因此不依赖嵌入结果。颜色表示三个域，x 表示 MDD、o 表示 HC。

    python plot.py --model EEGNet --gpu 0
'''

import argparse
import json
import os

from tools import *
from best_params import *
from data_models import TVTLoader
from pl_models import GRLModule, BaselineModule


def features_of(model_name, condition, fold=0, epochs=MAX_EPOCHS,
                n_concat=N_CONCAT):
    '''训练一次并用留出折的特征作图。'''
    hp = dict(best_param_dict['grl'])
    hp.update(dict(best_param_dict[model_name]))
    for group in ('lr', 'clf', 'var'):
        path = os.path.join(OUT_DIR, 'search_%s_%s.json' % (model_name, group))
        if os.path.exists(path):
            hp.update(json.load(open(path))['median'])
    hp['model'] = model_name
    hp['ablation'] = condition

    datamodule = TVTLoader(fold=fold, valid_group=None, condition=condition,
                           n_concat=n_concat)
    module = BaselineModule if condition == 'baseline' else GRLModule
    model = module(None, hp)
    trainer = L.Trainer(max_epochs=epochs, precision='bf16-mixed', devices=1,
                        enable_progress_bar=False, logger=False,
                        enable_model_summary=False, num_sanity_val_steps=0)
    trainer.fit(model, datamodule=datamodule)

    model.eval()
    feats, labels, subjects = [], [], []
    with torch.no_grad():
        for batch in datamodule.test_dataloader():
            input, y, s, _, _ = batch
            f = backbone_features(model, input)
            feats.append(f.numpy()); labels.append(y.numpy()); subjects.append(s.numpy())
    return (np.concatenate(feats), np.concatenate(labels),
            np.concatenate(subjects))


def tsne_figure(feats, labels, subjects, domain_of, save_to, title=''):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from sklearn.cluster import KMeans
    from sklearn.manifold import TSNE
    from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

    domains = np.array([domain_of[int(s)] for s in subjects])
    cluster = KMeans(K_DOMAINS, n_init=50, random_state=SEED).fit_predict(feats)
    ari = adjusted_rand_score(domains, cluster)
    ami = adjusted_mutual_info_score(domains, cluster)
    embedding = TSNE(**TSNE_SETTINGS).fit_transform(feats)

    fig, ax = plt.subplots(figsize=(6.4, 5.8))
    for d in range(K_DOMAINS):
        colour = DOMAIN_COLORS[d]
        m = domains == d
        ax.scatter(embedding[m & (labels == 1), 0], embedding[m & (labels == 1), 1],
                   marker='x', s=42, color=colour, alpha=0.85)
        ax.scatter(embedding[m & (labels == 0), 0], embedding[m & (labels == 0), 1],
                   marker='o', s=42, facecolors='none', edgecolors=colour,
                   alpha=0.9, linewidths=1.2)
    ax.legend(handles=[Line2D([], [], marker='x', color='k', linestyle='None',
                              markersize=8, label='MDD'),
                       Line2D([], [], marker='o', markerfacecolor='none',
                              markeredgecolor='k', linestyle='None', markersize=8,
                              label='HC')], fontsize=9)
    ax.set_title('ARI=%.3f  AMI=%.3f' % (ari, ami), fontsize=12)
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout(); fig.savefig(save_to, dpi=140); plt.close(fig)
    return {'ari': float(ari), 'ami': float(ami)}


def wasserstein_distance(feats, subjects):
    '''不同被试特征分布之间的平均 Wasserstein 距离（图 6）。'''
    from scipy.stats import wasserstein_distance
    subs = np.unique(subjects)
    dists = []
    for i, a in enumerate(subs):
        for b in subs[i + 1:]:
            fa, fb = feats[subjects == a], feats[subjects == b]
            dists.append(np.mean([wasserstein_distance(fa[:, k], fb[:, k])
                                  for k in range(feats.shape[1])]))
    return float(np.mean(dists))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, choices=MODELS)
    parser.add_argument('--epochs', type=int, default=MAX_EPOCHS)
    parser.add_argument('--gpu', default=None)
    parser.add_argument('--n-concat', type=int, default=N_CONCAT,
                        help='调试用：每个被试构造的样本数')
    args = parser.parse_args()

    set_seed(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)
    domains = np.load(DOMAIN_FILE)
    domain_of = {int(s): int(d) for s, d in zip(domains['subjects'],
                                                domains['domains'])}
    record = {'model': args.model, 'tsne': TSNE_SETTINGS, 'results': {}}
    for condition in ('baseline', 'full'):
        feats, labels, subjects = features_of(args.model, condition,
                                              epochs=args.epochs,
                                              n_concat=args.n_concat)
        png = os.path.join(OUT_DIR, 'tsne_%s_%s.png' % (args.model, condition))
        scores = tsne_figure(feats, labels, subjects, domain_of, png)
        scores['wasserstein'] = wasserstein_distance(feats, subjects)
        record['results'][condition] = scores
        print('%s / %s : ARI=%.3f  AMI=%.3f  Wasserstein=%.4f'
              % (args.model, condition, scores['ari'], scores['ami'],
                 scores['wasserstein']), flush=True)
    path = os.path.join(OUT_DIR, 'plot_%s.json' % args.model)
    json.dump(record, open(path, 'w'), indent=2)
    print('wrote', path)


if __name__ == '__main__':
    main()

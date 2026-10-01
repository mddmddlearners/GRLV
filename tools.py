from dependences import *


# ======================================================================
# 梯度反转层（论文公式 2、3）
# ======================================================================
class GradReverseLayer(torch.autograd.Function):
    '''
    通过这一层的数据正向传递不变；反向传播时梯度乘以 lambd。
    lambd < 0 时梯度被反转，用于域分类器的对抗训练。
    '''

    @staticmethod
    def forward(ctx, x, lambd, **kwargs):
        ctx.lambd = lambd
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.lambd, None


class straight_model_with_grl(nn.Module):
    '''
    把任意骨干网络包装成 GRLV 的形式（论文图 4）。

    输入形如 (b, 1, c, t) 或 (b, c, t)，骨干网络输出 50 维特征 f，之后：
        hedp_layer : f        -> 疾病预测 (b, 2)
        subj_grl_layers : GRL(f) -> 个体预测 (b, N_SUBJS)
        domain_layer    : 个体预测 -> 域预测 (b, K_DOMAINS)

    detach=True 时个体分类支路的梯度被截断（stop-gradient，论文 3.4）。
    '''

    def __init__(self, model: nn.Module, n_inputs: int = 50,
                 n_subjs: int = N_SUBJS, n_domains: int = K_DOMAINS,
                 output: str = 'all'):
        super().__init__()
        self.model = model
        self.output = output
        self.default_lambd = 1.0

        self.subj_grl_layers = nn.Sequential(
            nn.Linear(n_inputs, 300),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(300, n_subjs),
        )
        self.domain_layer = nn.Linear(n_subjs, n_domains)
        self.hedp_layer = nn.Sequential(
            nn.Linear(n_inputs, 300),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(300, 30),
            nn.ReLU(),
            nn.Linear(30, 2),
        )

    def forward(self, x: torch.Tensor, lambd: float = 1.0, detach: bool = False):
        f = self.model(x)
        return self.forward_from_feature(f, lambd=lambd, detach=detach)

    def forward_from_feature(self, f: torch.Tensor, lambd: float = None,
                             detach: bool = False):
        if lambd is None:
            lambd = self.default_lambd
        hedp_y = self.hedp_layer(f)
        grl_x = GradReverseLayer.apply(f, lambd)
        subj_x = grl_x.detach() if detach else grl_x
        subj_y = self.subj_grl_layers(subj_x)
        domain_y = self.domain_layer(subj_y)
        logits = {'hedp': hedp_y, 'subj': subj_y, 'domain': domain_y}
        if self.output == 'all':
            return hedp_y, subj_y, domain_y
        if isinstance(self.output, list):
            return [logits[key] for key in self.output]
        return logits[self.output]


def build_backbone(name: str):
    '''十个骨干网络的实例化方式（论文表 4）。'''
    bm_args = {'n_outputs': 50, 'n_chans': N_CHANS,
               'n_times': N_TIMES, 'sfreq': SFREQ}
    modellist = {
        'AttentionBaseNet': lambda: bm.AttentionBaseNet(**bm_args),
        'EEGMiner': lambda: bm.EEGMiner(**bm_args),
        'EEGNet': lambda: bm.EEGNet(**bm_args),
        'BrainModule': lambda: bm.BrainModule(**bm_args),
        'EEGConformer': lambda: bm.EEGConformer(**bm_args),
        # BIOT 默认 embed_dim=2048 无法在显存内运行，论文使用缩小配置
        'BIOT': lambda: bm.BIOT(**bm_args, embed_dim=32, num_heads=2, num_layers=1),
        'FBCNet': lambda: bm.FBCNet(**bm_args),
        'IFNet': lambda: bm.IFNet(**bm_args),
        'MSVTNet': lambda: bm.MSVTNet(**bm_args),
        'SSTDPN': lambda: bm.SSTDPN(**bm_args),
    }
    return modellist[name]()


def backbone_features(lightning_module, x):
    '''返回 Gf 的 50 维输出（论文图 5 中作图的 f）。'''
    return lightning_module.model.model(x)


# ======================================================================
# 样本构造（论文 4.3）
# ======================================================================
def concat_pics(inputs, labels, subjs, pics, use_subjs, n_concat=N_CONCAT, seed=SEED):
    '''
    将同一被试三种面孔的试次各随机取 n_concat 个，按“中性、悲伤、高兴”拼接：
    (n_trial, 1, 275, 256) -> (n_concat, 825, 256)。
    '''
    rng = np.random.RandomState(seed)
    xs, ys, ss = [], [], []
    for s in use_subjs:
        idx = np.nonzero(subjs == int(s))[0]
        xi, li, pi = inputs[idx], labels[idx], pics[idx]
        faces = []
        for f in range(N_FACES):
            pool = np.nonzero(pi == f)[0]
            faces.append(xi[pool[rng.randint(0, len(pool), n_concat)]])
        cat = torch.cat(faces, dim=1).reshape(n_concat, N_CHANS, N_TIMES)
        xs.append(cat)
        ys.append(torch.full((n_concat,), int(li[0]), dtype=torch.long))
        ss.append(torch.full((n_concat,), int(s), dtype=torch.long))
    return torch.cat(xs).float(), torch.cat(ys), torch.cat(ss)


# ======================================================================
# 域标签：被试级连接的对数欧氏表示的聚类（论文 3.4）
# ======================================================================
def _log_euclidean(cov):
    w, v = np.linalg.eigh(cov)
    return v @ np.diag(np.log(np.clip(w, 1e-10, None))) @ v.T


def _correlation(x):
    x = x - x.mean(axis=1, keepdims=True)
    x = x / (x.std(axis=1, keepdims=True) + 1e-12)
    return x @ x.T / x.shape[1]


def subject_connectivity(inputs, subjects):
    '''每个被试一个对数欧氏切空间向量。'''
    feats, index = [], []
    for s in np.unique(subjects):
        rows = np.nonzero(subjects == s)[0]
        x = inputs[rows, 0].float().numpy()
        cat = x.transpose(1, 0, 2).reshape(x.shape[1], -1)
        feats.append(_log_euclidean(_correlation(cat)).reshape(-1))
        index.append(int(s))
    return np.asarray(feats, dtype=np.float64), np.asarray(index)


def build_domains(seed=SEED, verbose=True):
    '''
    在 k = 2..10 上聚类，用轮廓系数与自举稳定性确定 k，
    使用全部被试参与聚类，不做离群剔除。
    返回 (被试编号, 固定域标签, 全部为 True 的保留标记, 各 k 的指标)。
    '''
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.metrics import adjusted_rand_score, silhouette_score
    from sklearn.preprocessing import QuantileTransformer

    dataset = torch.load(DATA_FILE, weights_only=False)[0]
    inputs, _, subjects, _ = dataset.tensors
    features, index = subject_connectivity(inputs, subjects.numpy())
    x = QuantileTransformer(output_distribution='normal',
                            n_quantiles=min(100, len(features)),
                            random_state=seed).fit_transform(features)
    points = PCA(n_components=20, whiten=True, random_state=seed).fit_transform(x)

    # 全部被试参与聚类
    kept = np.ones(len(points), dtype=bool)

    metrics = {}
    for k in range(2, 11):
        labels = KMeans(k, n_init=20, random_state=seed).fit_predict(points[kept])
        rng = np.random.RandomState(seed)
        aris = []
        for _ in range(30):
            idx = rng.choice(int(kept.sum()), size=int(0.8 * kept.sum()), replace=False)
            sub = KMeans(k, n_init=5, random_state=seed).fit_predict(points[kept][idx])
            aris.append(adjusted_rand_score(labels[idx], sub))
        metrics[k] = {'silhouette': float(silhouette_score(points[kept], labels)),
                      'stability': float(np.mean(aris))}
        if verbose:
            print('k=%2d  silhouette=%.4f  stability=%.3f'
                  % (k, metrics[k]['silhouette'], metrics[k]['stability']))

    model = KMeans(K_DOMAINS, n_init=50, random_state=seed).fit(points[kept])
    labels = model.predict(points)          # 离群被试按最近簇心指派
    return index, labels, kept, metrics


# ======================================================================
# 指标
# ======================================================================
def binary_metrics(labels, probs):
    '''准确率、AUC、F1、精确率、召回率、特异度（论文 5.1）。'''
    from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                                 recall_score, roc_auc_score)
    pred = probs.argmax(1)
    tn = int(((pred == 0) & (labels == 0)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    out = {'acc': float(accuracy_score(labels, pred)),
           'f1score': float(f1_score(labels, pred, zero_division=0)),
           'precision': float(precision_score(labels, pred, zero_division=0)),
           'recall': float(recall_score(labels, pred, zero_division=0)),
           'specificity': float(tn / (tn + fp)) if (tn + fp) else 0.0}
    try:
        out['auc'] = float(roc_auc_score(labels, probs[:, 1]))
    except ValueError:
        out['auc'] = float('nan')
    return out


def per_subject_metrics(labels, probs, subjects):
    '''逐被试准确率；其方差即论文 5.1 的异质性指标。'''
    pred = probs.argmax(1)
    out = {}
    for s in np.unique(subjects):
        m = subjects == s
        out[int(s)] = float((pred[m] == labels[m]).mean())
    return out

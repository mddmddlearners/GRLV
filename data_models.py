from tools import *


def study_subjects(seed=SEED):
    '''45 名 MDD + 45 名 HC（论文 4.4）。'''
    dataset, info = torch.load(DATA_FILE, weights_only=False)
    rng = np.random.RandomState(seed)
    dp = np.array([int(s) for s in info.dp_subjs]); rng.shuffle(dp)
    he = np.array([int(s) for s in info.he_subjs]); rng.shuffle(he)
    return [int(s) for s in dp[:N_MDD]], [int(s) for s in he[:N_HC]]


def six_groups(dp, he):
    '''按类别各自均分 6 组，每组 7–8 名 MDD 与 7–8 名 HC（论文 4.4）。'''
    dp_folds = np.array_split(np.asarray(dp), N_OUTER_FOLDS)
    he_folds = np.array_split(np.asarray(he), N_OUTER_FOLDS)
    return [{'dp': [int(s) for s in dp_folds[i]],
             'he': [int(s) for s in he_folds[i]]} for i in range(N_OUTER_FOLDS)]


def load_domains():
    '''读取 tools.build_domains 固化的域标签。'''
    data = np.load(DOMAIN_FILE)
    return {int(s): int(d) for s, d in zip(data['subjects'], data['domains'])}


class TVTLoader(L.LightningDataModule):
    '''
    嵌套交叉验证的数据划分。

    fold        : 外层折编号（该折的组作为测试集，其余 5 组用于训练）
    valid_group : 内层验证组编号；None 表示评估阶段（只训练与测试，无验证集）
    condition   : full / no_subj_grl_var / no_domain_grl_bce / baseline
    '''

    def __init__(self, fold: int = 0, valid_group: Optional[int] = None,
                 condition: str = 'full', n_concat: int = N_CONCAT,
                 batch_size: int = BATCH_SIZE, seed: int = SEED):
        super().__init__()
        self.dataset, self.info = torch.load(DATA_FILE, weights_only=False)
        self.inputs, self.labels, self.subjects, self.pics = self.dataset.tensors
        self.fold = fold
        self.valid_group = valid_group
        self.condition = condition
        self.n_concat = n_concat
        self.batch_size = batch_size
        self.seed = seed
        self.domain_of = load_domains()

    # ------------------------------------------------------------------
    def setup(self, stage=None):
        dp, he = study_subjects(self.seed)
        self.groups = six_groups(dp, he)
        others = [i for i in range(N_OUTER_FOLDS) if i != self.fold]
        dp_train = [s for i in others for s in self.groups[i]['dp']]
        he_train = [s for i in others for s in self.groups[i]['he']]

        if self.valid_group is None:
            train_subjects = dp_train + he_train
            valid_subjects = []
        else:
            train_subjects = [s for i in others if i != self.valid_group
                              for s in self.groups[i]['dp'] + self.groups[i]['he']]
            valid_subjects = (self.groups[self.valid_group]['dp'] +
                              self.groups[self.valid_group]['he'])
        test_subjects = self.groups[self.fold]['dp'] + self.groups[self.fold]['he']

        self.train_dataset = self._make(train_subjects, 0)
        self.valid_dataset = self._make(valid_subjects, 1) if valid_subjects else None
        self.test_dataset = self._make(test_subjects, 2)

    def _make(self, subjects, offset):
        if not subjects:
            return None
        x, y, s = concat_pics(self.inputs, self.labels, self.subjects.numpy(),
                              self.pics.numpy(), subjects, self.n_concat,
                              seed=self.seed + offset)
        domain = torch.tensor([self.domain_of[int(v)] for v in s.numpy()],
                              dtype=torch.long)
        face = torch.zeros_like(s)
        return TensorDataset(x, y, s, face, domain)

    def _loader(self, dataset, shuffle):
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle,
                          num_workers=1, drop_last=False)

    def train_dataloader(self):
        return self._loader(self.train_dataset, True)

    def val_dataloader(self):
        # 评估阶段不使用验证集；Lightning 要求返回空列表而不是 None
        return [] if self.valid_dataset is None else \
            self._loader(self.valid_dataset, False)

    def test_dataloader(self):
        return self._loader(self.test_dataset, False)

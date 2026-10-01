from tools import *


class BaselineModule(L.LightningModule):
    '''
    基线模型：只有骨干网络 Gf 与疾病分类器 Gy（论文表 4 的 baseline 条件）。
    hp 中值为 None 的键会被 Optuna 建议，非 None 的键固定不变。
    '''

    def __init__(self, trial: Optional[optuna.Trial], hp: dict):
        super().__init__()
        self.automatic_optimization = True
        self.hp = munchify(hp)
        self.init_hyperparams(trial)

        backbone = build_backbone(self.hp.model)
        self.model = straight_model_with_grl(backbone, n_inputs=50,
                                             n_subjs=N_SUBJS,
                                             n_domains=K_DOMAINS)
        # 二分类：模型输出 (batch, 2) 的概率/ logits，故按多分类度量
        self.train_loss = torchmetrics.MeanMetric()
        self.train_acc = torchmetrics.Accuracy(task='multiclass', num_classes=2)
        self.valid_loss = torchmetrics.MeanMetric()
        self.valid_metrics = torchmetrics.MetricCollection({
            'acc': torchmetrics.Accuracy(task='multiclass', num_classes=2),
            'auc': torchmetrics.AUROC(task='multiclass', num_classes=2),
            'f1score': torchmetrics.F1Score(task='multiclass', num_classes=2),
            'precision': torchmetrics.Precision(task='multiclass', num_classes=2),
            'recall': torchmetrics.Recall(task='multiclass', num_classes=2),
            'specificity': torchmetrics.Specificity(task='multiclass', num_classes=2),
        }, prefix='valid_')
        self.test_metrics = self.valid_metrics.clone(prefix='test_')

    # ------------------------------------------------------------------
    def lookup(self, fn: Callable, key: str, *args, **kwargs):
        '''值为 None 的键交给 Optuna，其余保持固定。'''
        if self.hp.get(key) is None:
            self.hp[key] = fn(key, *args, **kwargs)
        return self.hp[key]

    def init_hyperparams(self, trial: Optional[optuna.Trial]):
        if trial is None:
            return
        low, high, log = SEARCH_SPACE['label_opt_lr']
        self.lookup(trial.suggest_float, 'label_opt_lr', low, high, log=log)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hp.label_opt_lr,
                                weight_decay=self.hp.get('label_opt_wd', 0.0))

    def training_step(self, batch, batch_idx):
        input, labels, subjs, _, domain = batch
        hedp_y, _, _ = self.model(input)
        loss = F.cross_entropy(hedp_y, labels)
        self.train_loss(loss)
        self.train_acc(hedp_y.argmax(1), labels)
        self.log_dict({'train_loss': self.train_loss, 'train_acc': self.train_acc},
                      prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input, labels, subjs, _, domain = batch
        hedp_y, _, _ = self.model(input)
        probs = F.softmax(hedp_y, dim=1)
        self.valid_loss(F.cross_entropy(hedp_y, labels))
        self.valid_metrics(probs, labels)
        self.log_dict({'valid_loss': self.valid_loss}, prog_bar=True)
        self.log_dict(self.valid_metrics, prog_bar=True)

    def test_step(self, batch, batch_idx):
        input, labels, subjs, _, domain = batch
        hedp_y, _, _ = self.model(input)
        self.test_metrics(F.softmax(hedp_y, dim=1), labels)
        self.log_dict(self.test_metrics)


class GRLModule(BaselineModule):
    '''
    GRLV（论文 3.1–3.4）。每个 iteration 轮换使用三种损失：

        batch_idx % 3 == 0 : domain_grl_bce   —— 域分类损失经 GRL 反传
        batch_idx % 3 == 1 : subj_detach_bce  —— 个体分类损失被 stop-gradient 截断
        batch_idx % 3 == 2 : subj_grl_var     —— 方差损失

    λ 按论文公式 11 随 epoch 上升：
        λ_i = (2 / (1 + exp(-γ·i)) - 1) · λ_max
    '''

    def __init__(self, trial: Optional[optuna.Trial], hp: dict):
        super().__init__(trial, hp)
        # 三支路使用各自的优化器，需要手动控制反向传播
        self.automatic_optimization = False

    def init_hyperparams(self, trial: Optional[optuna.Trial]):
        super().init_hyperparams(trial)
        if trial is None:
            return
        for key in ('grl_gamma', 'std_gamma', 'max_grl_lambd', 'max_std_lambd'):
            low, high, log = SEARCH_SPACE[key]
            self.lookup(trial.suggest_float, key, low, high, log=log)

    def configure_optimizers(self):
        lr = self.hp.label_opt_lr
        wd = self.hp.get('label_opt_wd', 0.0)
        params = list(self.model.parameters())   # 生成器只能用一次，先转成列表
        return [torch.optim.Adam(params, lr=lr, weight_decay=wd)          # domain
                for _ in range(3)]

    def training_step(self, batch, batch_idx):
        from dependences import ABLATIONS
        input, labels, subjs, _, domain = batch
        mode = ['domain_grl_bce', 'subj_detach_bce', 'subj_grl_var'][batch_idx % 3]

        # 消融条件：关闭的支路直接跳过（论文 4.4）
        ablation = self.hp.get('ablation', 'full')
        if ablation == 'no_subj_grl_var' and mode == 'subj_grl_var':
            return
        if ablation == 'no_domain_grl_bce' and mode == 'domain_grl_bce':
            return

        gamma = (self.hp.grl_gamma, 0, self.hp.std_gamma)[batch_idx % 3]
        ramp = 2 / (1 + torch.exp(torch.as_tensor(-1 * gamma * self.current_epoch))) - 1
        lambd = ramp * (-self.hp.max_grl_lambd, 0, self.hp.max_std_lambd)[batch_idx % 3]

        with torch.autocast(device_type=self.device.type, dtype=torch.float16):
            hedp_y, subj_y, domain_y = self.model(
                input, lambd, detach=(mode == 'subj_detach_bce'))

        label_loss = F.cross_entropy(hedp_y, labels)
        if mode == 'domain_grl_bce':
            loss = label_loss + F.cross_entropy(domain_y, domain)
        elif mode == 'subj_detach_bce':
            loss = label_loss + F.cross_entropy(subj_y, subjs)
        else:
            # 论文公式 9：域分类概率与均匀分布的偏离
            p = torch.softmax(domain_y, dim=1)
            loss = label_loss + ((p - 1.0 / K_DOMAINS) ** 2).mean()

        opt = self.optimizers()[batch_idx % 3]
        self.model.zero_grad()
        self.manual_backward(loss)
        opt.step()

        self.train_loss(loss)
        self.train_acc(hedp_y.argmax(1), labels)
        self.log_dict({'train_loss': self.train_loss, 'train_acc': self.train_acc},
                      prog_bar=True)
        return loss

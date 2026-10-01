'''论文中使用的固定超参（表 2、表 3）与默认配置。'''

from munch import munchify, Munch

from dependences import MODELS, K_DOMAINS

best_param_dict = Munch()

# ----------------------------------------------------------------------
# 表 3：控制反传强度的四个系数（论文报告值）
# ----------------------------------------------------------------------
best_param_dict['grl'] = munchify({
    'model': None,
    'label_opt_name': 'Adam',
    'label_opt_lr': None,        # 由表 2 逐骨干给出
    'label_opt_wd': 0,
    'ablation': 'full',
    'grl_gamma': 0.5,            # gamma_clf
    'max_grl_lambd': 2.0,        # lambda_max,clf
    'std_gamma': 3.0,            # gamma_var
    'max_std_lambd': 1.0,        # lambda_max,var
})

# ----------------------------------------------------------------------
# 表 2：逐骨干学习率
# ----------------------------------------------------------------------
LEARNING_RATES = {
    'AttentionBaseNet': 2e-4,
    'EEGMiner': 3e-5,
    'EEGNet': 3e-4,
    'BrainModule': 2e-4,
    'EEGConformer': 2e-4,
    'BIOT': 3e-6,
    'FBCNet': 5e-5,
    'IFNet': 1e-3,
    'MSVTNet': 4e-4,
    'SSTDPN': 2e-4,
}

for _name in MODELS:
    best_param_dict[_name] = munchify({'label_opt_lr': LEARNING_RATES[_name]})


def hp_for(model: str, ablation: str = 'full', **overrides):
    '''返回某个骨干在某个消融条件下的超参字典。'''
    hp = Munch()
    hp.update(best_param_dict['grl'])
    hp.update(best_param_dict[model])
    hp['model'] = model
    hp['ablation'] = ablation
    hp['n_domains'] = K_DOMAINS
    hp.update(overrides)
    return hp

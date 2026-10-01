'''生成并保存固化的域标签（论文 3.4）。

    python domains.py

输出 config.DOMAIN_FILE，即 OUT_DIR/domains.npz，包含：
    subjects   被试编号
    domains    固定域标签（0/1/2）
    kept       保留标记（全部被试参与聚类，恒为 True）
    metrics    各 k 的轮廓系数与自举稳定性（JSON 字符串）
'''

from tools import *


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    set_seed(SEED)
    index, labels, kept, metrics = build_domains(seed=SEED)
    np.savez_compressed(DOMAIN_FILE, subjects=index, domains=labels, kept=kept,
                        metrics=json.dumps(metrics))
    print('每个域的样本数 :', np.bincount(labels).tolist())
    print('wrote', DOMAIN_FILE)


if __name__ == '__main__':
    main()

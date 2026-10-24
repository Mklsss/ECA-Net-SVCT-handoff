# train_dudotrans.py
import os
import sys

# 当前脚本所在目录（DuDoTrans/）
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

# svct2 根目录（上一级目录）
SVCT2_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if SVCT2_ROOT not in sys.path:
    sys.path.insert(0, SVCT2_ROOT)

# 导入原项目的 config.py
import config as svct_config

# 从你的 config.py 中读取稀疏程度
num_sparse_views = svct_config.num_sparse_views
print(f"[train_dudotrans] 使用的稀疏视图数: {num_sparse_views}")

# ----------------------------------------------------
# 导入 DuDoTrans 模型的 main.py（就在当前目录）
# ----------------------------------------------------
import main as dudotrans_main


if __name__ == "__main__":
    # 直接调用 Trainer(num_view=xxx)
    print("[train_dudotrans] 开始训练 DuDoTrans...")

    trainer = dudotrans_main.Trainer(
        num_view=num_sparse_views,     # ← 关键：稀疏视图来自 config.py
        learning_rate=0.0001,
        is_restart=True,
        max_epoch=100,
        is_cuda=True
    )

    trainer.train()
    print("[train_dudotrans] 训练结束")


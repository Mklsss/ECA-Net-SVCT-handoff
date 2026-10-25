# ECA-Net SVCT 项目交接

对应论文：**ECA-Net enhanced dual-domain learning for sparse-view CT reconstruction**。

这是 2026-10-24 对 U 盘 `G:\autodl_backup` 的交接快照。此项目与 [Mklsss/DDF](https://github.com/Mklsss/DDF) 中的 DDF 网络是不同项目。保留原始实验文件，不在此次交接中修改模型、数据或实验结果。

## 下载完整备份

完整备份在本仓库的 **Releases** 中，仓库首页的 “Download ZIP” 仅包含源码和文档，不能代替完整备份。

上传状态：完整备份已上传并校验。原始文件 291 个，总计 40,959,524,041 字节；共 22 个分卷。每个附件的 GitHub SHA-256 与本地一致。

在 Release 页面下载全部 `autodl_backup.full.tar.partNNN` 和 `backup_manifest.json`，放入同一个目录。分卷属于一个未压缩 TAR 流，不能单独解压；原来的 34 GB `.tar.gz` 文件完整保存在这个流中，没有重新压缩或改变内容。

使用 Python 3.10 或更新版本，从仓库目录运行：

```bash
python restore_backup.py --parts-dir /path/to/downloaded-assets --out /path/to/restore
```

脚本先校验所有分卷 SHA-256，再还原 `autodl_backup/`，最后校验每个原始文件的 SHA-256。不覆盖已有的同名备份目录。下载分卷约占 41 GB，此外至少预留 45 GB 用于还原；若还要展开主项目压缩包，再预留约 38 GB。若同时保留下载包、还原目录与展开后的主项目，应准备约 125 GB 空间。

本仓库已公开，下载无需协作者权限；GitHub CLI 用户也可以使用 `gh release download usb-backup-2026-10-24 --repo Mklsss/ECA-Net-SVCT-handoff` 下载附件。

## 完整备份目录

| 路径 | 内容 |
| --- | --- |
| `svct_backup_2026-05-03.tar.gz` | 主项目：投影 CNN、FBP、GAN/ECA、训练与评估代码、权重、训练日志 |
| `dataset/train_meiaonew.npz` | 1800 张训练切片及完整投影 |
| `dataset/test_meiaonew.npz` | 200 张测试切片及完整投影 |
| `DuDoTrans/` | 原始 baseline 代码及多个视角/轮次的权重备份 |
| `dudotrans_autodl/` | 整理后的 DuDoTrans 训练、测试、导出脚本 |
| `redcnn/`、`redcnn_autodl/` | RED-CNN 原目录和整理版代码、旧权重 |
| `exports/` | 主模型、消融与 baseline 示例 PNG、拼图脚本 |
| `dudotransCoauthor.zip`、`redcnnCoauthor.zip` | 历史交接压缩包；与展开目录不是完全相同版本 |

Git 仓库中的 `svct/` 是从原主压缩包提取的源码；baseline 源码和 `exports` 拼图脚本也提供在线可读副本。完整历史目录、缓存、权重和日志以 Release 中的完整备份为准。原压缩包内还有一份 `checkpoints/module/`，运行时应以项目根目录的 `module/` 为入口。

## 数据

两个 NPZ 均包含 float32 数组：

| 键 | 训练形状 | 测试形状 |
| --- | --- | --- |
| `ct_label` | `(1800, 256, 256, 1)` | `(200, 256, 256, 1)` |
| `sin357` | `(1800, 360, 357)` | `(200, 360, 357)` |

训练集 SHA-256：`772c1c3c6859c64a7ba5b029fdf1abde39f7fd991b4f72fcf1ed17ec8de24e04`。

测试集 SHA-256：`efaecbc43780e06d6701e275b7f0d12afab68190cd20a3ecc50e0fd05e720069`。

这两个文件是此次从电脑移入 U 盘的数据，字段、样本数和尺寸与论文描述吻合；未通过原始出表日志确认其一定就是论文最终使用的同一版本。没有原始 DICOM、患者划分清单或 HU 标定信息，不应仅凭归一化 CT 数组报告 HU 误差。数据源为论文所述 Mayo/NIH-AAPM 数据；本交接不改变第三方数据的使用条件。

## 运行入口与配置

1. 先还原完整备份。展开主项目：`tar -xzf autodl_backup/svct_backup_2026-05-03.tar.gz -C /path/to/work`，得到 `svct/`。
2. 主项目入口是 `svct/main.py`，投影域模型在 `module/projection_cnn.py`，图像域模型在 `module/generator.py` 和 `module/discriminator.py`；指标在 `utils/metrics.py`。
3. 按 `config.py` 的 `DATA_DIR` 配置数据位置，或在项目要求的 `data/` 目录放置上述两个 NPZ。FBP 预计算入口在 `main.py`。导出脚本要求 `test_fbp_<views>views.npz`，其字段包括 `fbp_li`、`fbp_cnn`、`ct_label`；这些预计算文件不在本备份中，需要按原流程生成。
4. 查看参数再运行：`python main.py --help`。历史训练命令和环境快照在压缩包内的 `swanlog/*/files/`；环境快照可能含服务器本地路径，不能直接视为可移植安装文件。
5. baseline 的运行方法见各自的 `README.md`。需要 PyTorch/torchvision、NumPy、SciPy、scikit-image、SwanLab，以及对应流程使用的 ASTRA/ODL 等依赖；GPU/CUDA 版本应与这些算子兼容。未完成全新环境的端到端复现验证。

`main.py` 在解析命令行参数之前就初始化 SwanLab，原代码的 workspace 是 `mkls`；接手时需配置自己的记录/认证方式。该初始化中的日志参数字典与 `config.py` 默认值不同，不应把日志字典直接当成实际生效配置。感知损失使用 `vgg19(pretrained=True)`，首次训练还需准备对应预训练权重缓存或允许 torchvision 下载。

**主包当前默认配置是 `num_sparse_views=120`、`USE_ECA_IN_GENERATOR=False`、`ECA_KERNEL_SIZE=3`。** 直接使用默认值会进入无 ECA 的设置。运行 GAN+ECA 时需要显式开启 ECA，并选择同视角、同架构的生成器和投影 CNN 权重。`export_svct_pred_by_indices.py` 也读取此配置；无 ECA 的独立导出脚本显式设置 `use_eca=False`。

该配置快照中的损失权重为 `LAMBDA_MSE=0.8`、`LAMBDA_ADV=0.05`、`LAMBDA_CONTENT=0.0001`、`LAMBDA_TV=1e-7`。这些是备份当前默认值，不代表已经核实了每个视角最终出表时的参数。

主包 `output/` 中有四组 `projection_cnn_final_<views>views.pth`、`generator_final_<views>views.pth`，以及无 ECA 权重。30-view 无 ECA 文件名为 `generator_final_30views_100enoeca.pth`，其他视角为 `generator_final_<views>views_noeca.pth`。名称和目录不构成实验来源证明，需与原始训练配置和论文表格对应。

例如，使用完整备份内的 120-view GAN+ECA 权重时，先在自己的工作副本中设置正确的 `DATA_DIR`、`num_sparse_views=120`、`USE_ECA_IN_GENERATOR=True`，再从展开后的 `svct/` 运行以下原有入口：

```bash
python main.py --mode precompute_fbp --cnn_weights output/projection_cnn_final_120views.pth
python main.py --mode test --cnn_weights output/projection_cnn_final_120views.pth --gen_weights output/generator_final_120views.pth
```

这些命令由原脚本参数和现有文件名整理，未在此次交接中完成 GPU 执行验证；切换视角时应同步修改配置、投影 CNN、生成器和预计算数据文件。

## 已知问题与复现边界

- **DuDoTrans：** `dudotrans_autodl/results/models/view_060/` 和 `view_120/` 的最终 `.pth.tar` 容器索引损坏。完整备份仍保留它们以忠实保存现场。`DuDoTrans/results/models/view_030/`、`view_060/`、`view_090/`、`view_120/` 下 `epoch_019_iter_001799.pth.tar` 均通过 ZIP 成员 CRC 校验，可作为完整候选副本；尚未在此次交接中运行 GPU 推理。
- **RED-CNN：** 只找到旧 `REDCNN_100000iter.ckpt`，未找到 30/60/90/120 四组明确对应论文实验的 `REDCNN_epoch_099.ckpt`。代码支持重训，现有材料不能保证直接还原论文该 baseline 的四组数值。README 内的权重路径是命令示例。
- **主模型：** `output/discriminator_final_60views_noeca.pth` 仅 377,749,504 字节，同类通常约 1,078,352,534 字节，需要验证；`checkpoints/60views/` 中另有候选备份。未逐个加载主模型权重。
- **设置对应：** 主表使用 30/60/90/120 views；论文正文写的 `S=2/4/8/12` 若从 360 个角度降采样，对应 180/90/45/30。需确认实际数据选择，不能混用两组设置。
- **角度索引：** `config.py` 中 `SPARSE_VIEW_ANGLES` 与 `SPARSE_VIEW_INDICES` 分别使用 360 和 359 作为 `linspace` 上界，两套索引并不相同。生成稀疏输入、CNN 训练和 FBP 时需核对实际使用的索引及对应角度；本交接未修改该配置。
- **baseline 可比性：** 当前 DuDoTrans/RED-CNN 数据代码使用 512×512，主模型为 256×256；DuDoTrans 还重新生成投影并加噪声。比较时需明确分辨率、几何、噪声、归一化、测试样本和指标口径。
- **评估材料：** `exports/` 是选定切片的 8-bit PNG，不能替代完整测试集、浮点预测或逐切片指标。备份中未发现完整逐切片 CSV 指标表。
- **后续修改：** 审稿涉及 HU、误差范围、参数量/FLOPs/推理时间、损失权重敏感性、空间注意力及创新性论证。此次仅交接原文件，没有实现或补做这些实验；也没有修改论文。正式邮件决定为拒稿并鼓励充分修改后重新投稿。

## 存储方式与验证范围

完整目录按文件原样归档，分卷与原始文件的长度/SHA-256 在 `backup_manifest.json` 中。GitHub [Release 单附件须小于 2 GiB](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases)，因此每卷上限为 1,900,000,000 字节。使用 Release 附件保存完整大备份，源码和说明使用普通 Git。

本次验证是文件备份和交接检查，不代表已重跑训练、复现论文表 1/表 2，或验证所有权重均可用于推理。原 U 盘文件不因上传而删除。

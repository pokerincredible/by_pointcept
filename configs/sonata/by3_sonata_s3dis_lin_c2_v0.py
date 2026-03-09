# ============================================================================
# 基础配置文件继承
# ============================================================================
_base_ = ["../_base_/default_runtime.py"]  # 继承默认运行时配置（日志、保存路径等基础设置）

# ============================================================================
# 训练基本设置
# ============================================================================
# batch_size: 所有GPU上的总批次大小
# - 如果使用4个GPU，每个GPU的batch_size = 24 / 4 = 6
# - 增大batch_size可以提高训练稳定性，但会增加显存占用
# - 如果遇到OOM（内存不足），可以减小此值（如12或16）
batch_size = 10  # total batch size across all GPUs (reduced from 24 to avoid OOM)

# num_worker: 数据加载的并行工作进程数
# - 增大此值可以加快数据加载速度，但会占用更多CPU和内存
# - 建议设置为CPU核心数的1/2到1倍
# - 如果遇到内存问题，可以减小此值（如24或16）
num_worker = 20

# mix_prob: MixUp数据增强的概率
# - 用于混合不同样本的数据增强技术
# - 0.8表示80%的概率进行MixUp增强
# - 有助于提高模型的泛化能力
mix_prob = 0.8

# clip_grad: 梯度裁剪阈值
# - 防止梯度爆炸，当梯度范数超过此值时会被裁剪
# - 3.0是一个常用的安全值
# - 如果训练不稳定，可以减小此值（如1.0或0.5）
clip_grad = 3.5

# empty_cache: 是否定期清空GPU缓存
# - True: 每个iteration后清空缓存，可以节省显存但可能略微降低速度
# - False: 不清空，速度更快但显存占用更高
# - 如果遇到OOM，建议设置为True
empty_cache = True  # enable to free GPU memory periodically (required for OOM prevention)

# enable_amp: 是否启用自动混合精度训练
# - True: 使用FP16混合精度，可以节省显存并加速训练
# - False: 使用FP32全精度，更稳定但更慢更占显存
# - 建议保持True以节省显存
enable_amp = True

# eval_epoch: 每多少个epoch进行一次验证评估
# - 50表示每训练50个epoch评估一次验证集性能
# - 评估会消耗额外时间，但可以监控模型性能
eval_epoch = 50  # evaluate every 50 epochs

# ============================================================================
# 模型配置
# ============================================================================
model = dict(
    # 模型类型：使用默认的语义分割器V2版本
    type="DefaultSegmentorV2",
    
    # num_classes: 分类类别数，这里是二分类任务
    num_classes=2,
    
    # backbone_out_channels: 骨干网络输出的特征通道数
    # - 需要与backbone的最终输出维度匹配
    backbone_out_channels=1232,
    
    # ========================================================================
    # 骨干网络配置（Point Transformer v3m2架构）
    # ========================================================================
    backbone=dict(
        # 骨干网络类型：Point Transformer v3m2版本
        type="PT-v3m2",
        
        # in_channels: 输入特征通道数
        # - 6 = 坐标(3维) + 颜色(3维)
        # - 如果只有坐标没有颜色/法向量，设置为3
        # - 如果有法向量，可以设置为9（坐标3 + 颜色3 + 法向量3）
        in_channels=6,  # coord (3) + color (3); if no color/normal, set to 3
        
        # order: 点云排序方式，用于不同的空间编码策略
        # - "z": Z轴排序
        # - "z-trans": Z轴平移排序
        # - "hilbert": Hilbert曲线排序
        # - "hilbert-trans": Hilbert曲线平移排序
        # - 多种排序方式可以增强模型的鲁棒性
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        
        # stride: 每个编码层的下采样步长
        # - (2, 2, 2, 2) 表示每层都进行2倍下采样
        # - 总共4层，最终下采样倍数为2^4=16倍
        stride=(2, 2, 2, 2),
        
        # enc_depths: 每个编码层的Transformer块数量
        # - (3, 3, 3, 12, 3) 表示5个编码层，每层的块数
        # - 第4层有12个块，是主要的特征提取层
        enc_depths=(3, 3, 3, 12, 3),
        
        # enc_channels: 每个编码层的特征通道数
        # - 随着层数加深，通道数逐渐增加：48 → 96 → 192 → 384 → 512
        # - 通道数越多，模型容量越大，但显存占用也越高
        enc_channels=(48, 96, 192, 384, 512),
        
        # enc_num_head: 每个编码层的多头注意力头数
        # - 头数越多，模型表达能力越强，但计算量也越大
        # - 通常通道数能被头数整除
        enc_num_head=(3, 6, 12, 24, 32),
        
        # enc_patch_size: 每个编码层的patch大小（点数）
        # - 每个patch包含的点数，用于分组处理
        # - 1024是一个常用的值，平衡了效率和精度
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        
        # mlp_ratio: MLP层的扩展比例
        # - Transformer中MLP层的隐藏层维度 = 输入维度 × mlp_ratio
        # - 4是常用的值，表示隐藏层是输入层的4倍
        mlp_ratio=4,
        
        # qkv_bias: 是否在Q、K、V线性层中使用偏置
        # - True: 使用偏置，通常能略微提升性能
        qkv_bias=True,
        
        # qk_scale: QK缩放因子
        # - None: 使用默认缩放（1/sqrt(head_dim)）
        # - 可以手动设置以调整注意力分布
        qk_scale=None,
        
        # attn_drop: 注意力层的dropout率
        # - 0.0表示不使用dropout
        # - 可以设置为0.1-0.2以增强正则化
        attn_drop=0.0,
        
        # proj_drop: 投影层的dropout率
        # - 0.0表示不使用dropout
        proj_drop=0.0,
        
        # drop_path: 随机深度（Stochastic Depth）的dropout率
        # - 0.25表示25%的概率跳过某些层，用于正则化
        # - 对于小数据集，可以适当减小（如0.15-0.2）
        drop_path=0.25,  # slightly reduced for small dataset
        
        # shuffle_orders: 是否在训练时随机打乱排序顺序
        # - True: 增强数据多样性，提高模型鲁棒性
        shuffle_orders=True,
        
        # pre_norm: 是否使用Pre-Norm架构
        # - True: LayerNorm在注意力/MLP之前（更稳定）
        # - False: LayerNorm在注意力/MLP之后（Post-Norm）
        pre_norm=True,
        
        # enable_rpe: 是否启用相对位置编码
        # - False: 不使用相对位置编码
        # - True: 使用相对位置编码，可能提升性能但增加计算量
        enable_rpe=False,
        
        # enable_flash: 是否启用Flash Attention优化
        # - True: 使用Flash Attention，可以节省显存并加速
        # - 需要GPU支持（如A100、RTX 3090等）
        # - ⚠️ 重要：启用Flash Attention时，upcast_attention和upcast_softmax必须为False
        enable_flash=True,
        
        # upcast_attention: 是否在注意力计算时上转为FP32
        # - False: 使用FP16，更快但可能精度略低
        # - True: 使用FP32，更精确但更慢
        # - ⚠️ 重要：当enable_flash=True时，必须设置为False（Flash Attention不支持上转）
        upcast_attention=False,
        
        # upcast_softmax: 是否在softmax时上转为FP32
        # - False: 使用FP16
        # - True: 使用FP32，避免数值不稳定
        # - ⚠️ 重要：当enable_flash=True时，必须设置为False（Flash Attention不支持上转）
        upcast_softmax=False,
        
        # traceable: 是否可追踪（用于导出模型）
        # - False: 不追踪，训练时使用
        traceable=False,
        
        # mask_token: 是否使用mask token（用于预训练）
        # - False: 不使用，用于下游任务
        mask_token=False,
        
        # enc_mode: 是否只使用编码器模式
        # - True: 只使用编码器，用于分割任务
        enc_mode=True,
        
        # freeze_encoder: 是否冻结编码器参数
        # - False: 不冻结，所有参数可训练（用于微调）
        # - True: 冻结编码器，只训练分类头（用于线性评估）
        freeze_encoder=False,
    ),
    
    # ========================================================================
    # 损失函数配置
    # ========================================================================
    criteria=[
        # 交叉熵损失：标准的分类损失
        dict(
            type="CrossEntropyLoss",
            loss_weight=1.0,  # 损失权重，控制该损失在总损失中的贡献
            ignore_index=-1,  # 忽略标签为-1的样本（通常是无效点）
            # Optional: add class weights if imbalanced
            # ⚠️ 提升 class_1 的 IoU：增加 class_1 的权重以提升召回率
            # - 从 2.5 增加到 3.5-4.0，让模型更关注 class_1 的样本
            # - 如果 class_1 是少数类且 IoU 低，说明漏检多，需要更大权重
            weight=[1.0, 1.0]  # 如果类别不平衡，可以为少数类设置更大的权重
            # 例如：如果class_1是少数类，可以设置weight=[1.0, 2.0]来平衡
        ),
        # Lovasz损失：基于Lovasz扩展的损失函数，对类别不平衡更鲁棒
        # ⚠️ 提升 IoU 的关键：LovaszLoss 直接优化 IoU，增加其权重可以更直接地提升 IoU
        # - 从 1.0 增加到 2.0-2.5，让模型更专注于优化 IoU 指标
        dict(
            type="LovaszLoss",
            mode="multiclass",  # 多分类模式
            loss_weight=1.5,  # 损失权重（增加以直接优化IoU）
            ignore_index=-1,  # 忽略标签
        ),
        # Dice损失：直接优化 Dice 系数（与 IoU 密切相关），对不平衡数据友好
        # - Dice Loss 专注于提升预测区域与真实区域的重叠度
        # - 有助于提升 class_1 的 IoU，特别是当漏检较多时
        # dict(
        #     type="DiceLoss",
        #     loss_weight=0.5,  # 较小的权重，作为辅助损失
        #     ignore_index=-1,
        # ),
    ],
    
    # freeze_backbone: 是否冻结骨干网络
    # - False: 不冻结，所有参数可训练（用于微调）
    # - True: 冻结骨干网络，只训练分类头（用于线性评估）
    freeze_backbone=False,  # unfreeze for fine-tuning
)

# ============================================================================
# 优化器和学习率调度器配置
# ============================================================================
# epoch: 训练的总轮数
# - 1500个epoch，对于小数据集可能需要较长时间
# - 可以根据验证集性能提前停止（early stopping）
epoch = 1500

# optimizer: 优化器配置
optimizer = dict(
    type="AdamW",  # 使用AdamW优化器（Adam with weight decay）
    lr=0.002,  # 初始学习率，对于微调任务通常较小
    weight_decay=0.02,  # 权重衰减系数，用于L2正则化防止过拟合
)

# scheduler: 学习率调度器配置
scheduler = dict(
    type="OneCycleLR",  # 单周期学习率调度策略
    # max_lr: 最大学习率列表
    # - [0.002, 0.0002]: 主网络学习率0.002，backbone学习率0.0002
    # - backbone使用更小的学习率，因为它是预训练的
    max_lr=[0.003, 0.0003],
    # pct_start: 学习率上升阶段占总训练的比例
    # - 0.05表示前5%的迭代用于学习率从初始值上升到最大值
    # - 剩余95%用于学习率衰减
    pct_start=0.05,
    # anneal_strategy: 衰减策略
    # - "cos": 余弦退火，学习率按余弦函数平滑下降
    anneal_strategy="cos",
    # div_factor: 初始学习率 = max_lr / div_factor
    # - 10.0表示初始学习率是最大学习率的1/10
    div_factor=10.0,
    # final_div_factor: 最终学习率 = initial_lr / final_div_factor
    # - 1000.0表示最终学习率是初始学习率的1/1000
    # - 学习率会从初始值逐渐衰减到很小的值
    final_div_factor=1000.0,
)

# param_dicts: 不同参数组的学习率设置
# - keyword="block": 匹配包含"block"的参数（通常是backbone的Transformer块）
# - lr=0.0002: 这些参数使用0.0002的学习率（比主网络小10倍）
# - 这样可以精细调整预训练模型，避免破坏预训练权重
param_dicts = [dict(keyword="block", lr=0.0002)]

# ============================================================================
# 数据集配置
# ============================================================================
# dataset_type: 数据集类型，使用S3DIS格式的数据集
dataset_type = "S3DISDataset"

# data_root: 数据集根目录路径
# - 需要包含训练、验证、测试数据的文件夹
data_root = "/root/autodl-fs/by3_class2"  # ← 修改为你的二分类数据路径

data = dict(
    # num_classes: 类别数量
    num_classes=2,
    
    # ignore_index: 忽略的标签索引
    # - -1表示标签为-1的点不参与损失计算和评估
    ignore_index=-1,
    
    # names: 类别名称列表
    # - 用于日志输出和可视化
    # - 可以替换为实际的类别名称，如 ["cortical", "trabecular"]（皮质骨、松质骨）
    names=["class_0", "class_1"],  # 可替换为实际名称，如 ["cortical", "trabecular"]
    # ========================================================================
    # 训练集配置
    # ========================================================================
    train=dict(
        type=dataset_type,  # 数据集类型
        split=("Area_1",),  # 训练集使用的数据分割，可以指定多个区域如("Area_1", "Area_2")
        data_root=data_root,  # 数据根目录
        
        # transform: 数据增强和预处理流水线（按顺序执行）
        transform=[
            # 1. 中心化：将点云中心移到原点（Z轴也应用）
            # - apply_z=True: Z轴也参与中心化
            dict(type="CenterShift", apply_z=True),
            
            # 2. 随机丢弃点：模拟遮挡或噪声
            # - dropout_ratio=0.2: 丢弃20%的点
            # - dropout_application_ratio=1.0: 100%的概率应用此增强
            dict(type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=1.0),
            
            # 3. 随机旋转（Z轴）：绕Z轴旋转，增加旋转不变性
            # - angle=[-1, 1]: 旋转角度范围（弧度），约-57°到57°
            # - p=0.5: 50%的概率应用此增强
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            
            # 4. 随机旋转（X轴）：小幅旋转，模拟视角变化
            # - angle=[-1/64, 1/64]: 约-0.9°到0.9°的小角度旋转
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=0.5),
            
            # 5. 随机旋转（Y轴）：小幅旋转
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=0.5),
            
            # 6. 随机缩放：模拟不同距离的观察
            # - scale=[0.9, 1.1]: 缩放因子在0.9到1.1之间
            dict(type="RandomScale", scale=[0.9, 1.1]),
            
            # 7. 随机翻转：水平翻转，增加数据多样性
            # - p=0.5: 50%的概率翻转
            dict(type="RandomFlip", p=0.5),
            
            # 8. 随机抖动：给坐标添加小噪声
            # - sigma=0.005: 噪声标准差
            # - clip=0.02: 噪声裁剪范围，防止过大偏移
            dict(type="RandomJitter", sigma=0.005, clip=0.02),
            
            # 9. 弹性形变：对医学数据可能有害，已注释
            # ElasticDistortion often harmful for medical data → commented
            # dict(type="ElasticDistortion", distortion_params=[[0.2, 0.4], [0.8, 1.6]]),
            
            # 10-12. 颜色增强：已移除，因为输入只有3通道坐标，没有颜色信息
            # dict(type="ChromaticAutoContrast", p=0.2, blend_factor=None),
            # dict(type="ChromaticTranslation", p=0.95, ratio=0.05),
            # dict(type="ChromaticJitter", p=0.95, std=0.05),
            
            # 13. 网格采样：将点云体素化为规则网格
            # - grid_size=0.005: 网格大小（单位：米），5mm的精细网格
            #   * 更小的grid_size会保留更多细节，但显存占用更大
            #   * 如果遇到OOM，可以增大到0.01或0.02
            # - hash_type="fnv": 使用FNV哈希函数进行快速查找
            # - mode="train": 训练模式，会进行随机采样
            # - return_grid_coord=True: 返回网格坐标，用于后续处理
            dict(
                type="GridSample",
                grid_size=0.005,  # increased from 0.005 to reduce memory usage
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
            ),
            
            # 14. 球体裁剪（按采样率）：随机裁剪点云的一部分
            # - sample_rate=0.6: 保留60%的点
            # - mode="random": 随机模式
            dict(type="SphereCrop", sample_rate=0.6, mode="random"),
            
            # 15. 球体裁剪（按点数上限）：限制点的最大数量
            # - point_max=102400: 最多保留102400个点
            #   * 如果点云太大，会随机采样到此数量
            #   * 如果遇到OOM，可以减小此值（如80000或60000）
            # - mode="random": 随机模式
            dict(type="SphereCrop", point_max=102400, mode="random"),  # reduced from 102400 to save memory
            
            # 16. 中心化（Z轴不应用）：再次中心化，但保持Z轴不变
            dict(type="CenterShift", apply_z=False),
            
            # 17. 颜色归一化：已移除，因为数据已经在[-1, 1]范围内
            # ❌ REMOVED: NormalizeColor (your data is already in [-1, 1]
            
            # 18. 转换为张量：将numpy数组转换为PyTorch张量
            dict(type="ToTensor"),
            
            # 19. 收集数据：指定需要收集的键
            # - keys: 主要数据键（坐标、网格坐标、分割标签）
            # - feat_keys: 特征键（颜色、法向量）
            # - coord总是被包含，不需要在feat_keys中指定
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment"),
                feat_keys=("color", "normal"),  # 输入只有3通道坐标，使用coord作为feat
            ),
        ],
        test_mode=False,  # 训练模式，会应用所有数据增强
        loop=2,  # 每个epoch循环1次数据集
    ),
    # ========================================================================
    # 验证集配置
    # ========================================================================
    val=dict(
        type=dataset_type,  # 数据集类型
        split="Area_5",  # 验证集使用的数据分割
        data_root=data_root,  # 数据根目录
        
        # transform: 验证时的预处理流水线（不包含数据增强）
        transform=[
            # 1. 中心化：将点云中心移到原点
            dict(type="CenterShift", apply_z=True),
            
            # 2. 网格采样：体素化处理
            # - grid_size=0.005: 与训练集保持一致
            # - mode="train": 验证模式必须使用"train"，因为test_mode=False时走prepare_train_data
            #   * "train"模式返回dict，可以被后续transform处理
            #   * "test"模式返回list，只适用于test_mode=True的情况
            dict(
                type="GridSample",
                grid_size=0.01,  # increased from 0.005 to match train config
                hash_type="fnv",
                mode="train",  # ✅ 验证模式必须使用"train"（test_mode=False时）
                return_grid_coord=True,
            ),
            
            # 3. 中心化（Z轴不应用）
            dict(type="CenterShift", apply_z=False),
            
            # 4. 颜色归一化：已移除
            # ❌ REMOVED: NormalizeColor
            
            # 5. 转换为张量
            dict(type="ToTensor"),
            
            # 6. 收集数据
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment"),
                feat_keys=("color", "normal"),  # 输入只有3通道坐标，使用coord作为feat
            ),
        ],
        test_mode=False,  # 验证模式：不使用test_cfg，直接使用transform流水线
    ),
    # ========================================================================
    # 测试集配置（用于最终评估和推理）
    # ========================================================================
    test=dict(
        type=dataset_type,  # 数据集类型
        split="Area_5",  # 测试集使用的数据分割
        data_root=data_root,  # 数据根目录
        
        # transform: 测试时的初始预处理
        transform=[
            # 1. 中心化：将点云中心移到原点
            dict(type="CenterShift", apply_z=True),
            # ❌ REMOVED: NormalizeColor（颜色归一化已移除）
        ],
        test_mode=True,  # 测试模式
        
        # test_cfg: 测试时的详细配置（包含测试时增强TTA）
        test_cfg=dict(
            # voxelize: 体素化配置
            voxelize=dict(
                type="GridSample",
                grid_size=0.005,  # increased from 0.005 to match train config
                hash_type="fnv",
                mode="test",  # 测试模式：返回list用于多块处理
                return_grid_coord=True,
            ),
            
            # crop: 裁剪配置
            # - None: 不进行裁剪，使用整个点云
            # - 可以设置裁剪策略以处理大点云
            crop=None,
            
            # post_transform: 体素化后的后处理
            post_transform=[
                dict(type="CenterShift", apply_z=False),  # 中心化
                dict(type="ToTensor"),  # 转张量
                dict(
                    type="Collect",
                    keys=("coord", "grid_coord", "index"),  # 注意：测试时收集index而不是segment
                    feat_keys=("color", "normal"),  # 输入只有3通道坐标，使用coord作为feat
                ),
            ],
            
            # aug_transform: 测试时增强（Test Time Augmentation, TTA）
            # - 对同一数据应用多种变换，然后对结果进行平均或投票
            # - 可以提高测试精度，但会增加推理时间
            aug_transform=[
                # 基础旋转：0°, 90°, 180°, 270°（绕Z轴）
                [dict(type="RandomRotateTargetAngle", angle=[0], axis="z", p=1)],
                [dict(type="RandomRotateTargetAngle", angle=[1 / 2], axis="z", p=1)],  # 90° = π/2
                [dict(type="RandomRotateTargetAngle", angle=[1], axis="z", p=1)],  # 180° = π
                [dict(type="RandomRotateTargetAngle", angle=[3 / 2], axis="z", p=1)],  # 270° = 3π/2
                
                # 旋转 + 缩小（0.98倍）：4个角度 × 缩小
                [dict(type="RandomRotateTargetAngle", angle=[0], axis="z", p=1), dict(type="RandomScale", scale=[0.98, 0.98])],
                [dict(type="RandomRotateTargetAngle", angle=[1 / 2], axis="z", p=1), dict(type="RandomScale", scale=[0.98, 0.98])],
                [dict(type="RandomRotateTargetAngle", angle=[1], axis="z", p=1), dict(type="RandomScale", scale=[0.98, 0.98])],
                [dict(type="RandomRotateTargetAngle", angle=[3 / 2], axis="z", p=1), dict(type="RandomScale", scale=[0.98, 0.98])],
                
                # 旋转 + 放大（1.02倍）：4个角度 × 放大
                [dict(type="RandomRotateTargetAngle", angle=[0], axis="z", p=1), dict(type="RandomScale", scale=[1.02, 1.02])],
                [dict(type="RandomRotateTargetAngle", angle=[1 / 2], axis="z", p=1), dict(type="RandomScale", scale=[1.02, 1.02])],
                [dict(type="RandomRotateTargetAngle", angle=[1], axis="z", p=1), dict(type="RandomScale", scale=[1.02, 1.02])],
                [dict(type="RandomRotateTargetAngle", angle=[3 / 2], axis="z", p=1), dict(type="RandomScale", scale=[1.02, 1.02])],
                
                # 水平翻转
                [dict(type="RandomFlip", p=1)],
            ],
            # 总共13种增强：4种旋转 + 4种旋转+缩小 + 4种旋转+放大 + 1种翻转
            # 每种增强都会产生一个预测结果，最终通过投票或平均得到最终结果
        ),
    ),
)

# ============================================================================
# 训练钩子（Hooks）配置
# ============================================================================
# hooks: 训练过程中执行的回调函数列表
hooks = [
    # 1. 迭代计时器：记录每次迭代的时间
    # - warmup_iter=2: 前2次迭代不计时（用于预热）
    dict(type="IterationTimer", warmup_iter=2),
    
    # 2. 信息写入器：将训练信息写入日志文件
    # - 记录损失、学习率、准确率等指标
    dict(type="InformationWriter"),
    
    # 3. 语义分割评估器：在验证集上评估模型性能
    # - 计算IoU、mIoU、准确率等指标
    dict(type="SemSegEvaluator"),
    
    # 4. 检查点保存器：保存模型权重
    # - save_freq=None: 使用默认保存频率（通常每个epoch保存一次）
    # - 可以设置为数字，如save_freq=10表示每10个epoch保存一次
    dict(type="CheckpointSaver", save_freq=None),
    
    # 5. 精确评估器：在测试集上进行详细评估
    # - test_last=True: 训练结束后在测试集上评估最终模型
    # - 会使用test_cfg中的测试时增强（TTA）
    dict(type="PreciseEvaluator", test_last=True),
]
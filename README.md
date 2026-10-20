# MAJC / IND+C

Marginal-Anchored Joint Control：在同一个真实前缀上先求 IND 分布，再以直接共同事件评分 C 调整下一 token 的分布。

本仓库是从研究工程提取的独立 Python 实现，包含共享数值求解器、固定候选支持接口、概率汇总、原始轻量预测头、测试和合成示例。**代码整理和数值测试不构成真实整答成功率提升的证据。**

## 安装与检查

```bash
python -m pip install -e .
python -m unittest -v test_majc
python example.py
# 可选：原始 PyTorch 边际/共同事件预测头
python -m pip install -e '.[heads]'
```

Python >=3.10；控制器只依赖 NumPy 和 SciPy。示例不加载语言模型，不访问网络，也不调用付费 API。

## 接口

```python
from majc import Config, control

probabilities, audit = control(
    reference=[0.32, 0.24, 0.16, 0.08, 0.13, 0.07, 0.0],
    candidate_ids=[0, 1, 2, 3],
    u=[0.9] * 4,
    q=[[0.7, 0.7]] * 4,
    eta_c=[0.55, 0.48, 0.44, 0.42],
    protocol="executed_scale",
    config=Config(tau_kl=0.05, delta_kl=0.02),
)
```

- `reference` 是共享单规则控制器输出的全词表分布；硬禁 token 的概率为 0。
- `candidate_ids` 由调用方按冻结的 LM top-k 顺序选取一次，通常 K=16；本模块不改变 tie-break。
- `u[a] = P(U=1|当前前缀,候选a)`。
- `q[a,j] = P(r_j=1|U=1,当前前缀,候选a)`；列对应去重后的不同事件。
- `eta_c[a] = P(全部r_j=1|U=1,当前前缀,候选a)`。
- `J_ind=u*prod(q)`；`J_c=u*eta_c`；`M=u*weighted_mean(q)`。原始重复要求权重只用于 M，不重复乘同一事件。
- 求解过程不采样、不推进 LM 缓存；调用方仅对最终全词表分布采样一次。

此接口需要当前前缀的候选预测值。仓库不附预训练权重、任务数据、完整语言模型、裁判服务或通用自然语言规则解析器。`majc_models.py` 保留原 FeatureCodec、MarginalModel、JointHead、训练损失与 Predictor，供已有特征/检查点接入；训练协议与运行预算由调用方显式管理。

## 三个协议明确分开

设 p 为候选 SL 条件分布，rho 为 IND 解，z 为 C 阶段分布。IND 阶段始终最大化 `E[J_ind] - tau*KL(z||p)`，保护相对 p 的 u/M，并满足 `KL(z||p)<=delta`。

| protocol | C 阶段软 KL 参照 | C 的 u/M 阈值 | C 的 IND 阈值 | 用途 |
|---|---|---|---|---|
| `executed_scale`（默认） | p | rho | rho | 现有 scale 代码的目标身份 |
| `executed_quick` | p | p | rho | 现有 quick / MOSAIC 代码的目标身份 |
| `disclosure_v1_1` | rho | rho | rho | v1.1 方法稿定义，不能追认为旧实验实跑版本 |

三者的硬 KL **全部相对同一个原始 p**，上限均为 0.02；第二阶段没有额外 KL 额度。软系数默认 0.05。`disclosure_v1_1` 是此次新增的显式实现路径，仅经数值测试，不附真实生成实验结果。

默认沿用已执行代码的身份，避免“整理代码”悄悄覆盖历史方法。运行时的 `audit.protocol/soft_reference/protection_reference/hard_reference` 可用于绑定具体版本。原源码名称中 `rho_protection` 或注释可能与实际公式不一致，见 PROVENANCE.json。

## 数值与执行约束

一个共享熵对偶求解器处理 IND 和 C，最多四个对偶变量。接受结果必须同时通过原单位可行性残差和原始/对偶目标差检查，不能仅依据 SciPy 的 success 标记。默认残差容差 1e-6，目标差容差 `min(1e-8,tolerance*0.1)`。

第一阶段未认证回 p；第二阶段未认证回 rho。复用 IND 结果时核对完整 p/J_ind/u/M/数值配置签名，防止复用其他前缀的锚点。候选分布只按原候选总质量 lift 回全词表；候选外每项概率原样保留，不进行全词表重新归一化。于是 `KL_full = candidate_mass * KL_candidate`（舍入误差范围内）。

`enable_c=False` 恢复 IND；少于两个不同事件时跳过 C。未知预测不能填 0，调用方应保留 UNKNOWN 并执行其共享回退。精确 q=0/1 在 `candidate_scores` 中保持精确；原预测头的历史 log clamp 保留并在 SOURCE_NOTES.md 披露。

## 文件

| 文件 | 内容 |
|---|---|
| `majc.py` | 评分、两阶段优化、证书、尾部保持与控制接口 |
| `majc_models.py` | 原因果特征编码和可选 PyTorch 轻量头 |
| `test_majc.py` | 解析解、独立原始求解器、保护、回退、输入与支持检查 |
| `example.py` | 可直接运行的合成单步示例 |
| `PROVENANCE.json` | 原文件 SHA256 与提取记录 |
| `SOURCE_NOTES.md` | 版本差异、接入限制和证据边界 |

当前仓库不宣称所有自然语言规则已被执行层覆盖，不保证真实共同成功，不以训练 loss、token 改变次数或数值可行性证明泛化或方法新颖性。

## 发布范围

仅包含方法代码和文档。API 密钥、SSH 密钥、服务器配置、运行账本、原始生成文本、模型权重和数据集均不属于发布内容。未替权利人新增开源许可；公开可见不自动等于授予复用许可。

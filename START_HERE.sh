#!/bin/bash
# IPIP Pipeline - 开始使用指南

cat << 'EOF'

╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║         🚀 IPIP 自动化管道已准备就绪！                                       ║
║         Iterative Pretraining Framework for Interatomic Potentials          ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

📦 已创建的文件清单（共7个新文件，~2000行代码）:

  ✅ run_ipip_pipeline.py (513行)     - 主自动化脚本
  ✅ run_ipip.sh (94行)                - Bash便捷启动脚本
  ✅ ipip_quickstart.py (398行)        - 依赖检查和快速开始工具
  ✅ IPIP_PIPELINE_README.md (450行)   - 完整详细文档
  ✅ QUICK_REFERENCE.txt (197行)       - 快速参考卡
  ✅ IPIP_FILES_SUMMARY.md (316行)     - 文件和功能总结
  ✅ ipip_config_template.json         - 配置文件模板

══════════════════════════════════════════════════════════════════════════════

🎯 三步快速开始:

  第一步: 检查环境准备
  ────────────────────
  python ipip_quickstart.py

  输出内容:
    ✓ 所有依赖检查结果
    ✓ GPU可用性和内存
    ✓ 所有必需文件验证
    ✓ 快速开始建议

  第二步: 准备你的数据
  ────────────────────
  • DFT数据格式: torch_geometric.data.Data 对象列表
  • 保存位置: ./datasets/finetune_data.pt
  • 每个样本需要: {pos, z, energy, force}

  第三步: 运行管道
  ────────────────
  python run_ipip_pipeline.py --iterations 3

  或使用 Bash:
  bash run_ipip.sh -i 3

══════════════════════════════════════════════════════════════════════════════

💡 常用命令汇总:

  基础运行 (3个迭代):
    python run_ipip_pipeline.py --iterations 3

  自定义输出目录:
    python run_ipip_pipeline.py -i 5 -r ./my_results

  使用自定义数据:
    python run_ipip_pipeline.py -i 3 -f ./data/my_dft.pt

  完整参数控制:
    python run_ipip_pipeline.py \
      --iterations 5 \
      --results-dir ./ipip_run_v2 \
      --finetune-data ./data/dft.pt \
      --pretrain-seeds 20 \
      --md-seeds-per-iter 10 \
      --convergence-threshold 0.005

  查看快速参考 (多种方式):
    cat QUICK_REFERENCE.txt                    # 终端查看
    python ipip_quickstart.py --quick-start    # 快速开始
    python ipip_quickstart.py --commands       # 命令参考
    python ipip_quickstart.py --help           # 全部信息

══════════════════════════════════════════════════════════════════════════════

📊 管道工作流程 (3次迭代为例，预计10-20小时):

  ┌─ 初始化 (仅一次)
  │  └─ 生成初始预训练数据 (MD + 教师模型): 2-4 小时
  │
  ├─ 第1次迭代 (3-6 小时)
  │  ├─ 预训练学生模型 (1-2小时)
  │  ├─ 微调学生模型 (1-2小时)
  │  ├─ MD模拟 (1小时)
  │  ├─ 数据重新标签 (自动)
  │  └─ 评估 (自动)
  │
  ├─ 第2次迭代 (同上)
  ├─ 第3次迭代 (同上)
  │
  └─ 完成！所有结果保存在 ipip_results/

══════════════════════════════════════════════════════════════════════════════

📈 预期性能改进 (来自IPIP论文):

  力预测精度:      30-80% 误差降低
  能量预测精度:    20-50% 误差降低
  推理速度:        10-100倍加速
  模拟稳定性:      90%+ 稳定轨迹

══════════════════════════════════════════════════════════════════════════════

📁 输出目录结构:

  ipip_results/
  ├── ipip_config.json              ← 配置 (用于重现)
  ├── ipip_pipeline.log             ← 详细日志
  ├── models/iteration_0*/          ← 训练好的模型
  ├── pretrain_data/iteration_0*/   ← 更新的预训练集
  ├── md_trajectories/iteration_0*/ ← MD模拟数据
  └── metrics/iteration_*_metrics.json ← 性能指标

══════════════════════════════════════════════════════════════════════════════

🔍 实时监控 (在另一个终端):

  查看日志:          tail -f ipip_pipeline.log
  监控GPU:           watch nvidia-smi
  查看性能:          cat ipip_results/metrics/iteration_*.json
  tensorboard观察:   tensorboard --logdir checkpoint/ipip/

══════════════════════════════════════════════════════════════════════════════

⚙️ 关键参数说明:

  --iterations NUM (默认: 3)
    迭代次数。更多=更好，但需要更多时间
    推荐: 3-5 次

  --md-seeds-per-iter NUM (默认: 5)
    每个迭代的MD模拟数。收集OOD样本
    推荐: 5-10 次

  --convergence-threshold FLOAT (默认: 0.01)
    收敛检测阈值。改进 < 此值则停止
    推荐: 0.005-0.02

  --data-retention-rate FLOAT (默认: 0.5)
    更新预训练集时保留多少旧数据
    推荐: 0.4-0.6

  --pretrain-seeds NUM (默认: 10)
    初始数据生成的MD模拟数 (仅运行一次)
    推荐: 10-20 次

══════════════════════════════════════════════════════════════════════════════

🐛 常见问题快速解决:

  Q: "CUDA out of memory"
  A: 减少批大小 (train.py: bz=16) 或使用CPU

  Q: "FileNotFoundError"
  A: 检查 ./datasets/finetune_data.pt 是否存在

  Q: "MD simulations diverge"
  A: 减少时间步长或温度 (Supp_traj_md.py)

  Q: "Force MAE 不改进"
  A: 增加迭代次数或改善微调数据质量

  Q: "管道太慢"
  A: 减少 --md-seeds-per-iter 或 --iterations

══════════════════════════════════════════════════════════════════════════════

📚 文档导航:

  概览和架构 → IPIP_PIPELINE_README.md
  快速开始   → python ipip_quickstart.py
  快速参考   → QUICK_REFERENCE.txt
  文件说明   → IPIP_FILES_SUMMARY.md
  所有帮助   → python ipip_quickstart.py --help

══════════════════════════════════════════════════════════════════════════════

✨ 特点总结:

  ✓ 完全自动化     - 无需手动干预
  ✓ 智能收敛检测   - 自动停止（当改进不足）
  ✓ 详细日志记录   - 每一步都有记录
  ✓ 灵活配置       - 所有参数可调
  ✓ 结果完整保存   - 模型、数据、轨迹、指标都保存
  ✓ 易于分析       - JSON格式结果，按迭代组织
  ✓ GPU加速        - 自动检测和利用
  ✓ 错误恢复       - 详细错误信息和建议

══════════════════════════════════════════════════════════════════════════════

🚀 现在开始吧！

  # 第1步: 检查准备
  python ipip_quickstart.py

  # 第2步: 准备数据
  # (将DFT数据保存为 ./datasets/finetune_data.pt)

  # 第3步: 运行管道
  python run_ipip_pipeline.py --iterations 3

  # 第4步: 监控进度 (在另一个终端)
  tail -f ipip_pipeline.log

══════════════════════════════════════════════════════════════════════════════

💬 需要帮助？

  1. 详细阅读: IPIP_PIPELINE_README.md
  2. 快速查阅: QUICK_REFERENCE.txt
  3. 验证环境: python ipip_quickstart.py
  4. 查看日志: tail -100 ipip_pipeline.log
  5. 查看指标: cat ipip_results/metrics/iteration_*.json

═════════════════════════════════════════════════════════════════════════════

🎉 所有工具已准备完毕！祝你的IPIP研究顺利！

═════════════════════════════════════════════════════════════════════════════

EOF

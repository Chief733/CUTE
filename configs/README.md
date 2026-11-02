# 训练配置

这里保存本项目拥有和维护的训练配置，不修改固定版本的 MindFormers 源码目录。

- `llama2/pretrain_llama2_7b_20steps.yaml`：普通 MindFormers profiler 与 benchmark 基线；从初始化开始采集，开启通信和内存专项。
- `llama2/pretrain_llama2_7b_sidecar_20steps.yaml`：profiler sidecar 配置；前 2 step 热身，从实际第 3 step 开始采集，开启通信、关闭内存专项。

两份配置都在实际第 20 step 停止，模型结构、数据集参数和并行策略保持一致。当前配置保留本机已验证的以下路径，迁移服务器或发布仓库时需要通过部署配置改为目标环境路径：

- `train_dataset.data_loader.dataset_dir`
- `processor.tokenizer.vocab_file`
- `context.compile_cache_path`（仅普通基线显式设置）
- `profile_output`

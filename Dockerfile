# 使用 vLLM 官方映像檔
FROM vllm/vllm-openai:latest

# 安裝加速下載工具 (hf_transfer)
RUN pip install huggingface_hub[hf_transfer]

# 啟用 HF Transfer 加速 (利用多線程最大化頻寬)
ENV HF_HUB_ENABLE_HF_TRANSFER=1

# 建立模型目錄
RUN mkdir -p /model

# 下載 Magnum V4 72B AWQ 模型
# 注意：由於模型高達 40GB+，一般的 Free Build Service (如 Docker Hub / GitHub Actions) 可能會因為超時或硬碟不足而失敗。
# 如果失敗，請考慮使用 RunPod Network Volume 方案。
RUN huggingface-cli download CED6688/magnum-v4-72b-AWQ \
    --local-dir /model/magnum-v4-72b-awq \
    --local-dir-use-symlinks False

# 設定環境變數
ENV MODEL=/model/magnum-v4-72b-awq
ENV SERVED_MODEL_NAME=magnum-v4-72b-awq

# 啟動指令
ENTRYPOINT ["python3", "-m", "vllm.entrypoints.openai.api_server", "--model", "/model/magnum-v4-72b-awq", "--cors-allow-origins", "*"]

运行：

python -m pip install -U "huggingface_hub>=0.20.0"

```bash
# 可选：登录或设置 token（提升下载速率/避免限流）
# huggingface-cli login
# export HF_TOKEN=xxxxxxxx

python - <<'PY'
from huggingface_hub import snapshot_download

repo_id = "mixedbread-ai/mxbai-embed-large-v1"
local_dir = "/scripts/TaiWei/TaiWei-flow-Agent/models/mxbai-embed-large-v1"

path = snapshot_download(
    repo_id=repo_id,
    local_dir=local_dir,
)
print("Downloaded to:", path)
PY
```

验证：
python - <<'PY'
from sentence_transformers import SentenceTransformer
m = SentenceTransformer("/scripts/TaiWei/TaiWei-flow-Agent/models/mxbai-embed-large-v1")
print("OK, dim =", m.get_sentence_embedding_dimension())
PY

后运行：
python3 optimize.py asap7_nangate45_3D gcd ECP 4 
就应该没啥问题了
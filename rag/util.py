import pandas as pd
import numpy as np
try:
    from sentence_transformers import SentenceTransformer, util as st_util  # type: ignore
except ImportError:
    SentenceTransformer = None  # type: ignore
    st_util = None  # type: ignore
import os

class modelUtility:
    def __init__(self, modelName: str):
        self.modelName = modelName
        # ... 把你现有的 systemPrompt 和所有 ragPromptTemplate* 复制进来 ...
        # (直接使用你之前提供的内容)
        if modelName == "OpenROAD-Assistant/Script_Adaptor":
            self.systemPrompt = """You are a tutor specializing in the knowledge of OpenROAD, the open-source EDA tool. You will be asked about general OpenROAD questions and OpenROAD Python API-related questions.
"""
        else:  
            self.systemPrompt = """You are a OpenROAD Python code generator. You will generate the Python code to complete the task. Follow the following guidelines:
1. Only generate Python code, do not include any other text.
2. First generate ```python before the start of the Python code.
3. Generate ``` when finished the Python code.
4. If you don't know the answer, respond with:
```python
```
"""
        if "llama" not in modelName.lower() and "script_adaptor" not in modelName.lower() and "retrained" not in modelName.lower():
            self.ragPromptTemplateWithContext = """<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
Here are the OpenROAD APIs, You do not need to use them unless they are directly relevant to the answer:
=====================
{context}
=====================
Here is your task:
=====================
{question}
=====================

If you define a function, you MUST actually call it in the code.
MUST NOT comment out the code that you write, especially the code you call the function.<|im_end|>
<|im_start|>assistant
"""
            self.ragPromptTemplateWithoutContext = """<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
Here is your task:
=====================
{question}
=====================

If you define a function, you MUST actually call it in the code.
MUST NOT comment out the code that you write, especially the code you call the function.<|im_end|>
<|im_start|>assistant
"""
            self.ragWrongCodePromptTemplateWithContext = """<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
Here is your OpenROAD Python code generation task:
=====================
{question}
=====================

Here is the wrong code you previously generated:
=====================
{wrongCode}
=====================

I got the warning message when running the above wrong code:
=====================
{message}
=====================

Here are some OpenROAD APIs, You do not need to use them unless they are directly relevant to the answer:
=====================
{context}
=====================

The wrong code is not correct, please correct the wrong code or generate a new code to accomplish the OpenROAD Python code generation task.

If you define a function, you MUST actually call it in the code.
MUST NOT comment out the code that you write, especially the code you call the function.<|im_end|>
<|im_start|>assistant
"""
            self.ragWrongCodePromptTemplateWithoutContext = """<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
Here is your OpenROAD Python code generation task:
=====================
{question}
=====================

Here is the wrong code you previously generated:
=====================
{wrongCode}
=====================

I got the warning message when running the above wrong code:
=====================
{message}
=====================

The wrong code is not correct, please correct the wrong code or generate a new code to accomplish the OpenROAD Python code generation task.

If you define a function, you MUST actually call it in the code.
MUST NOT comment out the code that you write, especially the code you call the function.<|im_end|>
<|im_start|>assistant
"""
        else:
            self.ragPromptTemplateWithContext = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{system_prompt}<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Here are the OpenROAD APIs, You do not need to use them unless they are directly relevant to the answer:
=====================
{context}
=====================
Here is your task:
=====================
{question}
=====================

If you define a function, you MUST actually call it in the code.
MUST NOT comment out the code that you write, especially the code you call the function.<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""
            self.ragPromptTemplateWithoutContext = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{system_prompt}<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Here is your task:
=====================
{question}
=====================

If you define a function, you MUST actually call it in the code.
MUST NOT comment out the code that you write, especially the code you call the function.<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""
            self.ragWrongCodePromptTemplateWithContext = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{system_prompt}<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Here is your OpenROAD Python code generation task:
=====================
{question}
=====================

Here is the wrong code you previously generated:
=====================
{wrongCode}
=====================

I got the warning message when running the above wrong code:
=====================
{message}
=====================

Here are some OpenROAD APIs, You do not need to use them unless they are directly relevant to the answer:
=====================
{context}
=====================

The wrong code is not correct, please correct the wrong code or generate a new code to accomplish the OpenROAD Python code generation task.
If you define a function, you MUST actually call it in the code.
MUST NOT comment out the code that you write, especially the code you call the function.<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""
            self.ragWrongCodePromptTemplateWithoutContext = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{system_prompt}<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Here is your OpenROAD Python code generation task:
=====================
{question}
=====================

Here is the wrong code you previously generated:
=====================
{wrongCode}
=====================

I got the warning message when running the above wrong code:
=====================
{message}
=====================

The wrong code is not correct, please correct the wrong code or generate a new code to accomplish the OpenROAD Python code generation task.
If you define a function, you MUST actually call it in the code.
MUST NOT comment out the code that you write, especially the code you call the function.<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""
    def isOpenROADAssistant(self):
        return "OpenROAD-Assistant/Script_Adaptor" == self.modelName

# =======================================================
# 🔹 新增函数：支持从多个 QA 文件中加载合并数据
# =======================================================
def load_multiple_qa_files(base_dir):
    """
    从 base_dir 中读取 Flow.csv, General.csv, Tools.csv 三个文件并合并。
    """
    paths = [
        os.path.join(base_dir, "Flow", "Flow.csv"),
        os.path.join(base_dir, "General", "General.csv"),
        os.path.join(base_dir, "Tools", "Tools.csv"),
    ]

    import glob
    csv_paths = glob.glob(os.path.join(base_dir, "**", "*.csv"), recursive=True)
    csv_paths = sorted(csv_paths)
    if not csv_paths:
        raise FileNotFoundError(f"[ERROR] No CSV files found under {base_dir}")


    dfs = []
    total_rows_before = 0 

    for p in paths:
        if os.path.exists(p):
            try:
                df = pd.read_csv(p)
                dfs.append(df)
                print(f"[INFO] Loaded {p} ({len(df)} rows)")
            except Exception as e:
                print(f"[WARN] Failed to read {p}: {e}")
        else:
            print(f"[WARN] File not found: {p}")

    if not dfs:
        raise FileNotFoundError("[ERROR] No valid QA CSV files found in base_dir")

    df_all = pd.concat(dfs, ignore_index=True)
    print(f"[INFO] Total rows before dedup: {total_rows_before}; concatenated: {len(df_all)}")

    # 不立即 drop_duplicates，先保存原始数，便于调试
    df_clean = df_all.drop_duplicates().fillna("")
    print(f"[INFO] After drop_duplicates & fillna: {len(df_clean)} rows")

    return df_clean



def prepareDocuments(df):
    """
    仅用于 QA 数据集。提取 Prompts 和 Answers 列，组合为文档。
    """
    documents = []
    documentsDict = {}

    if not {"Prompts", "Answers"}.issubset(df.columns):
        raise ValueError("[ERROR] The input DataFrame must contain 'Prompts' and 'Answers' columns.")

    for _, row in df.iterrows():
        q, a = str(row["Prompts"]).strip(), str(row["Answers"]).strip()
        if not q or not a:
            continue
        content = f"Q: {q}\nA: {a}"
        documents.append(content)
        documentsDict[content] = {"Prompts": q, "Answers": a}

    print(f"[INFO] Prepared {len(documents)} QA documents for embedding.")
    return documents, documentsDict



def answerWithRAG(question, embeddings, embeddingModel, allSplits, allDict, topk=10, score_threshold=0.4):
    if embeddingModel is None or st_util is None or embeddings is None:
        return ""
    q_emb = embeddingModel.encode(question, convert_to_tensor=True).cpu()
    # embeddings is a tensor (convert_to_tensor=True recommended)
    scores = st_util.cos_sim(q_emb, embeddings.cpu())  # shape (1, N)
    np_scores = scores.cpu().numpy().flatten()
    top_idx = np.argsort(np_scores)[-topk:][::-1]
    relevantDocs = []
    for idx in top_idx:
        print(f"[DEBUG] RAG candidate {idx} score={np_scores[idx]:.3f}")
        if np_scores[idx] < score_threshold:
            break
        relevantDocs.append(allSplits[idx])
    finalDocs = []
    for doc in relevantDocs:
        finalDocs.append("\n".join(f"# {k} {v}" for k, v in allDict[doc].items()))
    context = ""
    if finalDocs:
        context = "\n\n".join(finalDocs)
    return context

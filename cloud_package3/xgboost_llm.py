# X_Gboost - LLM judge (v13): Qwen2.5-7B-Instruct-AWQ (Apache-2.0, 7.6B <= 8B) reads hard pairs and gives P(same business).
# Hard pairs = cross-encoder score in [0.1, 0.9] (same rule on validation and test). Needs GPU T4 x2 + Internet ON.
# Outputs /kaggle/working/llm_val.parquet, llm_test.parquet  (qid, llm)
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "vllm"], check=False)
import glob, math, time, numpy as np, polars as pl, torch
from vllm import LLM, SamplingParams
from sklearn.metrics import roc_auc_score
MODEL = "Qwen/Qwen2.5-7B-Instruct-AWQ"
f = lambda n: glob.glob(f"/kaggle/input/**/{n}", recursive=True)[0]
val = pl.read_parquet(f("llm_val.parquet")); test = pl.read_parquet(f("llm_test.parquet"))
print("val", val.height, "test", test.height, torch.cuda.device_count(), "GPU(s)", flush=True)
SYSTEM = ("You match business records. Each record is 'name | address'. Records can be noisy: typos, abbreviations, "
          "Hindi transliteration, leetspeak (0/o, 1/l, 5/s), missing or partial address, a trade name (DBA) instead of the "
          "legal name. They are DIFFERENT businesses when: the legal form conflicts (Inc vs LLC, Pvt Ltd vs LLP), the house "
          "number differs, a business word is added (Pvt, Holdings, Group, Ventures, Exports, Industries), or a name word is "
          "swapped for another. Answer only Yes (same business) or No.")
llm = LLM(MODEL, quantization="awq", dtype="half", tensor_parallel_size=torch.cuda.device_count(), max_model_len=512,
          enable_prefix_caching=True, gpu_memory_utilization=0.90)
tok = llm.get_tokenizer()
YES = {tok.encode(w, add_special_tokens=False)[0] for w in ("Yes", " Yes", "yes")}
NO = {tok.encode(w, add_special_tokens=False)[0] for w in ("No", " No", "no")}
sp = SamplingParams(max_tokens=1, temperature=0, logprobs=20)

def prompts(df):
    return [tok.apply_chat_template([{"role": "system", "content": SYSTEM},
                                     {"role": "user", "content": f"Record A: {a[:200]}\nRecord B: {b[:200]}\nSame business?"}],
                                    tokenize=False, add_generation_prompt=True) for a, b in zip(df["qt"], df["st"])]

def judge(df, name):
    out = np.full(df.height, 0.5, np.float32); P = prompts(df); t0 = time.time(); CH = 20000
    for i in range(0, len(P), CH):
        res = llm.generate(P[i:i+CH], sp, use_tqdm=False)
        for j, r in enumerate(res):
            lp = r.outputs[0].logprobs[0] if r.outputs[0].logprobs else {}
            py = sum(math.exp(v.logprob) for k, v in lp.items() if k in YES)
            pn = sum(math.exp(v.logprob) for k, v in lp.items() if k in NO)
            out[i+j] = py / (py + pn) if py + pn > 0 else 0.5
        print(f"{name} {i+len(res)}/{len(P)}  {(i+len(res))/(time.time()-t0):.1f} pairs/s", flush=True)
    return out

vp = judge(val, "val")
print(f"VALIDATION AUC (LLM alone, hard pairs): {roc_auc_score(val['label'].to_numpy(), vp):.4f}", flush=True)
val.select("qid").with_columns(pl.Series("llm", vp)).write_parquet("/kaggle/working/llm_val.parquet")
tp = judge(test, "test")
test.select("qid").with_columns(pl.Series("llm", tp)).write_parquet("/kaggle/working/llm_test.parquet")
print("DONE")

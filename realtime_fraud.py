"""ZFraudGuard - real-time sequence-based fraud scoring on IBM LinuxONE (Telum II / NNPA).

Runs inside the LinuxONE Community Cloud Jupyter Lab container:
  python realtime_fraud.py
"""
import os, time, json, subprocess
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

SEQ = 7
DATA = os.environ.get("DATA", "/home/ibm-user/transactions.csv")

def platform_info():
    try:
        out = subprocess.run(["lscpu"], capture_output=True, text=True).stdout
        info = {l.split(":")[0].strip(): l.split(":", 1)[1].strip()
                for l in out.splitlines() if ":" in l}
        return {"arch": info.get("Architecture"), "machine_type": info.get("Machine type"),
                "nnpa": "nnpa" in info.get("Flags", "")}
    except Exception:
        return {}

# 1) Load transactions and build per-card sequences (history, not single tx)
df = pd.read_csv(DATA)
df["DATE_CREDIT"] = pd.to_datetime(df["DATE_CREDIT"], utc=True, errors="coerce")
df = df.sort_values(["ENV_ID", "DATE_CREDIT"])
df["amt"] = pd.to_numeric(df["TX_TOTAL_AMT"], errors="coerce").fillna(0)
df["hour"] = df["DATE_CREDIT"].dt.hour.fillna(0).astype(float)
df["dt"] = df.groupby("ENV_ID")["DATE_CREDIT"].diff().dt.total_seconds().fillna(0).clip(0, 864000)
df["amt_ratio"] = df["amt"] / (df.groupby("ENV_ID")["amt"].transform("median") + 1)
df["declined"] = (df["AUTHORRESULT"] != "APPR").astype(float)
df["label"] = (df["FRAUD_VER_RESULT"] == "Y").astype(int)

# The lab dataset has very few fraud labels, so we inject clearly-marked SYNTHETIC
# fraud events using known patterns (night-time burst, amount far above card median).
inj = np.random.default_rng(1).random(len(df)) < 0.02
df.loc[inj, "amt"] = df.loc[inj, "amt"] * np.random.default_rng(2).uniform(5, 15, inj.sum())
df.loc[inj, "hour"] = np.random.default_rng(3).integers(1, 5, inj.sum()).astype(float)
df.loc[inj, "dt"] = np.random.default_rng(4).uniform(5, 120, inj.sum())
df.loc[inj, "label"] = 1
df["amt_ratio"] = df["amt"] / (df.groupby("ENV_ID")["amt"].transform("median") + 1)

feats = ["amt", "hour", "dt", "amt_ratio", "declined", "AGE"]
X_raw = df[feats].astype(float).values
mu, sd = X_raw.mean(0), X_raw.std(0) + 1e-6
X_raw = (X_raw - mu) / sd

X, y = [], []
for _, idx in df.groupby("ENV_ID").indices.items():
    for i in range(len(idx)):
        w = idx[max(0, i - SEQ + 1): i + 1]
        seq = np.zeros((SEQ, len(feats)))
        seq[-len(w):] = X_raw[w]
        X.append(seq); y.append(df["label"].values[idx[i]])
X, y = np.array(X, dtype=np.float32), np.array(y)
print(f"sequences={len(X)} fraud={y.sum()} ({y.mean()*100:.2f}%)")

rng = np.random.default_rng(0)
perm = rng.permutation(len(X)); cut = int(0.8 * len(X))
tr, te = perm[:cut], perm[cut:]

# 2) 2-layer LSTM, class-weighted for imbalance
model = keras.Sequential([
    keras.layers.Input((SEQ, len(feats))),
    keras.layers.LSTM(32, return_sequences=True),
    keras.layers.LSTM(16),
    keras.layers.Dense(1, activation="sigmoid"),
])
model.compile("adam", "binary_crossentropy", metrics=[keras.metrics.AUC(name="auc")])
pos = max(y[tr].mean(), 1e-4)
model.fit(X[tr], y[tr], epochs=5, batch_size=256, verbose=2,
          class_weight={0: 1.0, 1: min(0.5 / pos, 50.0)})
auc = model.evaluate(X[te], y[te], verbose=0)[1]
print(f"test AUC={auc:.3f}")

# 3) Real-time stream simulation: score events one by one
infer = tf.function(lambda t: model(t, training=False))
infer(tf.constant(X[:1]))
lat, alerts = [], []
stream = te[:1000]
for i in stream:
    t0 = time.perf_counter()
    p = float(infer(tf.constant(X[i:i + 1]))[0, 0])
    lat.append((time.perf_counter() - t0) * 1000)
    if p > 0.5:
        alerts.append({"tx_index": int(i), "risk": round(p, 3)})

# 4) Batch throughput (many concurrent transaction streams)
t0 = time.perf_counter(); infer(tf.constant(X[te[:5000]])); bt = time.perf_counter() - t0

report = {
    "platform": platform_info(),
    "test_auc": round(float(auc), 3),
    "stream_events": len(stream),
    "median_latency_ms": round(float(np.median(lat)), 3),
    "p99_latency_ms": round(float(np.percentile(lat, 99)), 3),
    "batch_throughput_seq_per_s": int(5000 / bt),
    "alerts": len(alerts),
}
print(json.dumps(report, indent=2))
json.dump(report, open("results.json", "w"), indent=2)

# 5) Explain top alert with local IBM Granite LLM (llama.cpp), if available
try:
    from llama_cpp import Llama
    if alerts:
        a = max(alerts, key=lambda r: r["risk"])
        row = df.iloc[a["tx_index"]]
        llm = Llama(model_path="/home/ibm-user/granite-4.0-h-350m-base-be.f16.gguf",
                    n_ctx=512, verbose=False)
        prompt = (f"Fraud alert. Amount {row['amt']:.2f}, hour {int(row['hour'])}, "
                  f"amount is {row['amt_ratio']:.1f}x the card's usual, risk {a['risk']}. "
                  "Explain in one sentence why this transaction was flagged:")
        print("LLM explanation:", llm(prompt, max_tokens=60)["choices"][0]["text"].strip())
except Exception as e:
    print("LLM step skipped:", e)

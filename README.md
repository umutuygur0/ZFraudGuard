# ZFraudGuard — Real-Time AI Fraud Scoring on IBM LinuxONE

**IBM Z Datathon 2026 — Theme: Real-Time AI for Critical Decisions**

## Problem
Card fraud and payment scams cause billions in losses, often hitting vulnerable people (students, elderly users, small merchants). Many systems score transactions in batches or with high latency, so money is gone before an alert fires. Flagged transactions usually come with a bare score and no explanation.

## Solution
A hybrid predictive + generative AI pipeline running entirely on IBM LinuxONE:

1. **Streaming simulator**: replays card transactions event by event.
2. **Sequence model**: each card's last 7 transactions go into a 2-layer LSTM (Keras/TensorFlow), so the model scores behaviour history rather than a single transaction.
3. **Real-time alerting**: high-risk events raise an alert within milliseconds.
4. **Explainability**: a local IBM Granite LLM (llama.cpp) writes a one-sentence reason for the alert. No data leaves the platform.

```
transactions --> per-card sequence (7) --> LSTM on Telum II (NNPA) --> risk score
                                                         |
                                            risk > 0.5 --> alert --> Granite LLM explanation
```

## IBM Z / LinuxONE usage
- Runs on a LinuxONE Community Cloud VM: **IBM z17 (machine type 9175), s390x, Ubuntu 24.04, Docker**.
- The on-chip **Telum II AI accelerator (NNPA)** is enabled and used for TensorFlow inference (`lscpu` shows the `nnpa` flag).
- This follows the IBM Z **Transactional AI** pattern: AI runs next to the transaction data instead of shipping it to an external cloud.

## Results measured on LinuxONE (3 vCPU + NNPA)
| Metric | Value |
|---|---|
| Platform | s390x, machine type 9175, nnpa = true |
| Streaming latency (1 event), median | **1.37 ms** |
| Streaming latency, p99 | **2.19 ms** |
| Batch throughput | **~26,500 sequences/s** |
| Test AUC (with synthetic fraud injection, see note) | 0.999 |
| Alerts in 1,000 streamed events | 19, each explained by the local Granite LLM |

The high AUC reflects the simple synthetic fraud patterns; it is a pipeline demo, not a production accuracy claim.

## Data note
Transactions come from the IBM LinuxONE Jupyter Lab ML sample data. That sample contains very few fraud labels, so `realtime_fraud.py` **injects clearly marked synthetic fraud events** (2%: night-time bursts, amounts 5–15x the card median) to demonstrate the pipeline. A production version would train on the full IBM synthetic credit-card dataset (24M records).

## Run
Inside the LinuxONE Community Cloud Jupyter Lab container:
```bash
python realtime_fraud.py   # prints metrics and writes results.json
```

## Who benefits
Bank customers (especially vulnerable groups), small merchants and fraud analysts. Fraud is stopped before the payment completes, and each alert comes with a plain-language reason.

## Open-source resources
- https://github.com/linuxone-community-cloud/jupyter-lab-ml
- https://www.tensorflow.org, https://keras.io, https://onnx.ai
- https://github.com/ggml-org/llama.cpp, https://huggingface.co/ibm-granite
- https://pandas.pydata.org, https://numpy.org

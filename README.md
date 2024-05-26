# smart-glasses-llms
Using Foundation Models for QoS aware decision making

Next steps:
- [x] Add k-fold cross validation 
- [x] Add optuna 
- [x] Develop Early-Exit Model 
- [x] Modify timings
- [x] Add inference time to latency
- [ ] Visualization of predictions and classification scores
- [ ] Implement Baselines.
- [ ] Sensitivity analysis (20% threshold, QoS weights)

# Training

```python3
python3 train_llm.py --model FCNNet
python3 train_llm.py --model EENet

python3 train_rl --model 
```

# Next steps

1. Description of baseline adaptations.
2. Distribution of Early exits.
3. Generalization tests.
4. Memory and compute load. 
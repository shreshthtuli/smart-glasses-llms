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
python3 train_llm.py --model BranchyNet
python3 train_llm.py --model ZTWNet

python3 train_rl --model 
```

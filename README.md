# SMPO

## Install
```
conda create -n smpo python=3.9 -y
conda activate smpo
pip install torch gymnasium safety-gymnasium numpy pyyaml tensorboard
```

## Run
Default config file with targeted environments:
```
python -m smpo.scripts.train_smpo \
  --config smpo/configs/config.yaml \
  --env_id SafetyPointGoal1-v0 \
  --eval_every 1 \
  --eval_episodes 5
```
Override config file parameters:
```
python -m smpo.scripts.train_smpo \
  --config smpo/configs/config.yaml \
  --env_id SafetyPointGoal1-v0 \
  --eval_every 1 \
  --eval_episodes 5 \
  --set steps_per_epoch=10000 epochs=80 lr=0.0001 d=20 b=2.5
```

## Launch TensorBoard to monitor training (policy loss/value loss/weight means etc.):
```
tensorboard --logdir runs
```


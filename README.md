# uav-localization-experiments


### Start

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python main.py fit --conf conf/config.yaml # Modify conf based on your requirements
```


### Predict from checkpoint

```bash
python main.py predict --conf conf/config.yaml --ckpt_path=<path>
```

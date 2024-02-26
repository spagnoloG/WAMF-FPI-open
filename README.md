# WAMF-FPI-open 


Open Source Implementation of the [WAMF-FPI](https://www.mdpi.com/2072-4292/15/4/910) algorithm in Python. 
Including additional features and continious improvements.

## Start (OldSchool way) ~ Not recommended

```bash
sudo add-apt-repository ppa:ubuntugis/ppa -y && sudo apt update -y
sudo apt install gdal-bin libgdal-dev -y  # Hopefully this will work
sudo apt install python3.11 python3.11-venv git  -y
export CPLUS_INCLUDE_PATH=/usr/include/gdal
export C_INCLUDE_PATH=/usr/include/gdal
pip install GDAL --global-option=build_ext --global-option="-I/usr/include/gdal" --global-option="-lgdal"
git clone https://github.com/spagnoloG/WAMF-FPI-open.git
cd WAMF-FPI-open
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

## Start (nix way) ~ Ubuntu

- Install nix package manager
```bash
sudo apt install curl -y
sh <(curl -L https://nixos.org/nix/install) --daemon
```

- Enable nix-flakes

```bash
echo "experimental-features = nix-command flakes" | sudo tee -a /etc/nix/nix.conf > /dev/null
```

- Clone the repo and enter development environment

```bash
nix develop github:spagnoloG/WAMF-FPI-open
```

## Start ~ NixOS 

```bash
nix develop github:spagnoloG/WAMF-FPI-open
```

## After setting up

To automatically enter the environment when you `cd` into project, run the following command:

```bash
sudo apt install -y direnv # or nix-shell -p direnv
direnv allow
```

## Train

```bash
python code/main.py fit --conf code/conf/config.yaml # Modify conf based on your requirements
```

## Predict from checkpoint

```bash
python code/main.py predict --conf code/conf/config.yaml --ckpt_path=<path>
```

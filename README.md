# Few-Shot Console
![The GUI](fsc.jpg)
This repository contains a simple inference script for the models of the paper TODO and a GUI that allow to easily collect new demonstrations and to modify known classes.

### Setup
Clone the few-shot-console repository:
```
git clone https://github.com/hsp-iit/few-shot-console.git
```
Install Conda, then create a Conda environment with:
```
conda env create -f environment.yml
```
NOTE: environment can be simplified. (TODO)

Download the model checkpoints depending on your target settings:
- Table top:  (TODO) the checkpoint from SSv2
- Human-Robot Interaction: download (TODO) the checkpoint from NTURGBD120

### Launch
If needed, adjust the setting in config.json.
Within the conda environment activated, launch
```
python demo.py
```
Note that the demo.py script search for available cameras with OpenCV and then select the last one as input source.
Modify this behaviour if needed.
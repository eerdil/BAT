# **Predicting Standardized Uptake Value of Brown Adipose Tissue from CT scans using convolutional neural networks**

**Ertunc Erdil, Anton S. Becker, Moritz Schwyzer, Borja Martinez-Tellez, Jonatan R. Ruiz, Thomas Sartoretti, H. Alberto Vargas, A. Irene Burger, Alin Chirindel, Damian Wild, Nicola Zamboni, Bart Deplancke, Vincent Gardeux, Claudia Irene Maushart, Matthias Johannes Betz, Christian Wolfrum, Ender Konukoglu**

![CT Scan Example](path/to/image.png)

## **How to run code:**

### **1 - Installing dependencies:**
```bash
pip install -r /path/to/requirements.txt
```

### **2 - Running training code:**
```bash
python train.py --config_path <config_path>
```

Please replace `<config_path>` with the one prepared for your data. One can use the example config file in `./config/Granada/granada_cfg.py.` Please replace the data paths with the paths of your data.

The imaging data that we used in our experiments are not publicly available due to privacy reasons. Please see the data availability statement in our paper.

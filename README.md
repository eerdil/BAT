# **Predicting Standardized Uptake Value of Brown Adipose Tissue from CT scans using convolutional neural networks**

**Ertunc Erdil, Anton S. Becker, Moritz Schwyzer, Borja Martinez-Tellez, Jonatan R. Ruiz, Thomas Sartoretti, H. Alberto Vargas, A. Irene Burger, Alin Chirindel, Damian Wild, Nicola Zamboni, Bart Deplancke, Vincent Gardeux, Claudia Irene Maushart, Matthias Johannes Betz, Christian Wolfrum, Ender Konukoglu**

![ Illustration of the flow for predicting PET activity of BAT from CT scans and segmenting the active BAT region. (a) Illustration of cropping to obtain a region of interest (ROI) that contains the supraclavicular region. Note that C indicates the number of slices in the axial dimension and can slightly change for different subjects. After cropping, the slices are given as input to the CNN shown in (b). (b) Schematic of the Attention U-Net architecture. (c) Detecting active BAT regions from a PET volume. Note that “AND” represents the logical and operator that we used to mask out false positive regions obtained after thresholding.](figure.png)

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

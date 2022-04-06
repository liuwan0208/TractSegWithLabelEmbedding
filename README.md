## 1 Paper
Liu, Wan, Qi Lu, Zhizheng Zhuo, Yuxing Li, Yunyun Duan, Pinnan Yu, Liying Qu, Chuyang Ye, and Yaou Liu. "[Volumetric Segmentation of White Matter Tracts with Label Embedding.](https://www.sciencedirect.com/science/article/pii/S1053811922000635)" NeuroImage (2022), 250: 118934.


## 2 Prerequisites
### 2.1 Environment and Software
* Linux & OSX, Python>=3.6
* [Pytorch](https://pytorch.org/)
* [Mrtrix 3](https://mrtrix.readthedocs.io/en/latest/installation/build_from_source.html) (>=3.0 RC3)
* [FSL](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FslInstallation) 
### 2.2 Install Baseline Code and BatchGenerators
* We use [TractSeg](https://github.com/MIC-DKFZ/TractSeg/) as the baseline, and install it from local source code:
```
git clone https://github.com/MIC-DKFZ/TractSeg.git
pip install -e TractSeg
```
* Install `BatchGenerators`:
```
git clone https://github.com/MIC-DKFZ/batchgenerators.git
pip intall -e batchgenerators
```
* Create a file `~/.tractseg/config.txt`, and write the path of your own directory in config.txt, e.g. `working_dir=Your_OutputPath`.


## 3 Run Our Code
### 3.1 Download Code
* Download our code as zip, and unzip it.
* Save our code in the same directory of TractSeg code, i.e. `Your_CodePath`.
### 3.2 Data Preparation
* Download the [HCP scans](https://db.humanconnectome.org) and the [gold standard of WM tracts](https://db.humanconnectome.org).
* Extract the input peaks images from dMRI scans with 'Your_CodePath/TractSegWithLabelEmbedding/`bin/Generate_Peaks.py`'.
* Arrange the peaks and annotations of different subjects to the following structure -> `used for network testing`:
```
Your_DataPath/HCP_for_training_COPY/subject_01/
            '-> mrtrix_peaks.nii.gz       (mrtrix CSD peaks;  shape: [x,y,z,9])
            '-> bundle_masks.nii.gz       (Reference bundle masks; shape: [x,y,z,nr_bundles])
Your_DataPath/HCP_for_training_COPY/subject_02/
      ...
```
* Remove the non-brain area of data in `HCP_for_training_COPY` fold with 'Your_CodePath/TractSegWithLabelEmbedding/`bin/Remove_Nonbrain.py`', and arrange the data to the following structure -> `used for network training`:
```
Your_DataPath/HCP_preproc/subject_01/
            '-> mrtrix_peaks.nii.gz       (mrtrix CSD peaks;  shape: [x,y,z,9])
            '-> bundle_masks.nii.gz       (Reference bundle masks; shape: [x,y,z,nr_bundles])
Your_DataPath/HCP_preproc/subject_02/
      ...
```
* Adapt 'Your_CodePath/TractSegWithLabelEmbedding/`tractseg/libs/system_config.py`' and modify `DATA_PATH` to 'Your_DataPath/HCP_preproc'.
* Adapt 'Your_CodePath/TractSegWithLabelEmbedding/`tractseg/data/subjects.py`' with the list of your subject IDs.
### 3.3 Train the Network
* Set the temporary enviroment variable in terminal to our code path:
```
export PYTHONPATH=$PYTHONPATH:Your_CodePath/TractSegWithLabelEmbedding
```
* `Train` the network:
```
python run Your_CodePath/TractSegWithLabelEmbedding/bin/ExpRunner
```
* The `training output` is saved in 'Your_OutputPath/hcp_exp/my_custom_experiment'.
### 3.4 Test the Trained Model
* Set the temporary enviroment variable in terminal to our code path:
```
export PYTHONPATH=$PYTHONPATH:Your_CodePath/TractSegWithLabelEmbedding
```
* You can `directly test` the network using the provided data in the `example` fold and network model ([link1](https://drive.google.com/file/d/1O-DPM0vBV5Z58Bqt0_dC0kzqeThjSTjV/view?usp=sharing) & [link2](https://pan.baidu.com/s/1EjvdWTomN6D3inC7wrSpaw?pwd=5gv3)) after modifying the corresponding paths in 'Your_CodePath/TractSegWithLabelEmbedding/bin/`ExpRunner_test.py`':
```
python run Your_CodePath/TractSegWithLabelEmbedding/bin/ExpRunner_test.py
```
* `Test` the network with `your trained model` and test data after modifing the corresponding paths in ExpRunner_test.py.


---
`Note` that we did experiments through modificating the [TractSeg-v2.0](https://github.com/MIC-DKFZ/TractSeg/releases/tag/v2.0), and the main modifications are listed as follows:<br>
1) Add extra scripts to prepare data as in `3.2 Data preparation` and test model as in 'TractSegWithLabelEmbedding/`bin/ExpRunner_test.py`'. <br>
2) Add extra scripts to define network as in 'TractSegWithLabelEmbedding/tractseg/models/`unet_pytorch_deepup_simp1.py`' (used for network training) and 'TractSegWithLabelEmbedding/tractseg/models/`unet_pytorch_deepup_simp1_test.py`' (used for network testing). <br>
3) Modify the loss settings as in 'TractSegWithLabelEmbedding/`tractseg/libs/trainer.py`' and 'TractSeg_labelembed/`tractseg/models/base_model.py`'.








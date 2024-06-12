# GADE-mindspore
MindSpore framework implementation for Ambiguous Entity Oriented Targeted Document Detection

## Dependencies
* Python == 3.7.5
* mindspore == 2.1.0
* mindnlp == 0.2.0


### Datasets
We construct four labeled datasets for the targeted document detection task, i.e., `Wiki-100`, `Wiki-200`, `Wiki-300`, and `Web-Test`. The former three datasets are
constructed from Wikipedia and the last dataset is constructed from Web documents.

* The four labeled datasets `Wiki-100`, `Wiki-200`, `Wiki-300`, and `Web-Test` are placed in the `datasets` folder. Please unzip `Wiki100.zip`, `Wiki200.zip`, `Wiki300.zip`, and `Web_Test.zip` under `datasets/`.

### Usage

##### Run the main code (**GADE**):

* python train.py --model_name GADE_100 --data_type Wiki100

* python train.py --model_name GADE_200 --data_type Wiki200

* python train.py --model_name GADE_300 --data_type Wiki300



For more details about the data set and the experiment settings, please refer to our paper.

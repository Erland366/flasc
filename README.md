# flasc

This repository contains code for "[Federated LoRA with Sparse Communication](https://arxiv.org/abs/2406.05233)" .

We use PyTorch and Huggingface Transformers to simulate federated training of LoRA. To begin, install CUDA 11 or later, Python 3.10.14, and Anaconda / Miniconda on the host machine.

# Envrionment Setup
```
conda create --name flasc python=3.10.14
```

We install all the packages manually using pip.

```
conda activate flasc
```

```
python -m pip install tqdm scikit_learn numpy scipy torch torchvision torchaudio tensorflow_cpu peft transformers python-dotenv loguru
```


The code provides utilities to automatically download the CIFAR10 and 20NewsGroups datasets. We will provide full details for working with the other datasets once the paper is made public.

To train a global LoRA module, run the ```train_sparse_lora_dp.py``` script.

To reproduce the results on systems heterogeneity, we provide the script ```train_sparse_lora_het.py```. The script hard-codes tiers of per-round communication capability in powers of 4.

# Run One Experiment
To run one of the experiment, you can do
```
python train_sparse_lora_vanilla.py --config fedavg
```

Which the options are available at `configs/` folder

# Run Experiment

To run many experiment at the same time, you can do

```
python run_experiments.py --config-list experiments.yaml
```

Which it'll use `experiments.yaml` config that uses configs file that is available at `configs/`

# Update (4/4/2025)
## Running Script
Running the following command after uncommenting all configs in `experiments.yaml` can reproduce similar the results shown in our midterm presentation
    ```
    python run_experiments.py --config-list experiments.yaml
    ```
## File Structure
- `train_sparse_lora_vanilla.py`: The main script to run the experiments
- `configs/` stores the configuration files for the experiments. 
- `experiments.yaml` stores the configs for experiments when we run the command `python run_experiments.py --config-list experiments.yaml`.
- `merging.py` includes implementations of the merging algorithms
- `model_merging_methods/` stores merging methods from the [DARE repo](https://github.com/yule-BUAA/MergeLM/tree/main/model_merging_methods). Some
of our methods directly use the APIs. They serve as a reference for the others.
- `playfround.ipynb` is a jupyter notebook that provides a basic setup for finetuning and merging the models (in the FL setting). It can be used to conduct study in simple settings and can help to understand the code in `train_sparse_lora_vanilla.py`.

## Configuration
 1. *Change merging strategy*: change `merging_strategy` in the config file. Options are `[average, task_arithmetic, fisher_merging, regmean_merging, ties_merging]`.

 2. *Use FedProx*: set `fl_strategy` to be `fedprox` in the config file.

 3. *Change the server optimizer (i.e., FedOPT)*: change `server-opt` in the config file. Options are `[sgd, sgdm, adagrad, adam, yogi]`. For `sgd` and `sgdm`, learning rate is set to be 1 to follow the vanilla FedAvg.

 4. *Change the non-iid level*: change `iid-alpha` in the config file from 0 to 1. The larger the value, the more non-iid the data is.

## Need to improve
1. The `ties_merging` method doesn't have good performance at this point. Might be due to improper implementation of the method, alghoutgh it's based the API.

2. `dare_merging` or `mask_merging` is not supported yet.

3. (Future work) Improve the code structure and documentation.

## TODO List:

### Main Experiments
- Preparations

    - Check the reason why `ties_merging` not working well.

    - Integrate the `dare_merging` or `mask_merging` to our code (the reference code is also in `model_merging_methods/merging_methods.py`)

- First Experiment
    - Check the results when combining all the optimizers, fl strategy, and merging strategies.
    - Question to answer:
        - Does the *final accuracy* improve when using advanced merging stragegies?
        - Does the training *converge faster* when using advanced merging strategies?

- Second Experiment
    - Change the hyperparameters in the FL setting, such as the number of local training epochs and non-iid level. Check the results in these situations.
    - Question to answer:
        - Do we see the same trend as the first experiment? E.g., we might see that advanced merging strategies dominate when the non-iid level is high and the number of local training epochs is large.
        - Does having more local training epochs help the model converge faster? This serves as one of the motivation for using the advanced merging strategies.

### Preliminary Experiment
Consider a simple setting, e.g., based on the one in `playground.ipynb`, to study the merging strategies in FL. We can record some statistics here to support our main experiments. It also serves as the motivation for using advanced merging strategies.
- For instance, we can check:
    - The distance between different clients when changing the number of local training epochs and level of non-iidness.
    - In a single round, how much improvement we can get by using advanced merging strategies.
    - Does the above improvement decrease as the training goes on?
    - To have some specific examples showing that the vanilla FedAvg doesn't work well in such non-iid settings.
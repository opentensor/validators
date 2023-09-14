<div align="center">

# **Open Validators** <!-- omit in toc -->
[![Discord Chat](https://img.shields.io/discord/308323056592486420.svg)](https://discord.gg/bittensor)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 

---

</div>

This repository contains Bittensor Validators designed by the OpenTensor Foundation team for the community.
It offers several functionalities, such as:

- Building and running Bittensor validators
- Real-time analysis of validator performance integrated with wandb
- Offline analysis of data generated from the network
- Creation of datasets using network data for training miners 

The main goal of this repository is to facilitate the interaction with the Bittensor network by providing a set of
open-source validators to the community. The current validator implementation queries the network for responses and 
evaluations using carefully crafted prompts using CoT, that are later evaluated by a pipeline of reward functions, including diversity, relevance, rlhf, among others.

Additionally, the repository provides an analysis and data toolkit that allows users to analyze the data generated from
the validator's interaction with the network. By default, the validator collects various data points, such as question 
responses, evaluations, rewards and scorings by UID, and model performance data. This data is then sent to wandb, 
making it publicly accessible to the community.

The toolkit also includes scripts to analyze and extract data from specific validator runs or multiple runs, simplifying
the creation of valuable datasets for the community's miners.

To learn more about the Bittensor validation process, check out this [documentation](https://tensor-wiki.vercel.app/validating/validating).

# Running

These validators are designed to run and update themselves automatically. To run a validator, follow these steps:

1. Install this repository, you can do so by following the steps outlined in [the installation section](#install).
2. Install [Weights and Biases](https://docs.wandb.ai/quickstart) and run `wandb login` within this repository. This will initialize Weights and Biases, enabling you to view KPIs and Metrics on your validator. (Strongly recommended to help the network improve from data sharing)
3. Install [PM2](https://pm2.io/docs/runtime/guide/installation/) and the [`jq` package](https://jqlang.github.io/jq/) on your system.
   **On Linux**:
   ```bash
   sudo apt update && sudo apt install jq && sudo apt install npm && sudo npm install pm2 -g && pm2 update
   ``` 
   **On Mac OS**
   ```bash
   brew update && brew install jq && brew install npm && sudo npm install pm2 -g && pm2 update
   ```
4. Run the `run.sh` script which will handle running your validator and pulling the latest updates as they are issued. 
   ```bash
   pm2 start run.sh --name openvalidators_autoupdate -- --wallet.name <your-wallet-name> --wallet.hotkey <your-wallet-hot-key>
   ```

This will run **two** PM2 process: one for the validator which is called `auto_run_validator` by default (you can change this in `run.sh`), and one for the run.sh script (in step 4, we named it `validator_maintainer`). The script will check for updates every 30 minutes, if there is an update then it will pull it, install it, restart `auto_run_validator` and then restart itself.

# Usage
There are currently four main avenues for engaging with this repository:

1. [Validators](#Validators):
   - Designed for TAO holders who aim to build or run validators developed by the foundation.

2. [Real-time monitoring with wandb integration](#Real-time-monitoring-with-wandb-integration):
   - Allows users to analyze the performance of various validators runs in real-time using wandb.

3. [Network analysis](#Network-analysis)
   - Caters to individuals, researchers, and data scientists interested in analyzing the data generated from the validators' interaction with the Bittensor network.

4. [Dataset creation](#Dataset-creation)
   - Serves individuals, researchers, and developers who seek to create datasets for the community's miners.

# Install
From source:
```bash
$ git clone https://github.com/opentensor/validators.git
$ pip3 install -e openvalidators/
```

You can test the installation by running the following command:
```bash
$ python3 validators/openvalidators/neuron.py --help
```

# Validators
Participation in Network Validation is available to TAO holders. The validation mechanism utilizes a dual proof-of-stake and proof-of-work system known as *Yuma Consensus*, which you can learn more about [here](https://tensor-wiki.vercel.app/validating/validating). To start validating, you will need to have a Bittensor wallet with a sufficient amount of TAO tokens staked.

Once you have your wallet ready for validation, you can start the foundation validator by running the following command:
```bash
$ python3 validators/openvalidators/neuron.py --wallet.name <your-wallet-name> --wallet.hotkey <your-wallet-hot-key>
```

# Real-time monitoring with wandb integration
By default, the validator sends data to wandb, allowing users to monitor running validators and access key metrics in real time, such as:
- Gating model loss
- Hardware usage
- Forward pass time
- Block duration

All the data sent to wandb is publicly available to the community at the following [link](https://wandb.ai/opentensor-dev/openvalidators).

You don't need to have a wandb account to access the data or to generate a new run,
but bear in mind that
[data generated by anonymous users will be deleted after 7 days](https://docs.wandb.ai/guides/app/features/anon#:~:text=If%20there's%20no%20account%2C%20we,be%20available%20for%207%20days)
as default wandb policy.

# Network analysis
This repository provides a set of tools to analyze the data generated by the validators, including:
- Completions 
- Rewards
- Weights
- [Prompt scoring](#Prompt-based-scoring)

A basic tutorial for downloading and analyzing wandb data can be found in [analysis](./analysis/demo.ipynb).

# Dataset creation
For the individuals who are eager to create datasets tailored specifically for the community's miners.
With convenient scripts available in the [scripts](./scripts) folder, you can effortlessly download data from specific or multiple runs 
of wandb, empowering you to curate comprehensive and valuable datasets that align with your mining objectives.
Check the [README of the data collector](./scripts/README.md) for more information.

----
## Experimental Features

## Sentence Embedding Gating Model
Another cornerstone of the validator functionality is the use of a mixture of experts (MoE) model, which we call the gating model, to enable queries to be efficiently routed to the best-suited miners. **This incentivizes miners to become specialists, which in turn improves response quality**. It also reduces latency and addresses bandwidth issues in the network.
We are working on a new and improved gating model, based on sentence embeddings, which is expected to be a more powerful and robust router for queries. By default it is disabled, but can be enabled with the flags

```--neuron.use_custom_gating_model --gating.model_name sentence-transformers/all-distilroberta-v1```

## CUDA device placement
If you desire to place your validator on a specific GPU, it is recommended to prepend the command you are using to start and run your validator with `CUDA_VISIBLE_DEVICES`.

For running with pm2:
```bash
$ CUDA_VISIBLE_DEVICES=<device id> pm2 start run.sh --name openvalidators_autoupdate -- --wallet.name <your-wallet-name> --wallet.hotkey <your-wallet-hot-key>
```

For runing `neuron.py` directly:
```bash
$ CUDA_VISIBLE_DEVICES=<device id> python3 validators/openvalidators/neuron.py --wallet.name <your-wallet-name> --wallet.hotkey <your-wallet-hot-key>
```
# License

The MIT License (MIT) Copyright © 2023 Yuma Rao

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated 
documentation files (the “Software”), to deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit 
persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

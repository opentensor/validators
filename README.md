<div align="center">

# **Bittensor-Validators**

---

</div>

Bittensor-validators is the repository for all of the validator code currently running on the bittensor network.
The validators are responsible for identifing important/useful peers in the network and rank them accordingly. 
To achieve this, validators will send requests to different peers on the network and evaluate their responses. 

## 1. Install
From source:
```
$ git clone https://github.com/opentensor/validators.git
$ python3 -m pip install -e validators/
```
## 2. Using Bittensor-validators
```
from validators import finney_logit

if __name__ == "__main__":
    template = finney_logit.neuron().run()
```

# Quickstart to Mining

## Installation


This repository requires python3.9 or higher. To install it, simply clone this repository and run the [install.sh](./install.sh) script.
```bash
git clone https://github.com/brokespace/code
cd code
python -m pip install --use-deprecated=legacy-resolver -r requirements.txt
python -m pip install -e .
python -m pip uninstall uvloop # b/c it causes issues with threading/loops
```


## How to Run
You can use the following command to run a miner or a validator. 

```bash
python <SCRIPT_PATH>
    --netuid 45
    --subtensor.network <finney/local/test>
    --neuron.device cuda
    --wallet.name <your wallet> # Must be created using the bittensor-cli
    --wallet.hotkey <your hotkey> # Must be created using the bittensor-cli
    --logging.debug # Run in debug mode, alternatively --logging.trace for trace mode
    --axon.port # VERY IMPORTANT: set the port to be one of the open TCP ports on your machine
```

where `SCRIPT_PATH` is either: 
1. neurons/miner.py
2. neurons/validator.py

For ease of use, you can run the scripts as well with PM2. Installation of PM2 is: 
**On Linux**:
```bash
sudo apt update && sudo apt install jq && sudo apt install npm && sudo npm install pm2 -g && pm2 update
``` 

Example of running an openai miner:

```bash
pm2 start neurons/miner.py --interpreter python3 --name miner -- --netuid XY  --subtensor.network finney --wallet.name coldkey --wallet.hotkey hotkey --neuron.model_id gpt4 --axon.port 8091 --logging.debug --miner.name openai
```

## Subnet Wallet Registration
Register your wallet on the subnet: 
```
btcli s register --subtensor.network finney --netuid 45
```

Testnet: 
```
btcli s register --subtensor.network test --netuid 171
```


# Testnet 
We highly recommend that you run your miners on testnet before deploying on main. This is give you an opportunity to debug your systems, and ensure that you will not lose valuable immunity time. The SN1 testnet is **netuid 171**. 

In order to run on testnet, you will need to go through the same hotkey registration proceure as on main, but using **testtao**. You will need to ask for some in the community discord if you do not have any. 

To run:

```bash
pm2 start neurons/miner.py --interpreter python3 --name miner -- --netuid 171  --subtensor.network test --wallet.name test_coldkey --wallet.hotkey test_hotkey --neuron.model_id gpt4 --axon.port 8091 --logging.debug --miner.name openai
```


# Ramping up


## Tasks

A list of the provided tasks can be seen [here](./tasks.md). Tasks are scored equally based on speed and similarity to the answer.

## Sample Miners

There are some sample miners you can use check them out [here](./sample-miners.md). 


## Helpful Tips

It is suggested that you play around with mining on Testnet before going to Mainnet.
If issues are encountered with btcli, it is recommended to use btcli v7.1.2 (https://github.com/opentensor/bittensor/commits/release/7.1.2/)

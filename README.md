# finsight

Experiments for automatic trading and analysis of financial data

## Foreword
This is a RESEARCH repository and is not designed to be used in production.
I am not responsible for any losses or damages that may occur as a result of using this software.


## Dev environment setup
This project uses Miniconda to manage its environment.

Please use the environment specified in `environment.yml`. This assumes
a Linux based machine with a CUDA-enabled GPU.

If you are unable to use the `environment.yml` file for some reason,
here are the commands used to set up the environment

```bash
conda create -n finsight python=3.9
conda activate finsight
# This is for CUDA 11.3. Please see: https://pytorch.org/get-started/locally/
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
conda install -c conda-forge tqdm python-dotenv
conda install pandas ipympl matplotlib jupyter
# For testing
pip install alpaca-trade-api
```


## Testing against the actual market
The alpaca markets API is being used here. I am most certainly not sponsored by them.
Please use the paper trading API so you don't lose money.
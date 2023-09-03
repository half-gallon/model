# half-gallon: ML model to classify user voice

## Overview

This repository implements voice classification model with torch and ezkl.

```markdown
├── Verifier.sol # zkp verifier contract
├── data
│ ├── test # test dataset
│ └── train # train dataset
├── input.json # sample trainning data
├── kzg.srs # SNARK string
├── model.py # torch model implementation
├── network.onnx # torch model exported to .onnx format
├── pk.key # proving key
├── settings.json # ezkl build setting
├── vk.key # verification key
```

### Caveat

This model is not that accurate becuase of simplified model implementation (due to ezkl proving limitation) and small training data set.

## Requirement

- ezkl@1.13.2
- solc-select
- python@3.11.5

## Install

```bash
# ezkl
# if below curl doesn't work, download executable from https://github.com/zkonduit/ezkl/releases
curl https://hub.ezkl.xyz/install_ezkl_cli.sh | bash

# solc-select
brew install solc-select
solc-select install 0.8.17
solc-select use 0.8.17

# install pip requirements
python3 -m venv venv
source ./venv/bin/activate
pip install -r requirements.txt
```

## Develop

**1. export model**

```bash
# export simple model to .onnx file
python model.py

# or export complex model to .onnx file
python model_complex.py
```

**2. compile model**

> This generates Verifier.sol file with a error.

```bash
# compile .onnx file to circuit and verifier contract
./run.sh
```

**3. start api server**
See more https://github.com/half-gallon/server

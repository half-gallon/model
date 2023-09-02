set -ex

ezkl --version

SOLC_VERSION="0.8.17"

python model.py

read -p "Continue? (Y/N): " confirm && [[ $confirm == [yY\ ] || $confirm == [yY][eE][sS] ]] || exit 1

# https://docs.ezkl.xyz/command_line_interface/

ezkl gen-settings -M network.onnx

read -p "Continue? (Y/N): " confirm && [[ $confirm == [yY\ ] || $confirm == [yY][eE][sS] ]] || exit 1

ezkl calibrate-settings -M network.onnx -D input.json --target resources

ezkl get-srs -S settings.json

ezkl compile-model -M network.onnx -S settings.json --compiled-model network.ezkl

ezkl setup -M network.ezkl --srs-path=kzg.srs --vk-path=vk.key --pk-path=pk.key --settings-path=settings.json

ezkl gen-witness -D input.json -M network.ezkl -S settings.json

ezkl prove -M network.ezkl --witness witness.json --pk-path=pk.key --proof-path=model.proof --srs-path=kzg.srs --settings-path=settings.json

ezkl verify --proof-path=model.proof --settings-path=settings.json --vk-path=vk.key --srs-path=kzg.srs

# https://docs.ezkl.xyz/verifying_on-chain/#verifying-with-the-evm-

solc-select install $SOLC_VERSION
SOLC_VERSION=$SOLC_VERSION ezkl create-evm-verifier --srs-path=kzg.srs --vk-path vk.key --sol-code-path verif.sol --settings-path=settings.json

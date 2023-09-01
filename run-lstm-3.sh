set -ex

python lstm-3.py

# https://docs.ezkl.xyz/command_line_interface/

ezkl gen-settings -M lstm-3.onnx

ezkl calibrate-settings -M lstm-3.onnx -D input.json --target resources

ezkl get-srs -S settings.json

ezkl compile-model -M lstm-3.onnx -S settings.json --compiled-model network.ezkl

ezkl setup -M network.ezkl --srs-path=kzg.srs --vk-path=vk.key --pk-path=pk.key --settings-path=settings.json

ezkl gen-witness -D input.json -M network.ezkl -S settings.json

ezkl prove -M network.ezkl --witness witness.json --pk-path=pk.key --proof-path=model.proof --srs-path=kzg.srs --settings-path=settings.json

ezkl verify --proof-path=model.proof --settings-path=settings.json --vk-path=vk.key --srs-path=kzg.srs

# https://docs.ezkl.xyz/verifying_on-chain/#verifying-with-the-evm-

# solc@0.8.17 !
ezkl create-evm-verifier --srs-path=kzg.srs --vk-path vk.key --sol-code-path verif.sol --settings-path=settings.json

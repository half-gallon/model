set -ex

# model
cd ../model
source ./venv/bin/activate

./run.sh || true # can be error

# server
cd ../server
source ./venv/bin/activate

make copy-model && python compute-proof.py

# contract
cd ../test-contract
npm run build && npm run test

# back to model directory
cd ../model
source ./venv/bin/activate

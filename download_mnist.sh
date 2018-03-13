#!/bin/sh -eu

curl -o mnist/train-images-idx3-ubyte.gz --create-dirs http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
curl -o mnist/train-labels-idx1-ubyte.gz --create-dirs http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
curl -o mnist/t10k-images-idx3-ubyte.gz --create-dirs http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
curl -o mnist/t10k-labels-idx1-ubyte.gz --create-dirs http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz

gzip -d mnist/train-images-idx3-ubyte.gz
gzip -d mnist/train-labels-idx1-ubyte.gz
gzip -d mnist/t10k-images-idx3-ubyte.gz
gzip -d mnist/t10k-labels-idx1-ubyte.gz

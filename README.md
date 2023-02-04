# Machine learning

This example aims to run neural network inference on zkVM. In this example, what we want to keep secret is *a deep learning model* (precisely, how a model is organized is not secret, the weights inside are secret), and the prover feeds some input data and weights into zkVM and runs it.
By running it in zkVM, the prover can convince the verifier that "when an input data X is given to a model M, its output will be Y" without revealing model parameters (or weights).

# Data and model

What's used for input data is [the MNIST-1D dataset](https://github.com/greydanus/mnist1d) which is a tiny variation of the widely used MNIST data set. As the name suggests, this is one-dimensional data that represents a digit from 0 to 9 and forms a *(1,40)* matrix that is much smaller than that of the MNIST (*(28,28)* matrix). I've decided to go with it for its small size because a large data or model may not be functional on zkVM at this moment.

And, the organization of the model used is as follows.
```
-- Input data:  (1, 40)
-- Layer1:      (40, 16)  # matrix multiplication
   -- Activation: Relu    # use Relu as activation function
-- Layer2:      (16, 10)  # matrix multiplication
-- Argmax:      (10)      # pick the final prediction
```

The model is pre-trained offline with 4000 data. `data/w1_1d_40_16.csv` and `data/w2_1d_16_10.csv` indicate pre-trained weights for Layer1 and Layer2 respectively.

# Main program

The main program that calls a method in the guest zkVM is [host/src/main.rs](host/src/main.rs). It first loads pre-trained weights and test data, and then it invokes `inference()` over two data in the host environment. The reason to call it in the host is to check the consistency between host and guest execution.

```
# prediction: prediction genereted by the given model and data
# answer: correct answer
[host] prediction: 9, answer: 2   # wrong prediction
[host] prediction: 6, answer: 6   # correct prediction
[host] success: 1
```

Afterward, it tries to invoke `inference()` over the same data but into the guest zkVM environment. The prover sends two pieces of data at once to the guest, to eliminate unnecessary communication overheads. And the guest returns the result of the prediction for each piece of data.

```
[guest] prediction: 9, answer: 2
[guest] prediction: 6, answer: 6
[guest] prove() time elapsed: 675.430503558s
[guest] verify() time elapsed: 149.86107ms
[guest] success: 1
```

You can see the result of the guest execution matches that of the host execution. Also, it prints out how much time `prove()` and `verify()` take respectively.

# Run this example

```
cargo run
```

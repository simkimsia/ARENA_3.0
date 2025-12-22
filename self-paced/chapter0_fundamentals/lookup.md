# Lookup


- Understand the structure and function of neural networks
- Learn essential linear algebra concepts like matrix operations and transformations
- Understand core principles of probability and statistics, including expected value and variance
- Learn how calculus concepts (particularly differentiation) are applied in optimization tasks
- Cover some foundational information theory concepts, such as entropy and KL divergence
- Enhance Python programming skills, focusing on NumPy and PyTorch basics


## Neural Networks

Classic example

![784 neurons at layer 1 of multi-layer perceptron recognize handwritten digits](nn-recognize-1.png)

- CNN -> good for image recognition
- LSTM -> good for time series, good for speech recognition
- Transformer -> good for NLP

Simplest NN = multi-layer perceptron


## What's a Neuron?

A thing that holds a number and does some math.

The number inside a neuron is called the activation.

For classic MNIST, 784 neurons correspond to a 28x28 pixel image.

[3Blue1Brown - But what is a neural network? | Chapter 1, Deep learning](https://www.youtube.com/watch?v=aircAruvnKk)

The last layer of 10 neurons represents the 10 possible digits.

Whenever we want to squish a range of numbers into something between 0 and 1, we use a sigmoid function.

To determine how neurons of 1 layer determine if a neuron of the next layer should activate, we use series of weights and the activation of the previous layer.

![Previous layer's neurons affect next layer's 1 neuron](previous-layer-affect-neuron.png)

Sometimes you want the weighted sum to only activate the enxt neuron if it's above a number say 10.

![Explaining why bias](<bias-for-inactivity.png>)

so you add a bias term. THis is **bias for inactivity**.

Learning in this context is about finding the right weights and biases.

![Learning is about getting the right weights and biases](<learning.png>)

## Representing the Network Mathematically

![Representing the network mathematically](<math.png>)

**Scalar form** (for a single neuron):

$$
a_0^{(1)} = \sigma \left( w_{0,0} a_0^{(0)} + w_{0,1} a_1^{(0)} + \cdots + w_{0,k} a_k^{(0)} + b_0 \right)
$$

**Matrix form** (for all neurons in a layer):

$$
\begin{bmatrix}
a_0^{(1)} \\
a_1^{(1)} \\
\vdots \\
a_n^{(1)}
\end{bmatrix}
= \sigma \left(
\begin{bmatrix}
w_{0,0} & w_{0,1} & \cdots & w_{0,k} \\
w_{1,0} & w_{1,1} & \cdots & w_{1,k} \\
\vdots & \vdots & \ddots & \vdots \\
w_{n,0} & w_{n,1} & \cdots & w_{n,k}
\end{bmatrix}
\begin{bmatrix}
a_0^{(0)} \\
a_1^{(0)} \\
\vdots \\
a_k^{(0)}
\end{bmatrix}
+
\begin{bmatrix}
b_0 \\
b_1 \\
\vdots \\
b_n
\end{bmatrix}
\right)
$$

**Sigmoid function** (activation function that squishes values to 0-1):

$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$

**Where:**
- $a^{(l)}$ = activations (neuron values) at layer $l$
- $a^{(0)}$ = input layer activations (e.g., 784 pixel values for MNIST)
- $a^{(1)}$ = next layer activations after applying weights, biases, and sigmoid
- $W$ = weight matrix — each $w_{j,k}$ connects neuron $k$ in layer 0 to neuron $j$ in layer 1
- $b$ = bias vector — determines how easy it is for each neuron to activate
- $\sigma$ = activation function (sigmoid in this case) — applied element-wise
- $n$ = number of neurons in layer 1
- $k$ = number of neurons in layer 0 (input layer)


## WHy do we use ReLU?

Using sigmoid is old school.

Most networks use ReLU (Rectified Linear Unit) because it's easier to train.

![ReLU](<relu.png>)

$$ReLU  (x) = max(0, x)$$

## Different layers presumably learn different things

Given last layer is 10 neurons, each neuron is responsible for a digit.

So 2nd last layer may be learning different components such as the o in 9, 8, etc, or the vertical stem in 7, 4, 1, etc.

So the very first layer may learn edges.

But, this is just a guess!

We can visualize the weights of the connections from one layer to the next layer as a given pixel pattern. To know better and it looks just random.

![visualizing the weights of the connections from one layer to the next layer as a given pixel pattern](<pixel-pattern.png>)

This means the network can be just as confident of recognizing some random patterns as the digit 5.

![confidently wrong](<confidently-wrong.png>)

## Cost function explanation

Let's say we start with random weights and biases first. Then we run the network and get some output. We compare that with what we desire.

The difference is the cost function we want to apply.

![Cost function is the difference between the output and the desired output](<cost-function.png>)

Mathematically, cost function is the sum of the square of the differences between the output and the desired output.

![Sum of square of the differences](<sum-square-difference.png>)

The sum is small, when correct. The sum is large, when wrong.

Consider the average cost is our measure of how good or bad the network is.

The Neural Network function is such that

1. Input : 784 pixels
2. Output : 10 neurons
3. Parameters: 13,002 weights and biases (input for cost function)

The Cost Function is such that

1. Input : 10 neurons (output of Neural Network function)
2. Output : 1 number (the cost)
3. Parameters: Many, many, many training examples

## Finding minimum cost

1. Find direction of steepest gradient / ascent ($\nabla C$)
2. Small step in ($-\nabla C$) because we want to go down
3. Repeat

aka gradient descent.

## QnA

### What makes neural networks more powerful than basic statistical methods like linear regression?


- Neural networks exploit **nonlinearity**, which allows them to express a much wider set of possible functions, whereas linear regression is relatively limited.
- Neural networks are learned using **gradient descent**, meaning their **power isn't upper-bounded by the algorithms** which programmers or mathematicians can feasibly design by hand.


### What are the advantages of ReLU activations over sigmoids?


- ReLU more effectively avoids the **vanishing gradient**    problem, which is common in sigmoids.

> **Vanishing Gradient Problem**: The sigmoid function squishes all inputs to a range between 0 and 1. When the output is very close to 0 or 1 (i.e., saturated), the derivative (gradient) of the sigmoid becomes extremely small—nearly zero. During backpropagation, gradients are multiplied together as they flow backward through layers. If each layer contributes a tiny gradient, these small numbers multiply to produce vanishingly small updates to weights in the earlier layers. This means those layers learn very slowly, or not at all. ReLU avoids this because its gradient is either 0 or 1—no squishing, no vanishing.
- ReLU is more computationally efficient to evaluate than sigmoid.
- However, an important point about ReLU and much of ML in general - the better empirical results often come before the theoretical justifications! A lot of ML is built on the philosophy of "experiment until you find something that works, then figure out why it works."
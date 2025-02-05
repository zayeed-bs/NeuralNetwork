## Phase 1: Basic Structure (Core Properties & Methods)
Goal: Create the foundation of the class, including initialization, forward propagation, and a basic training loop.
🔹 Properties (Instance Variables)

    layers → A list defining the number of neurons per layer (e.g., [2, 3, 1] for a 2-layer network).
    weights → Dictionary to store weight matrices for each layer.
    biases → Dictionary to store bias vectors for each layer.
    learning_rate → Controls how much the weights are adjusted.

🔹 Methods

    __init__(self, layers, learning_rate=0.01) → Initializes weights, biases, and network structure.
    forward_propagation(self, X) → Computes activations for each layer and returns the final output.

## Phase 2: Adding Activation Functions
Goal: Implement activation functions and use them in forward propagation.
🔹 Additional Properties

    Store activation functions (Sigmoid, ReLU, etc.).

🔹 New Methods

    sigmoid(self, Z) → Implements the sigmoid function.
    relu(self, Z) → Implements the ReLU function.
    softmax(self, Z) → For multi-class classification.

## Phase 3: Implementing Backpropagation
Goal: Enable the model to learn by adjusting weights based on errors.
🔹 New Methods

    compute_loss(self, Y, Y_pred) → Calculates loss (MSE for regression, cross-entropy for classification).
    backpropagation(self, X, Y, cache) → Computes gradients and updates weights.
    update_weights(self, gradients) → Adjusts weights using gradient descent.

## Phase 4: Training the Model

Goal: Implement a training loop to fit the model to data.
🔹 New Methods

    train(self, X, Y, epochs=100) → Loops through forward pass, backpropagation, and weight updates.

## Phase 5: Making Predictions & Evaluation

Goal: Allow the trained model to make predictions and assess performance.
🔹 New Methods

    predict(self, X) → Runs forward propagation and returns predictions.
    evaluate(self, X, Y) → Computes accuracy and loss on test data.
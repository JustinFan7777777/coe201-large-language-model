"""
Assignment 1 - Problem 2: Domain Adversarial Neural Networks & GRL (40 points)

The Gradient Reversal Layer (GRL) is a key component for training Domain
Adversarial Neural Networks (DANN). It allows a model to learn features that are
useful for a main task but invariant across different data domains.

Background:
-----------
In domain adaptation, we have:
- Source Domain: Training data with labels (e.g., synthetic images)
- Target Domain: Test data without labels (e.g., real-world photos)

The goal is to learn features that work well on both domains by using adversarial
training. The GRL enables this by reversing gradients during backpropagation.

Architecture:
-------------
Input -> Feature Extractor -> Shared Features (256-d)
                                    |
                    +---------------+---------------+
                    |                               |
              Label Classifier              Gradient Reversal Layer
                    |                               |
                    v                               v
            Label Output (10 classes)        Domain Classifier
                                                    |
                                                    v
                                           Domain Output (Source/Target)

IMPORTANT:
- Do not modify the function signatures or the class structure.
- Implementation should only be done within the designated TODO blocks.
- Do not add any extra imports.
"""
import torch
import torch.nn as nn
from torch.autograd import Function

# ============================================================
# Part A & B: Custom Autograd Function (25 pts)
# ============================================================

class GradientReversalFunction(Function):
    """
    Custom autograd function for Gradient Reversal.

    The GRL behaves differently during forward and backward passes:

    Forward Pass: Acts as an identity function.
        y = x  (output equals input)

    Backward Pass: Negates and scales the gradient by -lambda_.
        grad_input = -lambda_ * grad_output

    This is achieved by implementing a custom backward() that PyTorch's
    autograd system will call during backpropagation.

    Reference:
        Ganin, Y., & Lempitsky, V. (2015). Unsupervised Domain Adaptation
        by Backpropagation. ICML.

    Example:
        >>> x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        >>> output = GradientReversalFunction.apply(x, 0.5)
        >>> output
        tensor([1., 2., 3.], grad_fn=<GradientReversalFunctionBackward>)
        >>> loss = output.sum()
        >>> loss.backward()
        >>> x.grad
        tensor([-0.5000, -0.5000, -0.5000])
    """

    @staticmethod
    def forward(ctx, x, lambda_):
        """
        Forward pass (Identity function).

        During the forward pass, the GRL simply passes the input through unchanged.
        The lambda_ parameter is saved in the context for use in backward pass.

        Args:
            ctx: context object to save information for backward pass
            x: input tensor of any shape
            lambda_: float, gradient scaling factor (typically between 0 and 1)

        Returns:
            output: Same as input x (identity function)
        """
        ### TODO: Implement forward pass and save lambda_

        ctx.lambda_ = lambda_
        # ctx: context object to save information for backward pass
        # put lambda_ in context for backward pass

        return x.view_as(x)
        # when doing forward pass, we want to return the input as output (identity function)
    
        ### END TODO

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass (Reverse and Scale).

        During the backward pass, the gradient is negated and scaled by -lambda_.
        This is the key operation that enables adversarial training.

        Mathematically:
            dL/dx = -lambda_ * dL/dy

        where dL/dy is the incoming gradient from downstream layers.

        Args:
            ctx: context object containing saved lambda_
            grad_output: incoming gradient (dL/dy) from downstream layers

        Returns:
            Tuple of (grad_input, None):
                - grad_input: Gradient w.r.t. input, computed as -lambda_ * grad_output
                - None: Gradient w.r.t. lambda_ (not needed since it's not a learnable parameter)


        Example:
            >>> # If lambda_ = 0.5 and grad_output = [2, 4, 6]
            >>> # Then grad_input = [-1, -2, -3]
        """
        ### TODO: Implement backward pass

        grad_input = -ctx.lambda_ * grad_output
        # grad_output: incoming gradient (dL/dy) from downstream layers
        # grad_input: Gradient w.r.t. input, computed as -lambda_ * grad_output

        return grad_input, None
        # the return value of backward() must be a tuple
        # with the same number of elements as the inputs to forward()
        # grad_lambda is None since lambda_ is not a learnable parameter
        # and we don't need to compute its gradient
    
        ### END TODO


class GradientReversalLayer(nn.Module):
    """
    Module wrapper for GradientReversalFunction.

    This wraps the custom autograd function as a PyTorch nn.Module,
    making it easy to use in neural network architectures.

    Usage:
        grl = GradientReversalLayer(lambda_=0.5)
        features = feature_extractor(x)
        reversed_features = grl(features)
        domain_output = domain_classifier(reversed_features)

    Attributes:
        lambda_: The scaling factor for gradient reversal.

    Example:
        >>> grl = GradientReversalLayer(lambda_=1.0)
        >>> x = torch.randn(32, 256)
        >>> output = grl(x)
        >>> output.shape
        torch.Size([32, 256])
        >>> torch.allclose(output, x)  # Forward is identity
        True
    """

    def __init__(self, lambda_=1.0):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        """
        Apply the Gradient Reversal Function.

        Args:
            x: Input tensor from feature extractor

        Returns:
            Output tensor with reversed gradients during backpropagation
        """
        ### TODO: Apply the custom function

        return GradientReversalFunction.apply(x, self.lambda_)
        # apply the custom function to input x with lambda_ as argument
    
        ### END TODO


# ============================================================
# Part D: DANN Architecture (10 pts)
# ============================================================

class DANNModel(nn.Module):
    """
    A minimal Domain Adversarial Neural Network (DANN) model architecture.

    This architecture implements the key components of domain adversarial training:

    1. Shared Feature Extractor: Extracts features from input data
       - Input: Flattened images (784 dimensions for 28x28 images)
       - Hidden: 256-dimensional features with ReLU activation

    2. Label Classifier (Main Task): Predicts class labels
       - Input: 256-d features
       - Output: 10 class logits (for digits 0-9)

    3. Domain Classifier (Adversarial Task): Predicts domain (Source/Target)
       - Input: Features AFTER gradient reversal
       - Output: 2 domain logits (Source=0, Target=1)

    The key insight: By using GRL, the feature extractor is trained to produce
    features that fool the domain classifier, making them domain-invariant.

    Attributes:
        feature_extractor: Shared feature extraction network
        label_classifier: Classifier for main task (label prediction)
        grl: Gradient Reversal Layer for adversarial training
        domain_classifier: Classifier for domain prediction

    Example:
        >>> model = DANNModel(lambda_=1.0)
        >>> x = torch.randn(64, 784)  # Batch of 64 flattened 28x28 images
        >>> label_logits, domain_logits = model(x)
        >>> label_logits.shape
        torch.Size([64, 10])
        >>> domain_logits.shape
        torch.Size([64, 2])
    """

    def __init__(self, lambda_=1.0):
        super().__init__()
        ### TODO: Define DANN layers

        self.feature_extractor = nn.Sequential(
            nn.Linear(784, 256),
            # Input: 784-d flattened image, Output: 256-d features
            nn.ReLU()
            # ReLU activation to introduce non-linearity in feature extraction
        )

        self.label_classifier = nn.Linear(256, 10)
        # Label classifier: 256 input features -> 10 class logits (for digits 0-9)

        self.grl = GradientReversalLayer(lambda_=lambda_)
        # Gradient Reversal Layer with specified lambda_ for adversarial training
        # This layer will reverse gradients during backpropagation to encourage domain-invariant features
        # The lambda_ parameter controls the strength of the adversarial signal
        # A higher lambda_ means stronger gradient reversal
        # which can lead to more domain-invariant features but may also make training more difficult

        self.domain_classifier = nn.Linear(256, 2)
        # Domain classifier: 256 input features -> 2 domain logits (Source=0, Target=1)
        # This classifier will be trained to distinguish between source and target domains
        # while the feature extractor will be trained to fool it, creating a minimax game

        ### END TODO

    def forward(self, x):
        """
        Forward pass through the DANN model.

        The forward pass branches after the shared feature extractor:
        - One path goes directly to the label classifier
        - The other path goes through GRL to the domain classifier

        Args:
            x: Input tensor of shape (batch_size, 784) for 28x28 flattened images


        Returns:
            Tuple of (label_logits, domain_logits):
                - label_logits: Class predictions of shape (batch_size, 10)
                - domain_logits: Domain predictions of shape (batch_size, 2)
        """
        ### TODO: Implement the forward pass

        features = self.feature_extractor(x)
        # Extract shared features from input x using the feature extractor
        # features will have shape (batch_size, 256)
        # These features will be used for both label classification and domain classification

        label_logits = self.label_classifier(features)
        # Predict class labels from the shared features using the label classifier
        # label_logits will have shape (batch_size, 10)

        reversed_features = self.grl(features)
        # Pass the shared features through the Gradient Reversal Layer
        # This will reverse the gradients during backpropagation
        # to encourage domain-invariant features

        domain_logits = self.domain_classifier(reversed_features)
        # Predict domain labels from the reversed features using the domain classifier
        # domain_logits will have shape (batch_size, 2)

        return label_logits, domain_logits
        # Return both label predictions and domain predictions as output of the forward pass
    
        ### END TODO

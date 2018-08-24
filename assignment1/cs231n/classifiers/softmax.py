import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  
  N = X.shape[0]
  C = W.shape[1]
  for i in range(N):
    scores = X[i].dot(W)
    exp_scores = np.exp(scores)
    softmax_scores = exp_scores / np.sum(exp_scores)
    loss += -np.log(softmax_scores[y[i]])

    # gradient: dW =  x(S(y) - 1(y_i == correct class))
    for j in range(C):
      if j == y[i]:
        dW[:, j] += X[i].dot(softmax_scores[j] - 1)
      else:
        dW[:, j] += X[i].dot(softmax_scores[j])


  # average over all samples
  loss /= N
  dW /= N

  # add regularization loss
  loss += 0.5 * reg * np.sum(W * W)

  # add regularization gradient
  dW += reg * W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  
  N = X.shape[0]
  C = W.shape[1]

  scores = X.dot(W) # N x C
  exp_scores = np.exp(scores) # N x C
  probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) # N x C
  neglogprobs = -np.log(probs[range(N), y]) # N x 1
  
  data_loss = np.sum(neglogprobs) / N
  reg_loss = 0.5 * reg * np.sum(W * W)
  loss = data_loss + reg_loss

  dscores = probs # N x C
  dscores[range(N), y] -= 1
  dscores /= N

  dW = np.dot(X.T, dscores)
  dW += reg * W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


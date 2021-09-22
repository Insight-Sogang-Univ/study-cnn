from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    # 하나의 이미지 선택
    # X.shape[0] = N
    for i in range(X.shape[0]):
        # 해당 이미지에 대한 class별 score를 저장할 변수 초기화
        scores = X[i] @ W
        scores -= scores.max()
        scores = np.exp(scores)

        # 해당 이미지가 y[i]일 확률 계산
        P = scores / scores.sum()

        # 해당 이미지에 대한 Loss 계산
        L_i = -np.log(P[y[i]])

        # 전체 Loss에 합산
        loss += L_i

        # Gradient 계산
        P[y[i]] -= 1
        dW += X[i].reshape(-1, 1) @ P.reshape(1, -1)

    # Data Loss
    loss = loss / X.shape[0]
    dW /= X.shape[0]

    # Regularization Loss
    loss += reg * (W ** 2).sum()
    dW += 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    scores = X @ W
    
    # for Numerical Stability
    scores -= scores.max(axis=1).reshape(-1,1)

    # Loss 계산
    loss += -np.sum(
      np.log( np.exp(scores[range(X.shape[0]),y]) / np.sum(np.exp(scores),axis=1) )
    )
    
    # class별 확률
    P = np.exp(scores) / np.sum(np.exp(scores), axis=1).reshape(-1,1)
    P[range(X.shape[0]), y] -= 1
    dW = X.T @ P
    
    loss = loss / X.shape[0]
    dW = dW / X.shape[0]
    
    # Regularization Loss
    loss += reg * (W ** 2).sum()
    dW += 2 * reg * W
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW

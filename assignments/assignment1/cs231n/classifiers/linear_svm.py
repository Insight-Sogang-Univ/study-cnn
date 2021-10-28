from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

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
    dW = np.zeros(W.shape)  # gradient를 0으로 초기화

    # loss와 gradient 계산하기
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue  # 정답 class이므로 loss에 더해줄 필요 없음
            margin = scores[j] - correct_class_score + 1  # note delta = 1
            if margin > 0:
                loss += margin 

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train

    # loss 함수에 L2 regularization을 추가해준다
    loss += reg * np.sum(W * W)

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    # gradient 만 보고 있는 것임!!
    
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue 
            margin = scores[j] - correct_class_score + 1  # note delta = 1
            if margin > 0:
                dW[:,j] += X[i]
                dW[:,y[i]] -= X[i]
    
    dW /= num_train
    
    dW = dW + reg * 2 * W  #L2 regularization을 w에 대해 미분한 걸 더해준다

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def svm_loss_vectorized(W, X, y, reg): # 걍 이게 이해가 안 돼
    """
    위와 같은 것을 vectorized해서 구하는 방법, 
    학습 속도를 높이기 위해 for loop 말고 vector연산을 하는 것임
    
    Structured SVM loss function, vectorized implementation.
    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Compute the loss
    num_classes = W.shape[1]
    num_train = X.shape[0]
    scores = X.dot(W)
    correct_class_scores = scores[ np.arange(num_train), y].reshape(num_train,1)
    margin = np.maximum(0, scores - correct_class_scores + 1)
    margin[np.arange(num_train), y] = 0 # do not consider correct class in loss
    loss = margin.sum() / num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    mask = np.zeros(margin.shape)
    mask[margin>0] = 1             #얘가 행렬을 알아서 반환함 (필기에있는파란색?행렬)
    
    mask[range(num_train), y] = -mask.sum(axis=1)
    dW += (X.T @ mask) / num_train   #된대///안되면 다시 .T.dot(mask)

    dW = dW + reg * 2 * W 


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW

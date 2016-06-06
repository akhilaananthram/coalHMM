"""
By: Akhila Ananthram (asa225)
"""
import math
import numpy as np
from scipy.linalg import expm

from constants import *

def generate_Q(t, pi=None, beta=1, gamma=1, delta=1):
  """
  Generate the probability of base b at time t starting from a. Defaults give
  jukes cantor
  @param t      (int)       time step
  @return       (np.array)  transition probability, P[a, b] of base b starting
                            from a for a given time
  """
  # To account for the CpG effect (higher mutation rates from CpG -> TpG and
  # CpG -> CpA on the opposite strand), pi['A'] == pi['T'] should be the
  # largest value in Q

  if pi is None:
    pi = {'A': 0.25, 'C': 0.25, 'G': 0.25, 'T': 0.25}

  assert pi['A'] == pi['T']
  assert pi['G'] == pi['C']

  Q = np.array([[0, pi['G'], beta * pi['C'], gamma * pi['T']],
                [pi['A'], 0, delta * pi['C'], beta * pi['T']],
                [beta * pi['A'], delta * pi['G'], 0, pi['T']],
                [gamma * pi['A'], beta * pi['G'], pi['C'], 0]])

  # Rows must sum to 0
  for i in range(len(Q)):
    Q[i, i] = -np.sum(Q[i])

  return expm(-Q * t)


def post_order(root):
  """
  Orders the nodes of the tree in post order traversal
  @param root   (Node)   root of tree
  @return       (list)   list of Nodes in post order traversal
  """
  if root is None:
    return []

  return post_order(root.left) + post_order(root.right) + [root]


def felsensteins(root):
  """
  Finds the log likelihood of the tree topology: log(P(X, T))
  @param root     (Node)    root of tree
  @return         (float)   log likelihood
  """
  ordering = post_order(root)

  # O(n * L * a^2)
  for node in ordering: # O(2n - 1) = O(n)
    if node.is_leaf():
      # handle gaps
      node.res[:, node.seq == 4] = 1
      # update to avoid index out of bounds
      node.seq[node.seq == 4] = 0
      node.res[node.seq, np.arange(len(node.seq))] = 1
    else:
      trans_l = generate_Q(node.left.time)
      trans_r = generate_Q(node.right.time)
      for u in range(node.len_seq): # O(L)
        for a in range(NUM_BASES): # O(a)
          left = 0
          right = 0
          for b in range(NUM_BASES): # O(a)
            left += trans_l[a, b] * node.left.res[b, u]
            right += trans_r[a, b] * node.right.res[b, u]
          node.res[a, u] = left * right

  # TODO: use correct pi if Q is not jukes-cantor
  like_u = np.sum(root.res, axis=0) * 0.25
  log_like = np.sum(np.log(like_u))

  return log_like

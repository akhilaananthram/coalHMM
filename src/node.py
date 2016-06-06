"""
By: Akhila Ananthram (asa225)
"""
import numpy as np
from constants import *


class Node(object):
  """ Data structure to store the tree topology and sequences """

  def __init__(self, seq, time=None, left=None, right=None, name=""):
    """
    Initialize a Node object
    @param seq    (int/list)    Either length of sequence of the actual
                                sequence
    @param time   (int)         time steps between ancestor and node
    @param left   (Node)        Left child
    @param right  (Node)        Right child
    @param name   (string)      name to identify the tree
    """
    self.seq = None if type(seq) == int else seq
    self.time = time
    self.left = left
    self.right = right
    self.name = name

    # Residues
    if type(seq) == int:
      self.len_seq = seq
    else:
      self.len_seq = len(seq)
    self.res = np.zeros((NUM_BASES, self.len_seq))

  def is_leaf(self):
    """
    Determines if the node is a leaf by seeing if it has children
    @return       (bool)      Whether or not it is a leaf
    """
    return self.left is None and self.right is None

  def max_sequence(self):
    """
    Determines the maximum a posteriori base for each position or returns the
    original sequence
    @return     (list)      DNA sequence
    """
    if self.seq is not None:
      return [RETURN_MAPPING[s] for s in self.seq]

    # P(r = a | X, theta) = P(X | r = a, theta) * P(r = a) / P(X, theta)
    seq = np.argmax(self.res, axis=0)
    return [RETURN_MAPPING[s] for s in seq]

  def __str__(self):
    """ Human readable print statement """
    return self.name



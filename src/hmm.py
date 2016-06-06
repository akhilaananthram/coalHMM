"""
By: Akhila Ananthram (asa225)
"""
import numpy as np


def back_trace(pointers):
  """
  @param pointers     (np.array)    pointer to previous state
  @return             (list)        list of hidden states
  """
  _, n = pointers.shape
  current = np.argmax(pointers[:, -1])
  states = [current]

  for i in reversed(range(1, n)):
    current = pointers[current, i]
    states.append(current)

  return [x for x in reversed(states)]


def viterbi(initial, emission, transition, alignment, groupings):
  """
  @param initial      (np.array)    initial[i] probability of starting in state
                                    i
  @param emission     (np.array)    emission[i,j] is the probability of
                                    releasing the observed value j from state i
  @param transition   (np.array)    transition[i,j] is the probability from
                                    state i to state j
  @param alignment    (list)        alignment with values represented as numbers
  @param groupings    (dict)        Maps column of alignment to index
  @return             (list)        list of hidden states
  """
  # Empty alignment
  _, len_alignment = alignment.shape
  if len_alignment == 0:
    return None

  V = np.zeros((len(initial), len_alignment))
  pointers = np.zeros((len(initial), len_alignment))

  # Initialize base
  column = groupings["".join(alignment[:, 0])]
  V[:, 0] = np.log(initial) + np.log(emission[:, column])

  for i in xrange(1, len_alignment):
    for l in xrange(len(initial)):
      column = groupings["".join(alignment[:, i])]
      V[l, i] = np.log(emission[l, column]) + \
          np.max(V[:, i - 1] + np.log(transition[:, l]))
      pointers[l, i] = np.argmax(V[:, i - 1] + np.log(transition[:, l]))

  return back_trace(pointers)


def forward_probability(initial, emission, transition, alignment, groupings):
  """
  @param initial      (np.array)    initial[i] log probability of starting in
                                    state i
  @param emission     (np.array)    emission[i,j] is the log probability of
                                    releasing the observed value j from state i
  @param transition   (np.array)    transition[i,j] is the log probability from
                                    state i to state j
  @param alignment    (list)        alignment with values represented as numbers
  @param groupings    (dict)        Maps column of alignment to index
  @return             (tuple)       Probability of sequence given the parameters;
                                    forward probabilities
  """
  # Empty alignment
  _, len_alignment = alignment.shape
  if len_alignment == 0:
    return None

  f = np.zeros((len(initial), len_alignment))

  # Initialize base
  column = groupings["".join(alignment[:, 0])]
  f[:, 0] = initial + emission[:, column]

  for i in xrange(1, len_alignment):
    for l in xrange(len(initial)):
      fmax_idx = np.argmax(f[:, i - 1] + transition[:, l])
      log_fmax = f[fmax_idx, i - 1] + transition[fmax_idx, l]
      # instead of using log1p i sum over fmax as well to get e^0 = 1
      column = groupings["".join(alignment[:, i])]
      f[l, i] = emission[l, column] + log_fmax + \
          np.log(np.sum(np.exp(f[:, i - 1] + transition[:, l] - log_fmax)))

  fmax_idx = np.argmax(f[:, -1])
  log_fmax = f[fmax_idx, -1]
  P_y = log_fmax + np.log(np.sum(np.exp(f[:, -1] - log_fmax)))
  return P_y, f


def backwards_probability(initial, emission, transition, alignment, groupings):
  """
  @param initial      (np.array)    initial[i] log probability of starting in
                                    state i
  @param emission     (np.array)    emission[i,j] is the log probability of
                                    releasing the observed value j from state i
  @param transition   (np.array)    transition[i,j] is the log probability from
                                    state i to state j
  @param alignment    (list)        alignment with values represented as numbers
  @param groupings    (dict)        Maps column of alignment to index
  @return             (tuple)       Probability of sequence given the parameters;
                                    backwards probabilities
  """
  # Empty alignment
  _, len_alignment = alignment.shape
  if len_alignment == 0:
    return None

  b = np.zeros((len(initial), len_alignment))

  # Initialize base: log(1) = 0
  b[:, -1] = 0

  for i in reversed(xrange(len_alignment - 1)):
    for l in xrange(len(initial)):
      column = groupings["".join(alignment[:, i + 1])]
      bmax_idx = np.argmax(b[:, i + 1] + transition[:, l] + \
          emission[:, column]) 
      log_bmax = b[bmax_idx, i + 1] + transition[bmax_idx, l] + \
          emission[bmax_idx, column]
      # instead of using log1p i sum over fmax as well to get e^0 = 1
      b[l, i] = log_bmax + np.log(np.sum(np.exp(b[:, i + 1] + \
          transition[:, l] + emission[:, column] - log_bmax)))

  column = groupings["".join(alignment[:, 0])]
  bmax_idx = np.argmax(b[:, 0] + initial + emission[:, column])
  log_bmax = b[bmax_idx, 0] + initial[bmax_idx] + emission[bmax_idx, column]
  P_y = log_bmax + np.log(np.sum(np.exp(b[:, 0] + initial + \
      emission[:, column] - log_bmax)))
  return P_y, b


def baum_welch(initial, emission, transition, alignments, groupings):
  """
  @param initial      (np.array)    initial[i] log probability of starting in
                                    state i
  @param emission     (np.array)    starting probability for emission
  @param transition   (np.array)    starting probability for transition
  @param alignments   (list)        alignments with values represented as numbers
  @param groupings    (dict)        Maps column of alignment to index
  @return             (tuple)       Updated emission; Updated transition
  """
  num_states = len(initial)

  P = []

  delta = 0.01
  iteration = 0
  # stop if the change in log likelihood is less than some predefined threshold
  while len(P) <= 1 or np.abs(P[-1] - P[-2]) > delta:
    if iteration >= 1000:
      break

    # Set all A and E to zero
    # A = number of transitions k to l in training data
    A = np.zeros_like(transition)
    # E = number of emissions of b from k in training data
    E = np.zeros_like(emission)
    log_like = 0.0

    for alignment in alignments: # O(n)
      # Calculate forwards and backwards
      P_xj, f = forward_probability(initial, emission, transition, alignment, groupings)
      _, b = backwards_probability(initial, emission, transition, alignment, groupings)
      log_like += P_xj
      
      _, len_alignment = alignment.shape

      # Add contribution of alignment to A and E
      for k in range(num_states):
        for l in range(num_states):
          for i in range(len_alignment - 1): # O(L)
            column = groupings["".join(alignment[:, i + 1])]
            if A[k, l] == 0:
              A[k, l] = f[k, i] + transition[k, l] + emission[l, column] + \
                  b[l, i + 1] - P_xj
            else:
              A[k, l] += np.log1p(np.exp(f[k, i] + transition[k, l] + \
                  emission[l, column] + b[l, i + 1] - P_xj - A[k, l]))
        for i in range(len_alignment):
          y = groupings["".join(alignment[:, i])]
          if E[k, y] == 0:
            E[k, y] = f[k, i] + b[k, i] - P_xj
          else:
            E[k, y] += np.log1p(np.exp(f[k, i] + b[k, i] - P_xj - E[k, y]))

    # Calculate new model parameters
    argmax = np.argmax(A, axis=1)
    n, m = A.shape
    vals = np.repeat(A[np.arange(n), argmax][:, np.newaxis], m, axis=1)
    transition = A - vals - \
        np.repeat(np.log(np.sum(np.exp(A - vals), axis=1))[:, np.newaxis], m, axis=1)

    argmax = np.argmax(E, axis=1)
    n, m = E.shape
    vals = np.repeat(E[np.arange(n), argmax][:, np.newaxis], m, axis=1)
    emission = E - vals - \
        np.repeat(np.log(np.sum(np.exp(E - vals), axis=1))[:, np.newaxis], m, axis=1)

    # Save the new log likelihood
    P.append(log_like)

    iteration += 1

  print "BW Completed {} iterations".format(iteration)

  return emission, transition

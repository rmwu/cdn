import numpy as np


def combined_mat(samples1, samples2, i, cond_set):
    nsamples1 = samples1.shape[0]
    nsamples2 = samples2.shape[0]
    mat = np.zeros([nsamples1 + nsamples2, 2 + len(cond_set)])
    # === FILL FIRST COLUMN WITH SAMPLE VALUES
    mat[:nsamples1, 0] = samples1[:, i]
    mat[nsamples1:, 0] = samples2[:, i]
    # === FILL SECOND COLUMN WITH 0/1 FOR SETTING
    mat[:nsamples1, 1] = np.zeros(nsamples1)
    mat[nsamples1:, 1] = np.ones(nsamples2)
    # === FILL REMAINING COLUMNS WITH VALUES OF CONDITIONING SET
    if len(cond_set) != 0:
        mat[:nsamples1, 2:] = samples1[:, cond_set]
        mat[nsamples1:, 2:] = samples2[:, cond_set]
    return mat
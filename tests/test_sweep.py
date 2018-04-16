import os
import numpy as np
import crepe


def test_sweep():
    # this data contains a sine sweep
    file = os.path.join(os.path.dirname(__file__), 'sweep.wav')

    model = crepe.build_and_load_model()
    crepe.process_file(model, file)
    f0_file = os.path.join(os.path.dirname(__file__), 'sweep.f0.csv')

    result = np.loadtxt(f0_file, delimiter=',', skiprows=1)

    # the result should be confident enough about the presence of pitch in every
    # frame
    assert np.mean(result[:, 2] > 0.5) > 0.99

    # the frequencies should be linear
    assert np.corrcoef(result[:, 1]) > 0.99

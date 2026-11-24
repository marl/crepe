import numpy as np
import pytest

import crepe


@pytest.mark.parametrize("center,last_start,starts", [
    (True, 803, [0, 161, 322, 482, 643]),
    (False, 804, [0, 161, 322, 482, 643, 804]),
])
def test_fractional_step(center, last_start, starts):
    # A hop of 160.75 samples puts the sixth frame at sample 804.
    length = last_start if center else last_start + 1024
    audio = np.random.RandomState(0).standard_normal(length).astype(np.float32)
    step_ms = 10.046875
    time, _, _, activation = crepe.predict(
        audio, 16000, model_capacity='tiny', center=center,
        step_size=step_ms, verbose=0)

    # Run the same model on the expected windows, packed back-to-back.
    padded = np.pad(audio, 512, mode='constant') if center else audio
    windows = np.concatenate([padded[s:s + 1024] for s in starts])
    reference = crepe.get_activation(
        windows, 16000, model_capacity='tiny', center=False,
        step_size=64, verbose=0)

    np.testing.assert_allclose(time, np.arange(len(starts)) * step_ms / 1000)
    np.testing.assert_allclose(activation, reference, rtol=1e-5, atol=1e-6)

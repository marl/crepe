import numpy as np
import pytest

import crepe
from crepe import cli
from crepe import core


def synthetic_salience(frames, seed=0):
    rng = np.random.RandomState(seed)
    return rng.uniform(low=0.0, high=1.0, size=(frames, 360)).astype(np.float64)


def dense_reference_path(observations):
    structure = core._viterbi_fast_structure()
    states = 360
    xx, yy = np.meshgrid(range(states), range(states))
    transition = np.maximum(12 - abs(xx - yy), 0).astype(np.float64)
    transition = transition / np.sum(transition, axis=1)[:, None]
    log_transition = np.log(transition)
    log_starting = structure['log_starting']
    log_emission = structure['log_emission']

    prev = log_starting + log_emission[observations[0]]
    backpointers = np.empty((len(observations), states), dtype=np.int16)
    backpointers[0] = np.arange(states, dtype=np.int16)

    for frame, observation in enumerate(observations[1:], start=1):
        candidates = prev[:, None] + log_transition
        best_sources = np.argmax(candidates, axis=0)
        backpointers[frame] = best_sources.astype(np.int16)
        prev = candidates[best_sources, np.arange(states)] + log_emission[observation]

    path = np.empty((len(observations),), dtype=np.int16)
    path[-1] = int(np.argmax(prev))
    for frame in range(len(observations) - 1, 0, -1):
        path[frame - 1] = backpointers[frame, path[frame]]
    return path


class TestViterbiFast:
    def test_fast_path_matches_dense_reference(self):
        salience = synthetic_salience(64, seed=123)
        observations = np.argmax(salience, axis=1)
        fast = core._viterbi_path_fast(observations)
        dense = dense_reference_path(observations)
        np.testing.assert_array_equal(fast, dense)

    def test_fast_cents_matches_legacy(self):
        pytest.importorskip("hmmlearn")
        salience = synthetic_salience(48, seed=321)
        legacy = core.to_viterbi_cents_impl(salience, impl='legacy')
        fast = core.to_viterbi_cents_impl(salience, impl='fast')
        np.testing.assert_allclose(fast, legacy, rtol=0.0, atol=0.0)

    def test_invalid_impl_raises(self):
        salience = synthetic_salience(8, seed=7)
        with pytest.raises(ValueError):
            core.to_viterbi_cents_impl(salience, impl='nope')

    def test_predict_routes_fast_impl(self, monkeypatch):
        salience = synthetic_salience(16, seed=11)
        monkeypatch.setattr(core, 'get_activation',
                            lambda *args, **kwargs: salience)
        _, frequency, confidence, activation = core.predict(
            np.zeros((16000,), dtype=np.float32),
            16000,
            viterbi=True,
            viterbi_impl='fast',
            verbose=0)
        cents = core.to_viterbi_cents_fast(salience)
        expected_frequency = 10 * 2 ** (cents / 1200)
        np.testing.assert_allclose(frequency, expected_frequency)
        np.testing.assert_allclose(confidence, salience.max(axis=1))
        np.testing.assert_allclose(activation, salience)

    def test_cli_run_passes_viterbi_impl(self, monkeypatch, tmp_path):
        wav_path = tmp_path / 'dummy.wav'
        wav_path.write_bytes(b'RIFF')
        captured = {}

        def fake_process_file(file, **kwargs):
            captured['file'] = file
            captured['kwargs'] = kwargs

        monkeypatch.setattr(cli, 'process_file', fake_process_file)
        cli.run(
            [str(wav_path)],
            viterbi=True,
            viterbi_impl='fast',
            verbose=False)
        assert captured['file'] == str(wav_path)
        assert captured['kwargs']['viterbi'] is True
        assert captured['kwargs']['viterbi_impl'] == 'fast'

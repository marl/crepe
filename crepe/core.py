from __future__ import division
from __future__ import print_function

import os
import re
import sys

from scipy.io import wavfile
import numpy as np
from numpy.lib.stride_tricks import as_strided

# store as a global variable, since we only support a few models for now
models = {
    'tiny': None,
    'small': None,
    'medium': None,
    'large': None,
    'full': None
}

# the model is trained on 16kHz audio
model_srate = 16000

viterbi_impls = ('legacy', 'fast')


def build_and_load_model(model_capacity):
    """
    Build the CNN model and load the weights

    Parameters
    ----------
    model_capacity : 'tiny', 'small', 'medium', 'large', or 'full'
        String specifying the model capacity, which determines the model's
        capacity multiplier to 4 (tiny), 8 (small), 16 (medium), 24 (large),
        or 32 (full). 'full' uses the model size specified in the paper,
        and the others use a reduced number of filters in each convolutional
        layer, resulting in a smaller model that is faster to evaluate at the
        cost of slightly reduced pitch estimation accuracy.

    Returns
    -------
    model : tensorflow.keras.models.Model
        The pre-trained keras model loaded in memory
    """
    from tensorflow.keras.layers import Input, Reshape, Conv2D, BatchNormalization
    from tensorflow.keras.layers import MaxPool2D, Dropout, Permute, Flatten, Dense
    from tensorflow.keras.models import Model

    if models[model_capacity] is None:
        capacity_multiplier = {
            'tiny': 4, 'small': 8, 'medium': 16, 'large': 24, 'full': 32
        }[model_capacity]

        layers = [1, 2, 3, 4, 5, 6]
        filters = [n * capacity_multiplier for n in [32, 4, 4, 4, 8, 16]]
        widths = [512, 64, 64, 64, 64, 64]
        strides = [(4, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1)]

        x = Input(shape=(1024,), name='input', dtype='float32')
        y = Reshape(target_shape=(1024, 1, 1), name='input-reshape')(x)

        for l, f, w, s in zip(layers, filters, widths, strides):
            y = Conv2D(f, (w, 1), strides=s, padding='same',
                       activation='relu', name="conv%d" % l)(y)
            y = BatchNormalization(name="conv%d-BN" % l)(y)
            y = MaxPool2D(pool_size=(2, 1), strides=None, padding='valid',
                          name="conv%d-maxpool" % l)(y)
            y = Dropout(0.25, name="conv%d-dropout" % l)(y)

        y = Permute((2, 1, 3), name="transpose")(y)
        y = Flatten(name="flatten")(y)
        y = Dense(360, activation='sigmoid', name="classifier")(y)

        model = Model(inputs=x, outputs=y)

        package_dir = os.path.dirname(os.path.realpath(__file__))
        filename = "model-{}.h5".format(model_capacity)
        model.load_weights(os.path.join(package_dir, filename))
        model.compile('adam', 'binary_crossentropy')

        models[model_capacity] = model

    return models[model_capacity]


def output_path(file, suffix, output_dir):
    """
    return the output path of an output file corresponding to a wav file
    """
    path = re.sub(r"(?i).wav$", suffix, file)
    if output_dir is not None:
        path = os.path.join(output_dir, os.path.basename(path))
    return path


def to_local_average_cents(salience, center=None):
    """
    find the weighted average cents near the argmax bin
    """

    if not hasattr(to_local_average_cents, 'cents_mapping'):
        # the bin number-to-cents mapping
        to_local_average_cents.cents_mapping = (
                np.linspace(0, 7180, 360) + 1997.3794084376191)

    if salience.ndim == 1:
        if center is None:
            center = int(np.argmax(salience))
        start = max(0, center - 4)
        end = min(len(salience), center + 5)
        salience = salience[start:end]
        product_sum = np.sum(
            salience * to_local_average_cents.cents_mapping[start:end])
        weight_sum = np.sum(salience)
        return product_sum / weight_sum
    if salience.ndim == 2:
        return np.array([to_local_average_cents(salience[i, :]) for i in
                         range(salience.shape[0])])

    raise Exception("label should be either 1d or 2d ndarray")


def to_viterbi_cents(salience):
    """
    Find the Viterbi path using a transition prior that induces pitch
    continuity.
    """
    return to_viterbi_cents_impl(salience, impl='legacy')


def to_viterbi_cents_impl(salience, impl='legacy'):
    """
    Find the Viterbi path using the requested implementation.
    """
    if impl == 'legacy':
        return to_viterbi_cents_legacy(salience)
    if impl == 'fast':
        return to_viterbi_cents_fast(salience)
    raise ValueError('expected viterbi_impl to be one of {}, got {}'.format(
        viterbi_impls, impl))


def to_viterbi_cents_legacy(salience):
    """
    Legacy hmmlearn-backed Viterbi smoothing path.
    """
    from hmmlearn import hmm

    # uniform prior on the starting pitch
    starting = np.ones(360) / 360

    # transition probabilities inducing continuous pitch
    xx, yy = np.meshgrid(range(360), range(360))
    transition = np.maximum(12 - abs(xx - yy), 0)
    transition = transition / np.sum(transition, axis=1)[:, None]

    # emission probability = fixed probability for self, evenly distribute the
    # others
    self_emission = 0.1
    emission = (np.eye(360) * self_emission + np.ones(shape=(360, 360)) *
                ((1 - self_emission) / 360))

    # fix the model parameters because we are not optimizing the model
    model = hmm.CategoricalHMM(360, starting, transition)
    model.startprob_, model.transmat_, model.emissionprob_ = \
        starting, transition, emission

    # find the Viterbi path
    observations = np.argmax(salience, axis=1)
    path = model.predict(observations.reshape(-1, 1), [len(observations)])

    return np.array([to_local_average_cents(salience[i, :], path[i]) for i in
                     range(len(observations))])


def to_viterbi_cents_fast(salience):
    """
    Exact structured Viterbi smoothing path for CREPE's local transition graph.
    """
    observations = np.argmax(salience, axis=1).astype(np.int64, copy=False)
    path = _viterbi_path_fast(observations)
    return np.array([to_local_average_cents(salience[i, :], path[i]) for i in
                     range(len(observations))])


def _viterbi_fast_structure():
    """
    Precompute the exact local predecessor structure for CREPE's transition.
    """
    cached = getattr(_viterbi_fast_structure, 'cached', None)
    if cached is not None:
        return cached

    states = 360
    starting = np.ones(states, dtype=np.float64) / states

    xx, yy = np.meshgrid(range(states), range(states))
    transition = np.maximum(12 - abs(xx - yy), 0).astype(np.float64)
    transition = transition / np.sum(transition, axis=1)[:, None]

    self_emission = 0.1
    emission = (np.eye(states, dtype=np.float64) * self_emission +
                np.ones(shape=(states, states), dtype=np.float64) *
                ((1 - self_emission) / states))

    valid = transition > 0
    width = int(np.max(np.sum(valid, axis=0)))
    source_idx = np.zeros((states, width), dtype=np.int16)
    log_trans = np.full((states, width), -np.inf, dtype=np.float64)

    for target in range(states):
        sources = np.flatnonzero(valid[:, target]).astype(np.int16)
        source_idx[target, :len(sources)] = sources
        log_trans[target, :len(sources)] = np.log(
            transition[sources.astype(np.int64), target])

    cached = {
        'state_idx': np.arange(states, dtype=np.int64),
        'log_starting': np.log(starting),
        'source_idx': source_idx,
        'log_trans': log_trans,
        'log_emission': np.log(emission)
    }
    _viterbi_fast_structure.cached = cached
    return cached


def _viterbi_path_fast(observations):
    """
    Exact structured Viterbi decode of CREPE's argmax observations.
    """
    structure = _viterbi_fast_structure()
    source_idx = structure['source_idx']
    log_trans = structure['log_trans']
    state_idx = structure['state_idx']
    log_emission = structure['log_emission']

    prev = structure['log_starting'] + log_emission[observations[0]]
    backpointers = np.empty((len(observations), 360), dtype=np.int16)
    backpointers[0] = np.arange(360, dtype=np.int16)

    for frame, observation in enumerate(observations[1:], start=1):
        candidates = prev[source_idx] + log_trans
        best_offsets = np.argmax(candidates, axis=1)
        best_sources = source_idx[state_idx, best_offsets]
        backpointers[frame] = best_sources
        prev = candidates[state_idx, best_offsets] + log_emission[observation]

    path = np.empty((len(observations),), dtype=np.int16)
    path[-1] = int(np.argmax(prev))
    for frame in range(len(observations) - 1, 0, -1):
        path[frame - 1] = backpointers[frame, path[frame]]
    return path


def get_activation(audio, sr, model_capacity='full', center=True, step_size=10,
                   verbose=1):
    """

    Parameters
    ----------
    audio : np.ndarray [shape=(N,) or (N, C)]
        The audio samples. Multichannel audio will be downmixed.
    sr : int
        Sample rate of the audio samples. The audio will be resampled if
        the sample rate is not 16 kHz, which is expected by the model.
    model_capacity : 'tiny', 'small', 'medium', 'large', or 'full'
        String specifying the model capacity; see the docstring of
        :func:`~crepe.core.build_and_load_model`
    center : boolean
        - If `True` (default), the signal `audio` is padded so that frame
          `D[:, t]` is centered at `audio[t * hop_length]`.
        - If `False`, then `D[:, t]` begins at `audio[t * hop_length]`
    step_size : int
        The step size in milliseconds for running pitch estimation.
    verbose : int
        Set the keras verbosity mode: 1 (default) will print out a progress bar
        during prediction, 0 will suppress all non-error printouts.

    Returns
    -------
    activation : np.ndarray [shape=(T, 360)]
        The raw activation matrix
    """
    model = build_and_load_model(model_capacity)

    if len(audio.shape) == 2:
        audio = audio.mean(1)  # make mono
    audio = audio.astype(np.float32)
    if sr != model_srate:
        # resample audio if necessary
        from resampy import resample
        audio = resample(audio, sr, model_srate)

    # pad so that frames are centered around their timestamps (i.e. first frame
    # is zero centered).
    if center:
        audio = np.pad(audio, 512, mode='constant', constant_values=0)

    # make 1024-sample frames of the audio with hop length of 10 milliseconds
    hop_length = int(model_srate * step_size / 1000)
    n_frames = 1 + int((len(audio) - 1024) / hop_length)
    frames = as_strided(audio, shape=(1024, n_frames),
                        strides=(audio.itemsize, hop_length * audio.itemsize))
    frames = frames.transpose().copy()

    # normalize each frame -- this is expected by the model
    frames -= np.mean(frames, axis=1)[:, np.newaxis]
    frames /= np.clip(np.std(frames, axis=1)[:, np.newaxis], 1e-8, None)

    # run prediction and convert the frequency bin weights to Hz
    return model.predict(frames, verbose=verbose)


def predict(audio, sr, model_capacity='full',
            viterbi=False, center=True, step_size=10, verbose=1,
            viterbi_impl='legacy'):
    """
    Perform pitch estimation on given audio

    Parameters
    ----------
    audio : np.ndarray [shape=(N,) or (N, C)]
        The audio samples. Multichannel audio will be downmixed.
    sr : int
        Sample rate of the audio samples. The audio will be resampled if
        the sample rate is not 16 kHz, which is expected by the model.
    model_capacity : 'tiny', 'small', 'medium', 'large', or 'full'
        String specifying the model capacity; see the docstring of
        :func:`~crepe.core.build_and_load_model`
    viterbi : bool
        Apply viterbi smoothing to the estimated pitch curve. False by default.
    viterbi_impl : {'legacy', 'fast'}
        Implementation used when `viterbi=True`. Defaults to the current
        `hmmlearn` path (`legacy`).
    center : boolean
        - If `True` (default), the signal `audio` is padded so that frame
          `D[:, t]` is centered at `audio[t * hop_length]`.
        - If `False`, then `D[:, t]` begins at `audio[t * hop_length]`
    step_size : int
        The step size in milliseconds for running pitch estimation.
    verbose : int
        Set the keras verbosity mode: 1 (default) will print out a progress bar
        during prediction, 0 will suppress all non-error printouts.

    Returns
    -------
    A 4-tuple consisting of:

        time: np.ndarray [shape=(T,)]
            The timestamps on which the pitch was estimated
        frequency: np.ndarray [shape=(T,)]
            The predicted pitch values in Hz
        confidence: np.ndarray [shape=(T,)]
            The confidence of voice activity, between 0 and 1
        activation: np.ndarray [shape=(T, 360)]
            The raw activation matrix
    """
    activation = get_activation(audio, sr, model_capacity=model_capacity,
                                center=center, step_size=step_size,
                                verbose=verbose)
    confidence = activation.max(axis=1)

    if viterbi:
        cents = to_viterbi_cents_impl(activation, impl=viterbi_impl)
    else:
        cents = to_local_average_cents(activation)

    frequency = 10 * 2 ** (cents / 1200)
    frequency[np.isnan(frequency)] = 0

    time = np.arange(confidence.shape[0]) * step_size / 1000.0

    return time, frequency, confidence, activation


def process_file(file, output=None, model_capacity='full', viterbi=False,
                 center=True, save_activation=False, save_plot=False,
                 plot_voicing=False, step_size=10, verbose=True,
                 viterbi_impl='legacy'):
    """
    Use the input model to perform pitch estimation on the input file.

    Parameters
    ----------
    file : str
        Path to WAV file to be analyzed.
    output : str or None
        Path to directory for saving output files. If None, output files will
        be saved to the directory containing the input file.
    model_capacity : 'tiny', 'small', 'medium', 'large', or 'full'
        String specifying the model capacity; see the docstring of
        :func:`~crepe.core.build_and_load_model`
    viterbi : bool
        Apply viterbi smoothing to the estimated pitch curve. False by default.
    viterbi_impl : {'legacy', 'fast'}
        Implementation used when `viterbi=True`.
    center : boolean
        - If `True` (default), the signal `audio` is padded so that frame
          `D[:, t]` is centered at `audio[t * hop_length]`.
        - If `False`, then `D[:, t]` begins at `audio[t * hop_length]`
    save_activation : bool
        Save the output activation matrix to an .npy file. False by default.
    save_plot : bool
        Save a plot of the output activation matrix to a .png file. False by
        default.
    plot_voicing : bool
        Include a visual representation of the voicing activity detection in
        the plot of the output activation matrix. False by default, only
        relevant if save_plot is True.
    step_size : int
        The step size in milliseconds for running pitch estimation.
    verbose : bool
        Print status messages and keras progress (default=True).

    Returns
    -------

    """
    try:
        sr, audio = wavfile.read(file)
    except ValueError:
        print("CREPE: Could not read %s" % file, file=sys.stderr)
        raise

    time, frequency, confidence, activation = predict(
        audio, sr,
        model_capacity=model_capacity,
        viterbi=viterbi,
        viterbi_impl=viterbi_impl,
        center=center,
        step_size=step_size,
        verbose=1 * verbose)

    # write prediction as TSV
    f0_file = output_path(file, ".f0.csv", output)
    f0_data = np.vstack([time, frequency, confidence]).transpose()
    np.savetxt(f0_file, f0_data, fmt=['%.3f', '%.3f', '%.6f'], delimiter=',',
               header='time,frequency,confidence', comments='')
    if verbose:
        print("CREPE: Saved the estimated frequencies and confidence values "
              "at {}".format(f0_file))

    # save the salience file to a .npy file
    if save_activation:
        activation_path = output_path(file, ".activation.npy", output)
        np.save(activation_path, activation)
        if verbose:
            print("CREPE: Saved the activation matrix at {}".format(
                activation_path))

    # save the salience visualization in a PNG file
    if save_plot:
        import matplotlib.cm
        from imageio import imwrite

        plot_file = output_path(file, ".activation.png", output)
        # to draw the low pitches in the bottom
        salience = np.flip(activation, axis=1)
        inferno = matplotlib.cm.get_cmap('inferno')
        image = inferno(salience.transpose())

        if plot_voicing:
            # attach a soft and hard voicing detection result under the
            # salience plot
            image = np.pad(image, [(0, 20), (0, 0), (0, 0)], mode='constant')
            image[-20:-10, :, :] = inferno(confidence)[np.newaxis, :, :]
            image[-10:, :, :] = (
                inferno((confidence > 0.5).astype(np.float))[np.newaxis, :, :])

        imwrite(plot_file, (255 * image).astype(np.uint8))
        if verbose:
            print("CREPE: Saved the salience plot at {}".format(plot_file))

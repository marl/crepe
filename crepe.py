#! /usr/bin/env python

from __future__ import print_function

import argparse
import os
import re
import sys
from argparse import RawDescriptionHelpFormatter
import numpy as np
from numpy.lib.stride_tricks import as_strided
from hmmlearn import hmm
from resampy import resample
from scipy.io import wavfile
import matplotlib.cm
from imageio import imwrite


def build_and_load_model():
    """
    Build the CNN model and load the weights; this needs to exactly match
    what's saved in the Keras weights file
    """
    from keras.layers import Input, Reshape, Conv2D, BatchNormalization, \
        MaxPooling2D, Dropout, Permute, Flatten, Dense
    from keras.models import Model

    model_capacity = 32
    layers = [1, 2, 3, 4, 5, 6]
    filters = [n * model_capacity for n in [32, 4, 4, 4, 8, 16]]
    widths = [512, 64, 64, 64, 64, 64]
    strides = [(4, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1)]

    x = Input(shape=(1024,), name='input', dtype='float32')
    y = Reshape(target_shape=(1024, 1, 1), name='input-reshape')(x)

    for layer, filters, width, strides in zip(layers, filters, widths, strides):
        y = Conv2D(filters, (width, 1), strides=strides, padding='same',
                   activation='relu', name="conv%d" % layer)(y)
        y = BatchNormalization(name="conv%d-BN" % layer)(y)
        y = MaxPooling2D(pool_size=(2, 1), strides=None, padding='valid',
                         name="conv%d-maxpool" % layer)(y)
        y = Dropout(0.25, name="conv%d-dropout" % layer)(y)

    y = Permute((2, 1, 3), name="transpose")(y)
    y = Flatten(name="flatten")(y)
    y = Dense(360, activation='sigmoid', name="classifier")(y)

    model = Model(inputs=x, outputs=y)
    model.load_weights("model.h5")
    model.compile('adam', 'binary_crossentropy')

    return model


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
        to_local_average_cents.mapping = (
                np.linspace(0, 7180, 360) + 1997.3794084376191)

    if salience.ndim == 1:
        if center is None:
            center = int(np.argmax(salience))
        start = max(0, center - 4)
        end = min(len(salience), center + 5)
        salience = salience[start:end]
        product_sum = np.sum(
            salience * to_local_average_cents.mapping[start:end])
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
    model = hmm.MultinomialHMM(360, starting, transition)
    model.startprob_, model.transmat_, model.emissionprob_ = \
        starting, transition, emission

    # find the Viterbi path
    observations = np.argmax(salience, axis=1)
    path = model.predict(observations.reshape(-1, 1), [len(observations)])

    return np.array([to_local_average_cents(salience[i, :], path[i]) for i in
                     range(len(observations))])


def process_file(model, file, output=None, viterbi=False,
                 save_activation=False, save_plot=False, plot_voicing=False):
    """
    Use the input model to perform pitch estimation on the input file.

    Parameters
    ----------
    model : Keras Model object
        Pre-trained CREPE model to use for prediction, as returned by
        crepe.build_and_load_model()
    file : str
        Path to WAV file to be analyzed.
    output : str or None
        Path to directory for saving output files. If None, output files will
        be saved to the directory containing the input file.
    viterbi : bool
        Apply viterbi smoothing to the estimated pitch curve. False by default.
    save_activation : bool
        Save the output activation matrix to an .npy file. False by default.
    save_plot: bool
        Save a plot of the output activation matrix to a .png file. False by
        default.
    plot_voicing : bool
        Include a visual representation of the voicing activity detection in
        the plot of the output activation matrix. False by default, only
        relevant if save_plot is True.

    Returns
    -------

    """



    model_srate = 16000  # the model is trained on 16kHz audio

    try:
        srate, data = wavfile.read(file)
        if len(data.shape) == 2:
            data = data.mean(1)  # make mono
        data = data.astype(np.float32)
        if srate != model_srate:
            # resample audio if necessary
            data = resample(data, srate, model_srate)
    except ValueError:
        print("CREPE: Could not read %s" % file, file=sys.stderr)
        raise

    # make 1024-sample frames of the audio with hop length of 10 milliseconds
    hop_length = int(model_srate / 100)
    n_frames = 1 + int((len(data) - 1024) / hop_length)
    frames = as_strided(data, shape=(1024, n_frames),
                        strides=(data.itemsize, hop_length * data.itemsize))
    frames = frames.transpose()

    # normalize each frame -- this is expected by the model
    frames -= np.mean(frames, axis=1)[:, np.newaxis]
    frames /= np.std(frames, axis=1)[:, np.newaxis]

    # run prediction and convert the frequency bin weights to Hz
    salience = model.predict(frames, verbose=1)
    confidence = np.max(salience, axis=1)

    if viterbi:
        prediction_cents = to_viterbi_cents(salience)
    else:
        prediction_cents = to_local_average_cents(salience)

    prediction_hz = 10 * 2 ** (prediction_cents / 1200)
    prediction_hz[np.isnan(prediction_hz)] = 0

    # write prediction as TSV
    f0_file = output_path(file, ".f0.csv", output)
    with open(f0_file, 'w') as out:
        print('time,frequency,confidence', file=out)
        for i, freq in enumerate(prediction_hz):
            print("%.2f,%.3f,%.6f" % (i * 0.01, freq, confidence[i]), file=out)
    print("CREPE: Saved the estimated frequencies and confidence values at "
          "{}".format(f0_file))

    # save the salience file to a .npy file
    if save_activation:
        activation_path = output_path(file, ".activation.npy", output)
        np.save(activation_path, salience)
        print("CREPE: Saved the activation matrix at {}".format(
            activation_path))

    # save the salience visualization in a PNG file
    if save_plot:

        plot_file = output_path(file, ".activation.png", output)
        # to draw the low pitches in the bottom
        salience = np.flip(salience, axis=1)
        inferno = matplotlib.cm.get_cmap('inferno')
        image = inferno(salience.transpose())

        if plot_voicing:
            # attach a soft and hard voicing detection result under the
            # salience plot
            image = np.pad(image, [(0, 20), (0, 0), (0, 0)], mode='constant')
            image[-20:-10, :, :] = inferno(confidence)[np.newaxis, :, :]
            image[-10:, :, :] = (
                inferno((confidence > 0.5).astype(np.float))[np.newaxis, :, :])

        imwrite(plot_file, 255 * image)
        print("CREPE: Saved the salience plot at {}".format(plot_file))


def main(filename, output=None, viterbi=False, save_activation=False,
         save_plot=False, plot_voicing=False):
    """
    Collect the WAV files to process and run the model

    Parameters
    ----------
    filename : list
        List containing paths to WAV files or folders containing WAV files to
        be analyzed.
    output : str or None
        Path to directory for saving output files. If None, output files will
        be saved to the directory containing the input file.
    viterbi : bool
        Apply viterbi smoothing to the estimated pitch curve. False by default.
    save_activation : bool
        Save the output activation matrix to an .npy file. False by default.
    save_plot: bool
        Save a plot of the output activation matrix to a .png file. False by
        default.
    plot_voicing : bool
        Include a visual representation of the voicing activity detection in
        the plot of the output activation matrix. False by default, only
        relevant if save_plot is True.

    """

    files = []
    for path in filename:
        if os.path.isdir(path):
            found = ([file for file in os.listdir(path) if
                      file.lower().endswith('.wav')])
            if len(found) == 0:
                print('CREPE: No WAV files found in directory {}'.format(path),
                      file=sys.stderr)
            files += [os.path.join(path, file) for file in found]
        elif os.path.isfile(path):
            if not path.lower().endswith('.wav'):
                print('CREPE: Expecting WAV file(s) but got {}'.format(path),
                      file=sys.stderr)
            files.append(path)
        else:
            print('CREPE: File or directory not found: {}'.format(path),
                  file=sys.stderr)

    if len(files) == 0:
        print('CREPE: No WAV files found in {}, aborting.'.format(filename))
        sys.exit(-1)

    # Load pre-trained CREPE model
    model = build_and_load_model()

    for i, file in enumerate(files):
        print('CREPE: Processing {} ... ({}/{})'.format(file, i+1, len(files)),
              file=sys.stderr)
        process_file(model, file, output, viterbi, save_activation, save_plot,
                     plot_voicing)


if __name__ == "__main__":

    description = """
    This is a script for running the pre-trained pitch estimation model, CREPE,
    by taking WAV files(s) as input. For each input WAV, a CSV file containing:

        time, frequency, confidence
        0.00, 424.24, 0.42
        0.01, 422.42, 0.84
        ...

    is created as the output, where the first column is a timestamp in seconds,
    the second column is the estimated frequency in Hz, and the third column is 
    a value between 0 and 1 indicating the model's voicing confidence (i.e. 
    confidence in the presence of a pitch for every frame).

    The script can also optionally save the output activation matrix of the 
    model to an npy file, where the matrix dimensions are (n_frames, 360) using
    a hop size of 10 ms (there are 360 pitch bins covering 20 cents each).
    The script can also output a plot of the activation matrix, including an 
    optional visual representation of the model's voicing detection. 
    """

    parser = argparse.ArgumentParser(
        sys.argv[0], description=description,
        formatter_class=RawDescriptionHelpFormatter)

    parser.add_argument('filename', nargs='+',
                        help='path to one ore more WAV file(s) to analyze OR '
                             'can be a directory')
    parser.add_argument('--output', '-o', default=None,
                        help='directory to save the ouptut file(s), must '
                             'already exist; if not given, the output will be '
                             'saved to the same directory as the input WAV '
                             'file(s)')
    parser.add_argument('--viterbi', '-V', action='store_true', default=False,
                        help='perform Viterbi decoding to smooth the pitch '
                             'curve')
    parser.add_argument('--save-activation', '-a', action='store_true',
                        default=False,
                        help='save the output activation matrix to a .npy '
                             'file')
    parser.add_argument('--save-plot', '-p', action='store_true',
                        default=False,
                        help='save a plot of the activation matrix to a .png '
                             'file')
    parser.add_argument('--plot-voicing', '-v', action='store_true',
                        default=False,
                        help='Plot the voicing prediction on top of the '
                             'output activation matrix plot')

    args = parser.parse_args()

    main(args.filename, args.output, args.viterbi, args.save_activation,
         args.save_plot, args.plot_voicing)

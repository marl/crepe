#! /usr/bin/env python

from __future__ import print_function

import argparse
import os
import re
import sys
from argparse import RawDescriptionHelpFormatter

description = """
This is a script for running the pre-trained pitch estimation model, CREPE,
by taking WAV files(s) as input. For each input WAV, a CSV file containing:

    # time, frequency, confidence
    0.00, 424.24, 0.42
    0.01, 422.42, 0.84
    ...
    
is created as the output, where the third column is a value between 0 and 1
which indicates the model's confidence of a presence of voicing.
 
In addition, a salience plot visualizing the activations in the last layer
is saved to a PNG image of size T*360 pixels, where T is the number of 
frames used in pitch estimation, which is 10 milliseconds by default.
"""

parser = argparse.ArgumentParser(sys.argv[0], description=description, formatter_class=RawDescriptionHelpFormatter)
parser.add_argument('wav_file_path', nargs='+',
                    help='path to the WAV file(s) to run pitch estimation; can be a directory.')
parser.add_argument('--output-dir', '-o', default=None,
                    help='directory to save the resulting files; '
                         'if not given, the output will be produced in the same directory as the input WAV file(s).')
parser.add_argument('--skip-salience', '-s', action='store_true',
                    help='skip saving the salience visualization in PNG')
parser.add_argument('--attach-voicing', '-v', action='store_true',
                    help='include the voicing predictions the salience plot.')
parser.add_argument('--save-numpy', '-n', action='store_true',
                    help='save the raw activation matrix to a .npy file')
parser.add_argument('--viterbi', '-V', action='store_true',
                    help='perform Viterbi decoding to smooth the pitch curve')


def build_and_load_model():
    """Build the CNN model and load the weights; this needs to exactly match what's saved in the Keras weights file"""
    from keras.layers import Input, Reshape, Conv2D, BatchNormalization, MaxPooling2D, Dropout, Permute, Flatten, Dense
    from keras.models import Model

    model_capacity = 32
    layers = [1, 2, 3, 4, 5, 6]
    filters = [n * model_capacity for n in [32, 4, 4, 4, 8, 16]]
    widths = [512, 64, 64, 64, 64, 64]
    strides = [(4, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1)]

    x = Input(shape=(1024,), name='input', dtype='float32')
    y = Reshape(target_shape=(1024, 1, 1), name='input-reshape')(x)

    for layer, filters, width, strides in zip(layers, filters, widths, strides):
        y = Conv2D(filters, (width, 1), strides=strides, padding='same', activation='relu', name="conv%d" % layer)(y)
        y = BatchNormalization(name="conv%d-BN" % layer)(y)
        y = MaxPooling2D(pool_size=(2, 1), strides=None, padding='valid', name="conv%d-maxpool" % layer)(y)
        y = Dropout(0.25, name="conv%d-dropout" % layer)(y)

    y = Permute((2, 1, 3), name="transpose")(y)
    y = Flatten(name="flatten")(y)
    y = Dense(360, activation='sigmoid', name="classifier")(y)

    model = Model(inputs=x, outputs=y)
    model.load_weights("model.h5")
    model.compile('adam', 'binary_crossentropy')

    return model


def output_path(file, suffix, output_dir):
    """return the output path of an output file corresponding to a wav file"""
    path = re.sub(r"(?i).wav$", suffix, file)
    if output_dir is not None:
        path = os.path.join(output_dir, os.path.basename(path))
    return path


def to_local_average_cents(label):
    """find the weighted average cents near the argmax bin"""

    import numpy as np

    if not hasattr(to_local_average_cents, 'cents_mapping'):
        # the bin number-to-cents mapping
        to_local_average_cents.mapping = np.linspace(0, 7180, 360) + 1997.3794084376191

    if label.ndim == 1:
        argmax = int(np.argmax(label))
        start = max(0, argmax - 4)
        end = min(len(label), argmax + 5)
        label = label[start:end]
        product_sum = np.sum(label * to_local_average_cents.mapping[start:end])
        weight_sum = np.sum(label)
        return product_sum / weight_sum
    if label.ndim == 2:
        return np.array([to_local_average_cents(label[i, :]) for i in range(label.shape[0])])

    raise Exception("label should be either 1d or 2d ndarray")


def process_file(model, file, args):
    """Perform pitch estimation on the file"""

    import matplotlib.cm
    import numpy as np
    from numpy.lib.stride_tricks import as_strided
    from resampy import resample
    from scipy.io import wavfile

    model_srate = 16000  # the model is trained on 16kHz audio

    try:
        srate, data = wavfile.read(file)
        if len(data.shape) == 2:
            data = data.mean(1)  # make mono
        data = data.astype(np.float32)
        if srate != model_srate:
            data = resample(data, srate, model_srate)  # resample audio if necessary
    except ValueError:
        print("CREPE: Could not read %s" % file, file=sys.stderr)
        raise

    # make 1024-sample frames of the audio with hop length of 10 milliseconds
    hop_length = int(model_srate / 100)
    n_frames = 1 + int((len(data) - 1024) / hop_length)
    frames = as_strided(data, shape=(1024, n_frames), strides=(data.itemsize, hop_length * data.itemsize))
    frames = frames.transpose()

    # normalize each frame -- this is expected by the model
    frames -= np.mean(frames, axis=1)[:, np.newaxis]
    frames /= np.std(frames, axis=1)[:, np.newaxis]

    # run prediction and convert the frequency bin weights to Hz
    salience = model.predict(frames, verbose=1)
    confidence = np.max(salience, axis=1)

    prediction_hz = 10 * 2 ** (to_local_average_cents(salience) / 1200)
    prediction_hz[np.isnan(prediction_hz)] = 0

    # write prediction as TSV
    f0_file = output_path(file, ".f0.csv", args.output_dir)
    with open(f0_file, 'w') as out:
        print('# time,frequency,confidence', file=out)
        for i, freq in enumerate(prediction_hz):
            print("%.2f,%.3f,%.6f" % (i * 0.01, freq, confidence[i]), file=out)
    print("CREPE: Saved the estimated frequencies and confidence values at {}".format(f0_file))

    # save the salience file to a .npy file
    if args.save_numpy:
        salience_path = output_path(file, ".salience.npy", args.output_dir)
        np.save(salience_path, salience)
        print("CREPE: Saved the salience matrix at {}".format(salience_path))

    # save the salience visualization in a PNG file
    if not args.skip_salience:
        from scipy.misc import imsave
        
        plot_file = output_path(file, ".salience.png", args.output_dir)
        salience = np.flip(salience, axis=1)  # to draw the low pitches in the bottom
        inferno = matplotlib.cm.get_cmap('inferno')
        image = inferno(salience.transpose())

        if args.attach_voicing:
            # attach a soft and hard voicing detection result under the salience plot
            image = np.pad(image, [(0, 20), (0, 0), (0, 0)], mode='constant')
            image[-20:-10, :, :] = inferno(confidence)[np.newaxis, :, :]
            image[-10:, :, :] = inferno((confidence > 0.5).astype(np.float))[np.newaxis, :, :]

        imsave(plot_file, 255 * image)
        print("CREPE: Saved the salience plot at {}".format(plot_file))


def main():
    """the main procedure; collect the WAV files to process and run the model"""
    args = parser.parse_args()
    files = []
    for path in args.wav_file_path:
        if os.path.isdir(path):
            found = [file for file in os.listdir(path) if file.lower().endswith('.wav')]
            if len(found) == 0:
                print('CREPE: No WAV files found in directory {}'.format(path), file=sys.stderr)
            files += [os.path.join(path, file) for file in found]
        elif os.path.isfile(path):
            if not path.lower().endswith('.wav'):
                print('CREPE: Expecting WAV file(s) but got {}'.format(path), file=sys.stderr)
            files.append(path)
        else:
            print('CREPE: File or directory not found: {}'.format(path), file=sys.stderr)

    if len(files) == 0:
        print('CREPE: No WAV files found in {}, aborting.'.format(args.wav_file_path))
        sys.exit(-1)

    model = build_and_load_model()

    for i, file in enumerate(files):
        print('CREPE: Processing {} ... ({}/{})'.format(file, i+1, len(files)), file=sys.stderr)
        process_file(model, file, args)


if __name__ == "__main__":
    main()

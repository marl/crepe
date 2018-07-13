from __future__ import print_function

import os
import sys
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from argparse import ArgumentTypeError

from .core import process_file


def run(filename, output=None, model_capacity='full', viterbi=False,
        save_activation=False, save_plot=False, plot_voicing=False,
        no_centering=False, step_size=10, verbose=True):
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
    model_capacity : 'tiny', 'small', 'medium', 'large', or 'full'
        String specifying the model capacity; see the docstring of
        :func:`~crepe.core.build_and_load_model`
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
    no_centering : bool
        Don't pad the signal, meaning frames will begin at their timestamp
        instead of being centered around their timestamp (which is the
        default). CAUTION: setting this option can result in CREPE's output
        being misaligned with respect to the output of other audio processing
        tools and is generally not recommended.
    step_size : int
        The step size in milliseconds for running pitch estimation.
    verbose : bool
        Print status messages and keras progress (default=True).
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

    for i, file in enumerate(files):
        if verbose:
            print('CREPE: Processing {} ... ({}/{})'.format(
                file, i+1, len(files)), file=sys.stderr)
        process_file(file, output=output,
                     model_capacity=model_capacity,
                     viterbi=viterbi,
                     center=(not no_centering),
                     save_activation=save_activation,
                     save_plot=save_plot,
                     plot_voicing=plot_voicing,
                     step_size=step_size,
                     verbose=verbose)


def positive_int(value):
    """An argparse type method for accepting only positive integers"""
    ivalue = int(value)
    if ivalue <= 0:
        raise ArgumentTypeError('expected a positive integer')
    return ivalue


def main():
    """
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

    parser = ArgumentParser(sys.argv[0], description=main.__doc__,
                            formatter_class=RawDescriptionHelpFormatter)

    parser.add_argument('filename', nargs='+',
                        help='path to one ore more WAV file(s) to analyze OR '
                             'can be a directory')
    parser.add_argument('--output', '-o', default=None,
                        help='directory to save the ouptut file(s), must '
                             'already exist; if not given, the output will be '
                             'saved to the same directory as the input WAV '
                             'file(s)')
    parser.add_argument('--model-capacity', '-c', default='full',
                        choices=['tiny', 'small', 'medium', 'large', 'full'],
                        help='String specifying the model capacity; smaller '
                             'models are faster to compute, but may yield '
                             'less accurate pitch estimation')
    parser.add_argument('--viterbi', '-V', action='store_true',
                        help='perform Viterbi decoding to smooth the pitch '
                             'curve')
    parser.add_argument('--save-activation', '-a', action='store_true',
                        help='save the output activation matrix to a .npy '
                             'file')
    parser.add_argument('--save-plot', '-p', action='store_true',
                        help='save a plot of the activation matrix to a .png '
                             'file')
    parser.add_argument('--plot-voicing', '-v', action='store_true',
                        help='Plot the voicing prediction on top of the '
                             'output activation matrix plot')
    parser.add_argument('--no-centering', '-n', action='store_true',
                        help="Don't pad the signal, meaning frames will begin "
                             "at their timestamp instead of being centered "
                             "around their timestamp (which is the default). "
                             "CAUTION: setting this option can result in "
                             "CREPE's output being misaligned with respect to "
                             "the output of other audio processing tools and "
                             "is generally not recommended.")
    parser.add_argument('--step-size', '-s', default=10, type=positive_int,
                        help='The step size in milliseconds for running '
                             'pitch estimation. The default is 10 ms.')
    parser.add_argument('--quiet', '-q', default=False,
                        action='store_true',
                        help='Suppress all non-error printouts (e.g. progress '
                             'bar).')

    args = parser.parse_args()

    run(args.filename,
        output=args.output,
        model_capacity=args.model_capacity,
        viterbi=args.viterbi,
        save_activation=args.save_activation,
        save_plot=args.save_plot,
        plot_voicing=args.plot_voicing,
        no_centering=args.no_centering,
        step_size=args.step_size,
        verbose=not args.quiet)

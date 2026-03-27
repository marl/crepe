from __future__ import print_function

import argparse
import os
import time

import numpy as np
from scipy.io import wavfile

import crepe
from crepe import core


def time_call(fn, warmup, repeats):
    last = None
    for _ in range(warmup):
        last = fn()
    t0 = time.perf_counter()
    for _ in range(repeats):
        last = fn()
    return ((time.perf_counter() - t0) * 1000.0 / repeats), last


def synthetic_salience(frames, seed):
    rng = np.random.RandomState(seed)
    return rng.uniform(low=0.0, high=1.0, size=(frames, 360)).astype(np.float64)


def has_weights(model_capacity):
    return os.path.isfile(
        os.path.join(os.path.dirname(core.__file__),
                     'model-{}.h5'.format(model_capacity)))


def benchmark_salience(salience, warmup, repeats):
    results = []
    for impl in ['legacy', 'fast']:
        if impl == 'legacy':
            try:
                __import__('hmmlearn')
            except ImportError:
                results.append((impl, 'skipped', None))
                continue
        ms, _ = time_call(
            lambda: core.to_viterbi_cents_impl(salience, impl=impl),
            warmup,
            repeats)
        results.append((impl, 'ok', ms))
    return results


def benchmark_predict(audio, sr, model_capacity, warmup, repeats, verbose):
    results = []
    for impl in ['legacy', 'fast']:
        if impl == 'legacy':
            try:
                __import__('hmmlearn')
            except ImportError:
                results.append((impl, 'skipped', None))
                continue
        ms, _ = time_call(
            lambda: crepe.predict(
                audio,
                sr,
                model_capacity=model_capacity,
                viterbi=True,
                viterbi_impl=impl,
                verbose=verbose),
            warmup,
            repeats)
        results.append((impl, 'ok', ms))
    return results


def print_results(title, results):
    print('## {}'.format(title))
    print('| Impl | Status | Mean time |')
    print('|------|--------|-----------|')
    for impl, status, ms in results:
        value = '' if ms is None else '**{:.3f} ms**'.format(ms)
        print('| `{}` | {} | {} |'.format(impl, status, value))
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--frames', type=int, nargs='+', default=[512, 2048])
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--repeats', type=int, default=10)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--include-sweep', action='store_true')
    parser.add_argument('--model-capacity', default='tiny',
                        choices=['tiny', 'small', 'medium', 'large', 'full'])
    parser.add_argument('--verbose', type=int, default=0)
    args = parser.parse_args()

    print('# CREPE Viterbi Benchmark')
    print()
    print('- `frames`: `{}`'.format(args.frames))
    print('- `warmup`: `{}`'.format(args.warmup))
    print('- `repeats`: `{}`'.format(args.repeats))
    print('- `seed`: `{}`'.format(args.seed))
    print('- `include_sweep`: `{}`'.format(args.include_sweep))
    print('- `model_capacity`: `{}`'.format(args.model_capacity))
    print()

    for frames in args.frames:
        salience = synthetic_salience(frames, seed=args.seed + frames)
        print_results('Synthetic decoder core: {} frames'.format(frames),
                      benchmark_salience(salience, args.warmup, args.repeats))

    if not args.include_sweep:
        return

    if not has_weights(args.model_capacity):
        print('> ⚠️ Sweep benchmark skipped: model weight file for `{}` is not '
              'present.'.format(args.model_capacity))
        return

    try:
        __import__('hmmlearn')
    except ImportError:
        print('> ⚠️ Sweep benchmark skipped: `hmmlearn` is not installed.')
        return

    sweep_path = os.path.join(os.path.dirname(__file__), '..', 'tests', 'sweep.wav')
    sr, audio = wavfile.read(sweep_path)
    activation = crepe.get_activation(
        audio,
        sr,
        model_capacity=args.model_capacity,
        verbose=args.verbose)
    print_results('Sweep activation decoder core: {} frames'.format(len(activation)),
                  benchmark_salience(activation, args.warmup, args.repeats))
    print_results('Sweep full predict(): {} frames'.format(len(activation)),
                  benchmark_predict(audio, sr, args.model_capacity,
                                    args.warmup, args.repeats, args.verbose))


if __name__ == '__main__':
    main()

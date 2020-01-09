CREPE Pitch Tracker [![Build Status](https://travis-ci.org/marl/crepe.svg?branch=master)](https://travis-ci.org/marl/crepe)
===================

CREPE is a monophonic pitch tracker based on a deep convolutional neural network operating directly on the time-domain waveform input. CREPE is state-of-the-art (as of 2018), outperfoming popular pitch trackers such as pYIN and SWIPE:

<p align="center"><img src="https://user-images.githubusercontent.com/3009670/36563051-ee6a69a0-17e6-11e8-8d7b-9a37d16ee7ad.png" width="500"></p>

Further details are provided in the following paper:

> [CREPE: A Convolutional Representation for Pitch Estimation](https://arxiv.org/abs/1802.06182)<br>
> Jong Wook Kim, Justin Salamon, Peter Li, Juan Pablo Bello.<br>
> Proceedings of the IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP), 2018.

We kindly request that academic publications making use of CREPE cite the aforementioned paper.


## Installing CREPE

CREPE is hosted on PyPI. To install, run the following command in your Python environment:

```bash
$ pip install --upgrade tensorflow  # if you don't already have tensorflow >= 2.0.0
$ pip install crepe
```

To install the latest version from source clone the repository and from the top-level `crepe` folder call:

```bash
$ python setup.py install
```

## Using CREPE
### Using CREPE from the command line

This package includes a command line utility `crepe` and a pre-trained version of the CREPE model for easy use. To estimate the pitch of `audio_file.wav`, run:

```bash
$ crepe audio_file.wav
```

or

```bash
$ python -m crepe audio_file.wav
```

The resulting `audio_file.f0.csv` contains 3 columns: the first with timestamps (a 10 ms hop size is used by default), the second contains the predicted fundamental frequency in Hz, and the third contains the voicing confidence, i.e. the confidence in the presence of a pitch:

    time,frequency,confidence
    0.00,185.616,0.907112
    0.01,186.764,0.844488
    0.02,188.356,0.798015
    0.03,190.610,0.746729
    0.04,192.952,0.771268
    0.05,195.191,0.859440
    0.06,196.541,0.864447
    0.07,197.809,0.827441
    0.08,199.678,0.775208
    ...

#### Timestamps

CREPE uses 10-millisecond time steps by default, which can be adjusted using 
the `--step-size` option, which takes the size of the time step in millisecond.
For example, `--step-size 50` will calculate pitch for every 50 milliseconds.

Following the convention adopted by popular audio processing libraries such as 
[Essentia](http://essentia.upf.edu/) and [Librosa](https://librosa.github.io/librosa/), 
from v0.0.5 onwards CREPE will pad the input signal such that the first frame 
is zero-centered (the center of the frame corresponds to time 0) and generally 
all frames are centered around their corresponding timestamp, i.e. frame 
`D[:, t]` is centered at `audio[t * hop_length]`. This behavior can be changed 
by specifying the optional `--no-centering` flag, in which case the first frame 
will *start* at time zero and generally frame `D[:, t]` will *begin* at 
`audio[t * hop_length]`. Sticking to the default behavior (centered frames) is 
strongly recommended to avoid misalignment with features and annotations produced 
by other common audio processing tools. 

#### Model Capacity

CREPE uses the model size that was reported in the paper by default, but can optionally
use a smaller model for computation speed, at the cost of slightly lower accuracy.
You can specify `--model-capacity {tiny|small|medium|large|full}` as the command
line option to select a model with desired capacity.
  
#### Temporal smoothing
By default CREPE does not apply temporal smoothing to the pitch curve, but 
Viterbi smoothing is supported via the optional `--viterbi` command line argument. 


#### Saving the activation matrix
The script can also optionally save the output activation matrix of the model 
to an npy file (`--save-activation`), where the matrix dimensions are 
(n_frames, 360) using a hop size of 10 ms (there are 360 pitch bins covering 20 
cents each). 

The script can also output a plot of the activation matrix (`--save-plot`), 
saved to `audio_file.activation.png` including an optional visual representation 
of the model's voicing detection (`--plot-voicing`). Here's an example plot of 
the activation matrix (without the voicing overlay) for an excerpt of male 
singing voice:

![salience](https://user-images.githubusercontent.com/266841/38465913-6fa085b0-3aef-11e8-9633-bdd59618ea23.png)

#### Batch processing
For batch processing of files, you can provide a folder path instead of a file path: 
```bash
$ python crepe.py audio_folder
```
The script will process all WAV files found inside the folder. 

#### Additional usage information
For more information on the usage, please refer to the help message:

```bash
$ python crepe.py --help
```

### Using CREPE inside Python
CREPE can be imported as module to be used directly in Python. Here's a minimal example:
```python
import crepe
from scipy.io import wavfile

sr, audio = wavfile.read('/path/to/audiofile.wav')
time, frequency, confidence, activation = crepe.predict(audio, sr, viterbi=True)
```

## Argmax-local Weighted Averaging

This release of CREPE uses the following weighted averaging formula, which is slightly different from the paper. This only focuses on the neighborhood around the maximum activation, which is shown to further improve the pitch accuracy:

<p align="center"><img src="https://user-images.githubusercontent.com/266841/38990411-68408544-4397-11e8-9e87-ca5a86c5508b.png" width="400"></p>

## Please Note

- The current version only supports WAV files as input.
- The model is trained on 16 kHz audio, so if the input audio has a different sample rate, it will be first resampled to 16 kHz using [resampy](https://github.com/bmcfee/resampy).
- Due to the subtle numerical differences between frameworks, Keras should be configured to use the TensorFlow backend for the best performance. The model was trained using Keras 2.1.5 and TensorFlow 1.6.0, and the newer versions of TensorFlow seems to work as well.
- Prediction is significantly faster if Keras (and the corresponding backend) is configured to run on GPU.
- The provided model is trained using the following datasets, composed of vocal and instrumental audio, and is therefore expected to work best on this type of audio signals.
  - MIR-1K [1]
  - Bach10 [2]
  - RWC-Synth [3]
  - MedleyDB [4]
  - MDB-STEM-Synth [5]
  - NSynth [6]


## References

[1] C.-L. Hsu et al. "On the Improvement of Singing Voice Separation for Monaural Recordings Using the MIR-1K Dataset", *IEEE Transactions on Audio, Speech, and Language Processing.* 2009.

[2] Z. Duan et al. "Multiple Fundamental Frequency Estimation by Modeling Spectral Peaks and Non-Peak Regions", *IEEE Transactions on Audio, Speech, and Language Processing.* 2010.

[3] M. Mauch et al. "pYIN: A fundamental Frequency Estimator Using Probabilistic Threshold Distributions", *Proceedings of the IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP).* 2014.

[4] R. M. Bittner et al. "MedleyDB: A Multitrack Dataset for Annotation-Intensive MIR Research", *Proceedings of the International Society for Music Information Retrieval (ISMIR) Conference.* 2014.

[5] J. Salamon et al.  "An Analysis/Synthesis Framework for Automatic F0 Annotation of Multitrack Datasets",  *Proceedings of the International Society for Music Information Retrieval (ISMIR) Conference*. 2017.

[6] J. Engel et al. "Neural Audio Synthesis of Musical Notes with WaveNet Autoencoders", *arXiv preprint: 1704.01279*. 2017.


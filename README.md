CREPE Pitch Tracker [![Build Status](https://travis-ci.org/marl/crepe.svg?branch=master)](https://travis-ci.org/marl/crepe)
===================

CREPE is a monophonic pitch tracker based on a deep convolutional neural network operating directly on the time-domain waveform input. CREPE is state-of-the-art (as of early 2018), outperfoming popular pitch trackers such as pYIN and SWIPE:

<p align="center"><img src="https://user-images.githubusercontent.com/3009670/36563051-ee6a69a0-17e6-11e8-8d7b-9a37d16ee7ad.png" width="500"></p>

Further details are provided in the following paper:

> [CREPE: A Convolutional Representation for Pitch Estimation](https://arxiv.org/abs/1802.06182)<br>
> Jong Wook Kim, Justin Salamon, Peter Li, Juan Pablo Bello.
> Proceedings of the IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP), 2018.

We kindly request that academic publications making use of CREPE cite the aforementioned paper.


## Using CREPE

CREPE requires a few Python dependencies as specified in [`requirements.txt`](requirements.txt). To install them, run the following command in your Python environment:

```bash
$ pip install -r requirements.txt
```

This repository includes a pre-trained version of the CREPE model for easy use. To estimate the pitch of `audio_file.wav`, run:

```bash
$ python crepe.py audio_file.wav
```

then, the resulting `audio_file.f0.csv` contains the predicted fundamental frequency, along with the confidence on the presence of voicing, is produced for each 10-millisecond frame. 

    # time,frequency,confidence
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

In addition, the script saves the salience representation -- the activations in the last layer in the neural network -- is saved as `audio_file.salience.png`. This is how a salience representation looks like for an excerpt of a male singing voice:

![salience](https://user-images.githubusercontent.com/266841/38465913-6fa085b0-3aef-11e8-9633-bdd59618ea23.png)

For more information on the usage, please refer to the help message:

```bash
$ python crepe.py --help
```


## Please Note

- The current version only supports WAV files as input.
- The model is trained on 16 kHz audio, and if the input audio has a different sample rate, it will be first resampled to 16 kHz using [resampy](https://github.com/bmcfee/resampy).
- While in principle the code should run with any Keras backend, it has only been tested with the TensorFlow backend. The model was trained using Keras 2.1.5 and TensorFlow 1.6.0.
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


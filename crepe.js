!function(window) {
  var audioContext = new AudioContext();
  document.getElementById('srate').innerHTML = audioContext.sampleRate

  function error(message) {
    document.getElementById('status').innerHTML = 'Error: ' + message;
    return message;
  }

  function status(message) {
    document.getElementById('status').innerHTML = message;
  }

  function resample(audioBuffer, onComplete) {
    if (audioBuffer.sampleRate == 48000) {
      var buffer = audioContext.createBuffer(audioBuffer.numberOfChannels, 1024, 16000);
      for (var c = 0; c < audioBuffer.numberOfChannels; c++) {
        var original = audioBuffer.getChannelData(c);
        var destination = buffer.getChannelData(c);

        for (var i = 0; i < 1024; i++) {
          destination[i] = original[i * 3];
        }
      }
      onComplete(buffer);
    } else {
      var channel = audioBuffer.numberOfChannels;
      var samples = audioBuffer.length * 16000 / audioBuffer.sampleRate;

      var offlineContext = new OfflineAudioContext(channel, samples, 16000);
      var bufferSource = offlineContext.createBufferSource();
      bufferSource.buffer = audioBuffer;

      bufferSource.connect(offlineContext.destination);
      bufferSource.start(0);
      offlineContext.startRendering().then(function(renderedBuffer){
        onComplete(renderedBuffer);
      })
    }
  }

  var cent_mapping = tf.add(tf.linspace(0, 7180, 360), tf.tensor(1997.3794084376191))

  function process_microphone_buffer(event) {
    var buffer = event.inputBuffer

    resample(buffer, function(resampled) {
      var frame = resampled.getChannelData(0).slice(0, 1024);
      var salience = model.predict([tf.tensor(frame).reshape([1, 1024])]).reshape([360])

      // cut the frequency of at 1150 Hz, because there aren't really much data trained for these high pitches
      var cutoff = 300
      salience = salience.slice([0], [cutoff]);

      var confidence = salience.max().dataSync()[0];
      var argmax = salience.argMax().dataSync()[0];

      var start = Math.max(0, argmax - 4);
      var end = Math.min(cutoff, argmax + 5);

      salience = salience.slice([start], [end - start]);
      cents = cent_mapping.slice([start], [end - start]);
      productSum = tf.sum(tf.mul(salience, cents)).dataSync()[0]
      weightSum = tf.sum(salience).dataSync()[0]
      predicted_cent = productSum / weightSum;
      predicted_hz = 10 * Math.pow(2, predicted_cent / 1200.0);

      document.getElementById('estimated-pitch').innerHTML = predicted_hz.toFixed(3);
      document.getElementById('voicing-confidence').innerHTML = confidence.toFixed(3);
    })
  }

  function initAudio() {
    if (!navigator.getUserMedia) {
      if (navigator.mediaDevices) {
        navigator.getUserMedia = navigator.mediaDevices.getUserMedia;
      } else {
        navigator.getUserMedia = navigator.webkitGetUserMedia || navigator.mozGetUserMedia || navigator.msGetUserMedia;
      }
    }
    if (navigator.getUserMedia) {
      status('Initializing audio...')
      navigator.getUserMedia({audio: true}, function(stream) {
        status('Setting up AudioContext ...');
        console.log('Audio context sample rate = ' + audioContext.sampleRate);
        var mic = audioContext.createMediaStreamSource(stream);

        var minBufferSize = audioContext.sampleRate / 16000 * 1024;
        for (var bufferSize = 4; bufferSize < minBufferSize; bufferSize *= 2);
        console.log('Buffer size = ' + bufferSize);
        var scriptNode = audioContext.createScriptProcessor(bufferSize, 1, 1);
        scriptNode.onaudioprocess = process_microphone_buffer;

        var gain = audioContext.createGain();
        gain.gain.setTargetAtTime(0, audioContext.currentTime, 0);

        mic.connect(scriptNode);
        scriptNode.connect(gain);
        gain.connect(audioContext.destination);

        status('Running ...')
      }, function(message) {
        error('Could not access microphone - ' + message);
      });
    } else error('getUserMedia not available');
  }

  async function initTF() {
    try {
      status('Loading Keras model...');
      window.model = await tf.loadModel('model/model.json');
      status('Model loading complete');
    } catch (e) {
      error(e);
    }
    initAudio();
  }

  initTF();
}(window);

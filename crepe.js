!function(window) {
  const audioContext = new AudioContext();
  document.getElementById('srate').innerHTML = audioContext.sampleRate

  function error(message) {
    document.getElementById('status').innerHTML = 'Error: ' + message;
    return message;
  }

  function status(message) {
    document.getElementById('status').innerHTML = message;
  }

  function resample(audioBuffer, onComplete) {
    if (audioBuffer.sampleRate % 16000 == 0) {
      const multiplier = audioBuffer.sampleRate / 16000;
      const buffer = audioContext.createBuffer(audioBuffer.numberOfChannels, 1024, 16000);
      for (var c = 0; c < audioBuffer.numberOfChannels; c++) {
        const original = audioBuffer.getChannelData(c);
        const destination = buffer.getChannelData(c);

        for (var i = 0; i < 1024; i++) {
          destination[i] = original[i * multiplier];
        }
      }
      onComplete(buffer);
    } else {
      const channel = audioBuffer.numberOfChannels;
      const samples = audioBuffer.length * 16000 / audioBuffer.sampleRate;

      const offlineContext = new OfflineAudioContext(channel, samples, 16000);
      const bufferSource = offlineContext.createBufferSource();
      bufferSource.buffer = audioBuffer;

      bufferSource.connect(offlineContext.destination);
      bufferSource.start(0);
      offlineContext.startRendering().then(function(renderedBuffer){
        onComplete(renderedBuffer);
      })
    }
  }

  const cent_mapping = tf.add(tf.linspace(0, 7180, 360), tf.tensor(1997.3794084376191))

  function process_microphone_buffer(event) {
    resample(event.inputBuffer, function(resampled) {
      tf.tidy(() => {
        const frame = tf.tensor(resampled.getChannelData(0).slice(0, 1024));
        const mean = tf.mean(frame)
        const zeromean = tf.sub(frame, mean);
        const framenorm = tf.norm(zeromean)
        const framestd = tf.tensor(framenorm.dataSync()/Math.sqrt(1024))
        const normalized = tf.div(zeromean, framestd);
        const input = normalized.reshape([1, 1024])
        const output = model.predict([input])
        const salience = output.reshape([360]);
        const confidence = salience.max().dataSync()[0];
        const argmax = salience.argMax().dataSync()[0];

        const start = Math.max(0, argmax - 4);
        const end = Math.min(360, argmax + 5);

        const weights = salience.slice([start], [end - start]);
        const cents = cent_mapping.slice([start], [end - start]);
        const products = tf.mul(weights, cents)
        const productSum = tf.sum(products).dataSync()[0]
        const weightSum = tf.sum(weights).dataSync()[0]
        const predicted_cent = productSum / weightSum;
        const predicted_hz = 10 * Math.pow(2, predicted_cent / 1200.0);
        const result = (confidence > 0.25) ? predicted_hz.toFixed(3) + ' Hz' : 'no voice';

        document.getElementById('estimated-pitch').innerHTML = result;
        document.getElementById('voicing-confidence').innerHTML = confidence.toFixed(3);
      });
    });
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
        const mic = audioContext.createMediaStreamSource(stream);

        const minBufferSize = audioContext.sampleRate / 16000 * 1024;
        for (var bufferSize = 4; bufferSize < minBufferSize; bufferSize *= 2);
        console.log('Buffer size = ' + bufferSize);
        const scriptNode = audioContext.createScriptProcessor(bufferSize, 1, 1);
        scriptNode.onaudioprocess = process_microphone_buffer;

        const gain = audioContext.createGain();
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

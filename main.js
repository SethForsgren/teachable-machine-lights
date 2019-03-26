// Copyright 2018 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import "@babel/polyfill";
import * as mobilenetModule from '@tensorflow-models/mobilenet';
import * as tf from '@tensorflow/tfjs';
import * as knnClassifier from '@tensorflow-models/knn-classifier';

// Number of classes to classify
const NUM_CLASSES = 2;
// Webcam Image size Must be 227. 
const IMAGE_SIZE = 227;
// K value for KNN
const TOPK = 10;

//
// Lights on constants
const BULB = 2;
let triggered = 0;
let match = 0;
let jog = 0;
//

//look for the sign
function look() {
  console.log('im looking!');
  while (jog = 1) {
    if (triggered == 0) {
      if (match == 1) {
        window.open("https://maker.ifttt.com/trigger/wake/with/key/gcCwXtuJtRSBKdA8vmb-q1npkiZTye7rs3R1OS6TEKZ");
        triggered = 1;
      } 
    }
  }
}

class Main {
  constructor() {
    // Initiate variables
    this.infoTexts = [];
    this.training = -1; // -1 when no class is being trained
    this.videoPlaying = false;

    // Initiate deeplearn.js math and knn classifier objects
    this.bindPage();

    // Create video element that will contain the webcam image
    this.video = document.createElement('video');
    this.video.setAttribute('autoplay', '');
    this.video.setAttribute('playsinline', '');

    // Add video element to DOM
    document.body.appendChild(this.video);

    // Create training buttons and info texts    
    for (let i = 0; i < NUM_CLASSES; i++) {
      const div = document.createElement('div');
      document.body.appendChild(div);
      div.style.marginBottom = '10px';

      // Create training button
      const button = document.createElement('button')
      button.innerText = "Train " + i;
      div.appendChild(button);

      // Listen for mouse events when clicking the button
      button.addEventListener('mousedown', () => this.training = i);
      button.addEventListener('mouseup', () => this.training = -1);

      // Create info text
      const infoText = document.createElement('span')
      infoText.innerText = " No examples added";
      div.appendChild(infoText);
      this.infoTexts.push(infoText);
    }

    // Create space for run button
    const div = document.createElement('div');
    document.body.appendChild(div);
    div.style.marginBottom = '10px';

    // Create run button
    const button = document.createElement('button')
    button.innerText = "Run";
    div.appendChild(button);

    // Listen for mouse events when clicking the button
    button.addEventListener('click', function(e) {
      console.log('button was clicked');

      if (triggered == 0) {
        if (match == 1) {
          window.open("https://maker.ifttt.com/trigger/wake/with/key/gcCwXtuJtRSBKdA8vmb-q1npkiZTye7rs3R1OS6TEKZ");
          triggered = 1;
        } else if (match == 0) {
          console.log('match equals 0')
        }
      } else {
        console.log('triggered not equal to 0');
      }
    });

    // Create space for reset button
    const divR = document.createElement('div');
    document.body.appendChild(divR);
    divR.style.marginBottom = '10px';

    // Create reset button
    const buttonR = document.createElement('button')
    buttonR.innerText = "Reset trigger";
    divR.appendChild(buttonR);

    // Listen for mouse events when clicking the button
    buttonR.addEventListener('click', function(e) {
      console.log('reset button was clicked');
      triggered = 0;
    });

    // Create space for start button
    const divS = document.createElement('div');
    document.body.appendChild(divS);
    divS.style.marginBottom = '10px';

    // Create start button
    const buttonS = document.createElement('button')
    buttonS.innerText = "Start";
    divS.appendChild(buttonS);

    // Listen for mouse events when clicking the start button
    buttonS.addEventListener('click', function(e) {
      console.log('start button was clicked');
      jog = 1;
      look();
    });

    // Setup webcam
    navigator.mediaDevices.getUserMedia({ video: true, audio: false })
      .then((stream) => {
        this.video.srcObject = stream;
        this.video.width = IMAGE_SIZE;
        this.video.height = IMAGE_SIZE;

        this.video.addEventListener('playing', () => this.videoPlaying = true);
        this.video.addEventListener('paused', () => this.videoPlaying = false);
      })
  }

  async bindPage() {
    this.knn = knnClassifier.create();
    this.mobilenet = await mobilenetModule.load();

    this.start();
  }

  start() {
    if (this.timer) {
      this.stop();
    }
    this.video.play();
    this.timer = requestAnimationFrame(this.animate.bind(this));
  }

  stop() {
    this.video.pause();
    cancelAnimationFrame(this.timer);
  }

  async animate() {
    if (this.videoPlaying) {
      // Get image data from video element
      const image = tf.fromPixels(this.video);

      let logits;
      // 'conv_preds' is the logits activation of MobileNet.
      const infer = () => this.mobilenet.infer(image, 'conv_preds');

      // Train class if one of the buttons is held down
      if (this.training != -1) {
        logits = infer();

        // Add current image to classifier
        this.knn.addExample(logits, this.training)
      }

      const numClasses = this.knn.getNumClasses();
      if (numClasses > 0) {

        // If classes have been added run predict
        logits = infer();
        const res = await this.knn.predictClass(logits, TOPK);

        for (let i = 0; i < NUM_CLASSES; i++) {

          // The number of examples for each class
          const exampleCount = this.knn.getClassExampleCount();

          // Make the predicted class bold
          if (res.classIndex == i) {
            this.infoTexts[i].style.fontWeight = 'bold';
            // this.infoTexts[i].style.fontSize = "20px";
          } else {
            this.infoTexts[i].style.fontWeight = 'normal';
            // this.infoTexts[i].style.fontSize = "40px";
          }

          // make res.class change match status
          if (res.classIndex == 1) {
            match = 1;
          } else {
            match = 0;
          }

          // Update info text
          if (exampleCount[i] > 0) {
            this.infoTexts[i].innerText = ` ${exampleCount[i]} examples - ${res.confidences[i] * 100}%`
          }
        }
      }

      // Dispose image when done
      image.dispose();
      if (logits != null) {
        logits.dispose();
      }
    }
    this.timer = requestAnimationFrame(this.animate.bind(this));
  }
}

window.addEventListener('load', () => new Main());
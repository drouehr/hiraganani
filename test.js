const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');
const path = require('path');
const PNG = require('pngjs').PNG;
const characterMap = ['A','I','U','E','O','KA','KI','KU','KE','KO','SA','SHI','SU','SE','SO','TA','CHI','TSU','TE','TO','NA','NI','NU','NE','NO','HA','HI','FU','HE','HO','MA','MI','MU','ME','MO','YA','YU','YO','RA','RI','RU','RE','RO','WA','WO','N'];

let correctlyLabelled = 0;

async function loadAndPreprocessImage(imagePath) {  
  let buffer = fs.readFileSync(imagePath);
  const pixels = PNG.sync.read(buffer).data;
  
  const grayPixels = [];
  for (let i = 0; i < pixels.length; i += 4) {
    const gray = 0.299*pixels[i] + 0.587*pixels[i+1] + 0.114*pixels[i+2];
    grayPixels.push(gray / 255.0);
  }
  
  const input = tf.tensor(grayPixels, [1, 127, 128, 1]);
  return input;
}

async function main() {
  let modelDir = path.join(__dirname, 'models');
  let modelFiles = fs.readdirSync(modelDir).filter(file => fs.statSync(path.join(modelDir, file)).isDirectory()).sort();
  const mostRecentModel = path.join(modelDir, modelFiles.at(-1));
  console.log(`loading model '${modelFiles.at(-1)}'.`);
  const model = await tf.node.loadSavedModel(mostRecentModel);
  
  const testImagesDir = path.join(__dirname, 'test_images');
  const testImages = fs.readdirSync(testImagesDir).sort((a, b) => parseInt(a.split('_')[0]) - parseInt(b.split('_')[0]));

  for (const imageName of testImages) {
    const imagePath = path.join(testImagesDir, imageName);
    const input = await loadAndPreprocessImage(imagePath);
    const prediction = model.predict(input);
    const predictedClass = tf.argMax(prediction, 1).dataSync()[0];
    const predictedProbability = prediction.dataSync()[predictedClass];
    if(predictedClass === parseInt(imageName.split('_')[0])) correctlyLabelled++;
    console.log(`image '${imageName}' (${characterMap[parseInt(imageName.split('_')[0])]}): predicted class = ${predictedClass} (${characterMap[predictedClass]}), prob = ${predictedProbability}`);
  }
  console.log(`accuracy on test images: ${correctlyLabelled}/${testImages.length}, ${correctlyLabelled/testImages.length*100}%`);
}

main();
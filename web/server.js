const bodyParser = require('body-parser');
const express = require('express');
const fs = require('fs');
const path = require('path');
const PNG = require('pngjs').PNG;
const tf = require('@tensorflow/tfjs-node');

const app = express();
const charMap = [["a", "あ"], ["i", "い"], ["u", "う"], ["e", "え"], ["o", "お"], ["ka", "か"], ["ki", "き"], ["ku", "く"], ["ke", "け"], ["ko", "こ"], ["sa", "さ"], ["shi", "し"], ["su", "す"], ["se", "せ"], ["so", "そ"], ["ta", "た"], ["chi", "ち"], ["tsu", "つ"], ["te", "て"], ["to", "と"], ["na", "な"], ["ni", "に"], ["nu", "ぬ"], ["ne", "ね"], ["no", "の"], ["ha", "は"], ["hi", "ひ"], ["fu", "ふ"], ["he", "へ"], ["ho", "ほ"], ["ma", "ま"], ["mi", "み"], ["mu", "む"], ["me", "め"], ["mo", "も"], ["ya", "や"], ["yu", "ゆ"], ["yo", "よ"], ["ra", "ら"], ["ri", "り"], ["ru", "る"], ["re", "れ"], ["ro", "ろ"], ["wa", "わ"], ["wo", "を"], ["n", "ん"]];

let model;
async function loadModel(){
  const modelDir = path.join(__dirname, 'models');
  const modelFiles = fs.readdirSync(modelDir).filter(file => fs.statSync(path.join(modelDir, file)).isDirectory()).sort();
  const mostRecentModel = path.join(modelDir, modelFiles.at(-1));
  console.log(`<debug> loading model '${modelFiles.at(-1)}'.`);
  model = await tf.node.loadSavedModel(mostRecentModel);
}
loadModel();

const port = 80;
app.use(bodyParser.json()); 
app.use(express.static('public'));
app.set('views', __dirname + 'views');
app.set('view engine', 'ejs');

// route handling, etc

app.post('/submit_hiragana', (req, res) => {
  const data = req.body;
  console.log(`received submission for hiragana '${charMap[data.charIndex][0]}'.`);
  const imageDir = './collected_samples';
  const currentCharImages = fs.readdirSync(imageDir).filter(file => file.split('_')[0] === (data.charIndex).toString());
  const imageFileName = `${data.charIndex}_${currentCharImages.length}.png`;
  fs.writeFileSync(`${imageDir}/${imageFileName}`, Buffer.from(data.image.split(",")[1], 'base64'));
  res.send(`saved ${charMap[data.charIndex][1]} #${currentCharImages.length}`);
  console.log(`saved image '${imageFileName}'.`);
});

app.post('/predict_hiragana', (req, res) => {
  const data = req.body;
  const pixels = PNG.sync.read(Buffer.from(data.image.split(",")[1], 'base64')).data;
  let grayPixels = [];
  for (let i = 0; i < pixels.length; i += 4) {
    const gray = 0.299*pixels[i] + 0.587*pixels[i+1] + 0.114*pixels[i+2];
    grayPixels.push(gray / 255.0);
  }
  let tensor = tf.tensor(grayPixels, [1, 127, 128, 1]);
  const prediction = model.predict(tensor);
  const predictedClass = tf.argMax(prediction, 1).dataSync()[0];
  const predictedProbability = prediction.dataSync()[predictedClass];
  res.send(`${predictedClass},${predictedProbability}`);
});

app.listen(port, () => {
  console.log(`listening on port ${port}`);
});
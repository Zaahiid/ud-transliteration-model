const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');

// Load your data
const rawData = JSON.parse(fs.readFileSync('your_data.json', 'utf8'));

// Preprocess data
function preprocessData(data) {
  const urduTexts = data.map(item => item.urdu);
  const latinTexts = data.map(item => item.latin);

  // Create vocabularies
  const urduVocab = new Set(urduTexts.join('').split(''));
  const latinVocab = new Set(latinTexts.join('').split(''));

  const urduToIndex = Object.fromEntries([...urduVocab].map((char, i) => [char, i + 1]));
  const latinToIndex = Object.fromEntries([...latinVocab].map((char, i) => [char, i + 1]));

  // Convert texts to sequences
  const urduSequences = urduTexts.map(text => text.split('').map(char => urduToIndex[char]));
  const latinSequences = latinTexts.map(text => text.split('').map(char => latinToIndex[char]));

  // Pad sequences
  const maxLen = Math.max(...urduSequences.map(seq => seq.length));
  const paddedUrdu = urduSequences.map(seq => tf.util.arraysEqual(seq, maxLen));
  const paddedLatin = latinSequences.map(seq => tf.util.arraysEqual(seq, maxLen));

  return {
    inputData: tf.tensor2d(paddedUrdu),
    outputData: tf.tensor2d(paddedLatin),
    urduVocabSize: urduVocab.size + 1,  // +1 for padding
    latinVocabSize: latinVocab.size + 1,
    urduToIndex,
    latinToIndex
  };
}

const { inputData, outputData, urduVocabSize, latinVocabSize, urduToIndex, latinToIndex } = preprocessData(rawData);

// Define model
const model = tf.sequential();
model.add(tf.layers.embedding({ inputDim: urduVocabSize, outputDim: 64, inputLength: inputData.shape[1] }));
model.add(tf.layers.bidirectional({ layer: tf.layers.lstm({ units: 64, returnSequences: true }) }));
model.add(tf.layers.bidirectional({ layer: tf.layers.lstm({ units: 64, returnSequences: true }) }));
model.add(tf.layers.timeDistributed({ layer: tf.layers.dense({ units: latinVocabSize, activation: 'softmax' }) }));

// Compile model
model.compile({
  optimizer: 'adam',
  loss: 'sparseCategoricalCrossentropy',
  metrics: ['accuracy']
});

// Train model
async function trainModel() {
  const history = await model.fit(inputData, outputData, {
    epochs: 50,
    batchSize: 32,
    validationSplit: 0.2
  });

  console.log(history.history);

  // Save model
  await model.save('file://./model');

  // Save tokenizers
  fs.writeFileSync('urdu_tokenizer.json', JSON.stringify(urduToIndex));
  fs.writeFileSync('latin_tokenizer.json', JSON.stringify(latinToIndex));
}

trainModel();
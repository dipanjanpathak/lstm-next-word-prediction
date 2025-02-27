# FAQ-Based Text Generation with LSTM

## Overview
This project implements a **text generation model** using **LSTM (Long Short-Term Memory)** networks trained on FAQ data. The model is built using **TensorFlow/Keras** and learns to predict the next word in a sequence based on a given input text.

## Features
- Tokenizes and processes FAQ-based text data
- Creates sequences for training using **n-gram modeling**
- Uses **LSTM layers** for sequence learning
- Generates text predictions word by word

## Installation
Ensure you have Python 3.x installed and install the required dependencies:

```bash
pip install tensorflow numpy
```

## Model Architecture
The LSTM model consists of the following layers:
1. **Embedding Layer**: Converts words into dense vectors
2. **LSTM Layers**: Captures sequential patterns in text
3. **Dense Layer**: Outputs a probability distribution over vocabulary

## Training
The model is trained on **FAQ-based sequences** using **categorical cross-entropy loss** and the **Adam optimizer**.

```python
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, epochs=100)
```

### Sample Training Progress:
```
Epoch 1/100 - Loss: 5.75 - Accuracy: 2.98%
Epoch 10/100 - Loss: 3.31 - Accuracy: 32.92%
Epoch 100/100 - Loss: 0.28 - Accuracy: 91.51%
```

## Text Generation
Once trained, the model can generate text predictions:
```python
text = "Deep Learning"
for i in range(10):
    token_text = tokenizer.texts_to_sequences([text])[0]
    padded_token_text = pad_sequences([token_text], maxlen=56, padding='pre')
    pos = np.argmax(model.predict(padded_token_text))
    for word, index in tokenizer.word_index.items():
        if index == pos:
            text = text + " " + word
            print(text)
```
### Example Output:
```
Deep Learning models
Deep Learning models are
Deep Learning models are powerful
...
```

## Future Improvements
- Improve dataset quality by adding more FAQs
- Experiment with **Bidirectional LSTMs** for better performance
- Implement **beam search** for more advanced text generation

## License
This project is open-source and available under the MIT License.




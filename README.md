# Morrolingo
Welcome to the English to Darija Translator project! This web application utilizes a sequence-to-sequence model and NLP techniques to translate English text into Darija, Moroccan dielect. Below, you'll find an overview of the project, its features, and instructions to get started.

## Project Structure

- `model.py`: Source code for the sequence-to-sequence model used in translation.
- `templates/`: Contains the HTML templates for the web interface.
  - `index.html`: Landing page of the web application.
  - `interface.html`: Translation interface in action.
- `front.py`: Flask framework code that connects the frontend and backend.
- `static/`: Contains static assets like CSS files and images.
  - `css/`: CSS files for styling the web pages.
  - `images/`: Images used in the application.
- `test.txt`: The dataset used for training and testing.

  ## Steps to build :
  
  by following this tutorial " https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html#evaluation "
  as a teamwork we were able to build this amazing friendly web app :
  1.Prepare Resources:
  -Install Python and required libraries like TensorFlow or PyTorch for the model.
  -Collect a dataset with English-Darija sentence pairs.

  2.Preprocess Data:
  -Clean and format the dataset.
  -Tokenize sentences into words or subword units.

  3.Build Model:
  -Create an encoder-decoder architecture.
  -Use embeddings to convert words into vector representations.

  4.Train Model:
  -Split data into training and validation sets.
  -Train the model using training data and monitor performance using validation data.

  5.Develop Web App:
  -Set up a Flask web app framework.
  -Design a user interface for input and display of translations (html ,css and js)



  ## Getting Started
  1. Clone the repository: "git clone https://github.com/LahnoukiAicha/Morrolingo.git"
  2. cd english-to-darija-translator
  3. pip install -r requirements.txt
  4. Run the Flask app: python front.py
  5. Access the translator in your web browser by visiting http://localhost:5000.


## Limitations
1.The accuracy of translations is constrained by the training data used.
2.Complex sentence structures and idiomatic expressions might not be translated accurately.
3.The model's performance is optimized for common phrases but might struggle with specialized vocabulary.




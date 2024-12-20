# Project-Spam-Email-Classification-using-NLP-and-Machine-Learning-
<h1>Spam Email Classification Application</h1>

<p> This project is a <b>Machine Learning-based web application</b> that classifies emails as either spam or ham (not spam). The app is built using <b>Streamlit</b> for the frontend interface and a trained <b>Naive Bayes classifier</b> for the backend prediction model.
</p>


## Features

- **Dynamic User Input**: Allows users to input email content for classification.
- **Real-Time Prediction**: Classifies emails as either spam or ham with instant results.
- **Interactive Interface**: User-friendly interface built with Streamlit.
- **Model Persistence**: Pre-trained model and vectorizer are saved and loaded using `pickle` for efficiency.

---

## Technologies Used

- **Programming Language**: Python
- **Frontend Framework**: Streamlit
- **Machine Learning Model**: Naive Bayes Classifier
- **Libraries**:
  - `pandas`: For data manipulation
  - `numpy`: For numerical operations
  - `scikit-learn`: For training and evaluating the model
  - `nltk`: For text preprocessing
  - `pickle`: For model and vectorizer serialization

---

## Dataset

The dataset used for this project is `spam.csv`, which contains:
- Email content
- Labels:
  - `ham`: Not spam
  - `spam`: Spam

### Dataset Preparation
1. Dropped unnecessary columns.
2. Renamed columns for clarity.
3. Mapped labels to numerical values (`ham = 0`, `spam = 1`).
4. Preprocessed email text by:
   - Removing special characters and numbers.
   - Converting text to lowercase.
   - Removing stopwords.
   - Applying stemming to reduce words to their root form.

---

## How the Application Works

### 1. **Preprocessing**
The email content is preprocessed to remove noise and prepare it for vectorization:
- Special characters and numbers are removed.
- Stopwords (e.g., "the," "is") are filtered out.
- Words are reduced to their root forms using stemming.

### 2. **Vectorization**
The text is transformed into a numerical format using **CountVectorizer** with a maximum of 5000 features.

### 3. **Model Training**
A **Multinomial Naive Bayes** model is trained on the vectorized data, split into training and testing sets with an 80-20 ratio.

### 4. **Prediction**
The trained model predicts whether the user-inputted email content is spam or ham based on the input text.

---

## How to Run the Application

### 1. Install Dependencies
Create a virtual environment and install the required Python packages:
```bash
pip install pandas
pip install scikit-learn
pip install numpy
pip install nltk
pip install streamlit

OR Try the below:
pip install -r requirements.txt

```

### 2. Run the Streamlit Application
```bash
streamlit run .\spamDetector.py
```

### 4. Access the App
Open the URL displayed in your terminal (default: `http://localhost:8501`) to access the application.

---

## Usage

1. Enter the email content you want to classify in the text area.
2. Click on the **Classify** button.
3. View the result:
   - **"This is Not A Spam Email"**: The email is classified as ham.
   - **"This is A Spam Email"**: The email is classified as spam.

---

## Files Included

- `spam.csv`: Dataset file used for training and testing.
- `app.py`: Python script containing the Streamlit app code.
- `spam123.pkl`: Serialized Naive Bayes model.
- `vec123.pkl`: Serialized CountVectorizer object.
- `requirements.txt`: List of dependencies required to run the application.

---

## Future Improvements

- Add support for additional datasets.
- Implement advanced preprocessing techniques like lemmatization.
- Add more machine learning models for comparison.
- Deploy the application using cloud services (e.g., AWS, Heroku).

---

## Acknowledgments

- **Dataset**: Publicly available email spam classification dataset.
- **Libraries**: Open-source libraries including scikit-learn, nltk, and Streamlit.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.


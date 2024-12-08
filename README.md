# Farmers01_IITB


# Crop Recommendation System

This project is a **Crop Recommendation System** that uses a trained **RandomForest model** to recommend the most suitable crop based on soil characteristics and environmental factors. The system uses Flask to create a web interface, allowing users to input parameters like nitrogen, phosphorus, potassium content, temperature, humidity, pH, and rainfall to get a crop recommendation.

---

## Features

- **User-friendly Web Interface**: Built with HTML and Flask to allow users to input soil and environmental parameters.
- **Model-Based Prediction**: Uses a trained **RandomForest model** to predict the best crop based on the input features.
- **Interactive**: Allows multiple predictions without restarting the server.

---

## Prerequisites

Before you begin, ensure you have met the following requirements:

- Python 3.x or later
- Installed dependencies using `pip`

---

## Installation

1. **Clone or Download the Repository**

   Download or clone this repository to your local machine.

   ```bash
   git clone <repository_url>
   ```

2. **Create a Virtual Environment (Optional but Recommended)**

   It’s a good practice to create a virtual environment to avoid conflicts with other Python packages.

   ```bash
   python -m venv venv
   ```

   To activate the virtual environment:

   - **Windows**:
     ```bash
     venv\Scripts\activate
     ```

   - **Mac/Linux**:
     ```bash
     source venv/bin/activate
     ```

3. **Install Dependencies**

   Install the required libraries using `pip`:

   ```bash
   pip install -r requirements.txt
   ```

   If you don’t have a `requirements.txt` file, you can manually install the necessary dependencies:

   ```bash
   pip install flask scikit-learn joblib pandas
   ```

---

## Directory Structure

Your project directory should look like this:

```
project/
├── app.py
├── optimized_crop_recommendation_model.pkl
├── scaler.pkl
├── label_encoder.pkl
├── crop.csv
├── templates/
│   └── index.html
└── static/
```

- **`app.py`**: Flask application script that handles user input and prediction logic.
- **`optimized_crop_recommendation_model.pkl`**: The trained model that predicts the recommended crop.
- **`scaler.pkl`**: Scaler used for feature scaling before making predictions.
- **`label_encoder.pkl`**: Label encoder that converts the crop names back to human-readable labels.
- **`crop.csv`**: Dataset (optional) used for training the model.
- **`templates/index.html`**: HTML file that contains the user interface.
- **`static/`**: (Optional) Folder for static assets like images, styles, or scripts.

---

## How to Run

1. **Run the Flask Server**

   Navigate to your project directory and run the `app.py` file:

   ```bash
   python app.py
   ```

   By default, the Flask server will start at `http://127.0.0.1:5000`.

2. **Access the Web Interface**

   Open a web browser and go to `http://127.0.0.1:5000`. You will see the user interface where you can enter the soil and environmental parameters.

3. **Enter Inputs**

   Enter values for the following fields:
   - Nitrogen (N)
   - Phosphorus (P)
   - Potassium (K)
   - Temperature (°C)
   - Humidity (%)
   - pH level
   - Rainfall (mm)

4. **Get Recommended Crop**

   After entering the values, click on "Recommend Crop" to see the recommended crop based on your inputs.

---

## Model Description

This system uses a **RandomForestClassifier** from `scikit-learn` to predict the best crop based on soil and environmental characteristics. The model was trained using a dataset with features such as nitrogen content, phosphorus content, potassium content, temperature, humidity, pH level, and rainfall. 

Hyperparameter tuning was performed using **GridSearchCV** to improve the model's performance.

---

## Saving and Loading the Model

The trained model, along with the scaler and label encoder, are saved using **joblib**. These files can be loaded later to make predictions without retraining the model.

- `optimized_crop_recommendation_model.pkl`: The trained RandomForest model.
- `scaler.pkl`: The scaler used to standardize the input features.
- `label_encoder.pkl`: The label encoder used to map crop names to numeric values.

---

## Future Improvements

- **Cross-validation**: Use cross-validation to evaluate the model performance on different splits of the data.
- **Model Optimization**: Experiment with other machine learning models and tuning techniques to further improve accuracy.
- **Deployment**: Host the web application on a cloud platform like Heroku or AWS for public access.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---



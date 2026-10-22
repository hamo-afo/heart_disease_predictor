# Heart Disease Predictor

A small machine-learning web application that predicts whether heart disease is
likely from four exercise and diagnostic features. The interface is built with
[Streamlit](https://streamlit.io/) and uses a trained Support Vector Classifier
created in `Heart_Disease_Prediction.ipynb`.

> **Educational use only:** This project is a learning exercise and is not
> medical advice, a diagnosis, or a substitute for evaluation by a qualified
> healthcare professional.

## Features

- Interactive Streamlit interface
- Predictions from a serialized SVC model
- Reproducible categorical encoding for:
  - Chest pain type
  - Exercise-induced angina
  - ST slope
  - Thalassemia test result
- Optional display of the encoded feature vector sent to the model

## Project structure

| File | Description |
| --- | --- |
| `app.py` | Streamlit application |
| `Heart_Disease_Prediction.ipynb` | Notebook used for data preparation and model training |
| `svc_trained_model.pkl` | Trained Support Vector Classifier |
| `heart-disease-cleveland-full.csv` | Cleveland heart-disease dataset |
| `heart-sample-data.csv` | Sample input data |
| `requirements.txt` | Python dependencies |
| `runtime.txt` | Recommended Python runtime version |
| `training-data-encoded.csv` | Encoded training data |
| `testing-data-encoded.csv` | Encoded testing data |
| `model-predictions.csv` | Saved model predictions |

## Requirements

- Python 3.11
- pip

Install the dependencies from the project directory:

```bash
python -m pip install -r requirements.txt
```

## Run the application

Start Streamlit with:

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal, usually:

```text
http://localhost:8501
```

## How to use it

1. Select values for the four available clinical/exercise features.
2. Click **Predict**.
3. Review the prediction and, if needed, expand the encoded feature vector.

The app loads `svc_trained_model.pkl` from the same directory as `app.py`, so
the model file must remain in the project directory.

## Model and data notes

The model was trained using the UCI Cleveland heart-disease data prepared in
the included notebook. The application rebuilds the categorical encoders from
fixed category lists so that the values entered in the UI use the same numeric
mapping as the training workflow. The notebook reports test accuracy of about
79% on the prepared dataset.

Model performance can vary with data preparation, random splits, and the
selected features. The prediction should not be interpreted as a clinical
assessment.

## License

No separate license is currently provided. See the original dataset and source
materials for their respective terms of use.

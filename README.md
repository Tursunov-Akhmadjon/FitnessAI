# Health Journey Predictor App

A Studio Ghibli-inspired Streamlit application for predicting sleep hours and daily calorie intake based on user health and workout data.

## Features

- **Beautiful Ghibli-Inspired UI**: Whimsical design with soft colors and nature-inspired elements
- **User Input Form**: Collects comprehensive health and workout data
- **Dual Predictions**: Estimates optimal sleep duration and daily calorie needs
- **Responsive Design**: Works on various screen sizes

## How It Works

1. The app collects user data through an intuitive form
2. The first model predicts optimal sleep hours based on user inputs
3. The second model uses all inputs plus the predicted sleep hours to estimate daily calorie needs
4. Results are displayed in visually appealing cards

## Setup Instructions

### Prerequisites
- Python 3.8+
- pip package manager

### Installation

1. Clone this repository or download the files
2. Install the required packages:

```bash
pip install -r requirements.txt
```

3. Run the Streamlit app:

```bash
streamlit run app.py
```

## Deploying Your Models

To use your actual trained models, replace the dummy models in the `load_models()` function with your trained models:

```python
def load_models():
    # Load your actual trained models
    with open('sleep_model.pkl', 'rb') as file:
        sleep_model = pickle.load(file)
    with open('calorie_model.pkl', 'rb') as file:
        calorie_model = pickle.load(file)
    
    return sleep_model, calorie_model
```

Make sure your models expect input features in the same order as they're provided in the app.

## Customization

- Modify the CSS in the `add_ghibli_style()` function to change colors and styling
- Add additional input fields as needed for your specific models
- Enhance the visualization section with charts or graphs

## Deployment Options

- **Streamlit Cloud**: Deploy directly from GitHub
- **Heroku**: Use a Procfile with `web: streamlit run app.py`
- **Docker**: Containerize the application for consistent deployment

## License

This project is available for personal and commercial use.

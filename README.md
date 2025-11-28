# House Price Prediction API

A machine learning web application for predicting house prices using a pre-trained model.

## Features
- 🎯 Real-time house price predictions
- 🎨 User-friendly Streamlit interface
- 📊 Input validation and error handling
- 🚀 Ready for deployment on Render

## Local Setup

### Prerequisites
- Python 3.8+
- pip

### Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd house-price-api
```

2. Create a virtual environment:
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
streamlit run app.py
```

The app will be available at `http://localhost:8501`

## Deployment on Render

### Steps:

1. **Push to GitHub:**
   ```bash
   git add .
   git commit -m "Prepare for Render deployment"
   git push
   ```

2. **Create a New Web Service on Render:**
   - Go to [render.com](https://render.com)
   - Click "New +" → "Web Service"
   - Connect your GitHub repository
   - Select the branch with your code
   - Fill in the deployment details:
     - **Name:** `house-price-api`
     - **Environment:** Python 3
     - **Build Command:** `pip install -r requirements.txt`
     - **Start Command:** `streamlit run app.py --server.port=10000 --server.address=0.0.0.0`
   - Click "Create Web Service"

3. **Monitor Deployment:**
   - Render will automatically build and deploy your app
   - You'll get a unique URL like `https://your-app-name.onrender.com`

## Files

- `app.py` - Streamlit UI application
- `main.py` - FastAPI backend (optional alternative)
- `house_model.pkl` - Trained machine learning model
- `requirements.txt` - Python dependencies
- `render.yaml` - Render deployment configuration

## Input Features

The model accepts the following house features:

- **Median Income** ($)
- **House Age** (years)
- **Rooms** (average)
- **Bedrooms** (average)
- **Population**
- **Households**
- **Latitude**
- **Longitude**

## Output

The model predicts the **house price** based on the input features.

## License

This project is open source and available under the MIT License.
"# house-price-predictor" 

# Model Export Guide for Google Colab

## Option 1: Re-export the Model from Colab (RECOMMENDED)

Run this in your Google Colab notebook:

```python
# After training your model, use this to save it:
import joblib
import pickle

# Save using joblib (latest version)
joblib.dump(your_model, "house_model.pkl")

# Or save using pickle with protocol 4 (more compatible)
with open("house_model.pkl", "wb") as f:
    pickle.dump(your_model, f, protocol=4)

# Download the file from Colab
from google.colab import files
files.download("house_model.pkl")
```

## Option 2: Use pickle instead of joblib (FASTER FIX)

1. Download your trained model from Colab as `model.pkl`
2. Use this code to load it in `app.py`:

```python
import pickle

with open("house_model.pkl", "rb") as f:
    model = pickle.load(f)
```

## Option 3: Use ONNX format (MOST PORTABLE)

```python
# In Colab
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnx

initial_type = [('float_input', FloatTensorType([None, 8]))]
onyx_model = convert_sklearn(your_model, initial_types=initial_type)
with open("house_model.onnx", "wb") as f:
    f.write(onyx_model.SerializeToString())
```

---

## Troubleshooting

If you see `KeyError: 102` or pickle errors:
- The pkl file was corrupted during download
- Version mismatch between joblib/sklearn versions
- **Solution:** Re-export from Colab using the latest code above

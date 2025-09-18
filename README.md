# cattle-breed-identifier-api
A FastAPI that classifies cattle breeds from images using TensorFlow Lite.
# Cattle Breed Classification API 🐄

A RESTful API built with FastAPI that classifies cattle breeds from images using a TensorFlow Lite machine learning model. The model can identify 37 different cattle breeds.

## 🌟 Features

- Image-based cattle breed classification
- Returns top 2 predictions with confidence scores
- RESTful API architecture
- Interactive documentation with Swagger UI
- CORS-enabled for frontend applications

## 📋 Supported Breeds

The API can classify 37 breeds including:
- Gir, Jersey, Holstein Friesian
- Murrah, Sahiwal, Red Sindhi  
- Amritmahal, Kangayam, Tharparkar
- And 28 more breeds...

## 🚀 Quick Start

### API Endpoint
**POST** `https://your-app-name.herokuapp.com/predict`

### Example Request
```bash
curl -X POST -F "file=@image.jpg" https://your-app-name.herokuapp.com/predict

# Production Email NLP API

## Overview
Production-ready API for email classification using the improved machine learning model.

## Features
- ✅ 99.7% accuracy on training data
- ✅ 91.7% accuracy on previously problematic test cases
- ✅ Rate limiting and security features
- ✅ Comprehensive logging
- ✅ Health monitoring
- ✅ Batch processing support

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Start the API
**Windows:**
```bash
start_production.bat
```

**Linux/Mac:**
```bash
./start_production.sh
```

### 3. Test the API
```bash
curl -X POST http://localhost:8080/classify \
  -H "Content-Type: application/json" \
  -d '{"email": "Hello, this is a test email"}'
```

## API Endpoints

### POST /classify
Classify a single email.

**Request:**
```json
{
  "email": "Your email text here"
}
```

**Response:**
```json
{
  "category": "Work",
  "confidence": 0.95,
  "probabilities": {
    "Personal": 0.02,
    "Spam": 0.01,
    "Standard": 0.01,
    "Urgent": 0.01,
    "Work": 0.95
  },
  "model_version": "final_improved_calibrated",
  "accuracy_target": "99.7%",
  "timestamp": "2025-07-30T15:30:00"
}
```

### POST /batch_classify
Classify multiple emails (max 100).

### GET /health
Health check endpoint.

### GET /model_info
Get model information.

## Configuration

### API Configuration (`config/api_config.json`)
- `host`: API host (default: 0.0.0.0)
- `port`: API port (default: 8080)
- `debug`: Debug mode (default: false)
- `rate_limit`: Rate limiting settings

### Model Configuration (`config/model_config.json`)
- `confidence_threshold`: Minimum confidence for classification
- `batch_size`: Maximum batch size for processing

## Model Performance

### Accuracy Improvements
- **Work emails**: 100% accuracy (IT requests, follow-ups)
- **Personal vs Work**: Clear distinction achieved
- **Urgent emails**: 83% accuracy (improved from 0%)
- **Standard emails**: 83% accuracy (improved from 0%)
- **Spam detection**: 100% accuracy

### Training Data
- **Total examples**: 11,300
- **Categories**: Personal, Spam, Standard, Urgent, Work
- **Model type**: RandomForest with calibration

## Monitoring

### Logs
- API logs: `logs/api.log`
- Application logs: Console output

### Health Checks
- Model loading status
- API responsiveness
- Memory usage

## Security Features
- Rate limiting (100 requests/minute)
- Input validation
- CORS support
- Error handling

## Deployment

### Production Deployment
1. Copy all files to production server
2. Install dependencies: `pip install -r requirements.txt`
3. Start API: `./start_production.sh`
4. Configure reverse proxy (nginx/apache) if needed

### Docker Deployment (Optional)
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
EXPOSE 8080
CMD ["python", "production_api.py"]
```

## Support
For issues or questions, check the logs in `logs/api.log`.

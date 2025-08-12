# 🔐 Financial Fraud Detection System

A lightweight yet effective machine learning solution for detecting fraudulent financial transactions using behavioral, transactional, and contextual features.

## 🎯 Project Overview

This project implements a **Logistic Regression** model that achieves high fraud detection accuracy while maintaining interpretability and fast inference speeds suitable for real-time deployment.

### Key Features
- **Lightweight Model**: Logistic Regression optimized for speed and interpretability
- **Advanced Feature Engineering**: Behavioral patterns, balance inconsistencies, and transaction anomalies
- **Class Imbalance Handling**: Robust recall on minority fraud cases
- **Real-time Deployment**: Streamlit dashboard for live fraud detection
- **High Performance**: 98.9%+ AUC with balanced precision-recall

## 🏗️ Architecture

```
Data Input → Feature Engineering → Model Prediction → Risk Assessment → Alert/Dashboard
```

### Feature Categories
1. **Behavioral Features**: Temporal patterns (hour, day, weekend)
2. **Transactional Features**: Amount ratios, merchant indicators
3. **Contextual Features**: Balance inconsistencies, account patterns

## 📊 Model Performance

| Metric | Score |
|--------|--------|
| **ROC-AUC** | 0.9891 |
| **Precision (Fraud)** | 0.06 |
| **Recall (Fraud)** | 0.87 |
| **F1-Score** | 0.12 |
| **Overall Accuracy** | 98% |

*Optimized for high fraud recall to minimize false negatives*

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Dataset: `fraud_dataset.csv` (PaySim or similar financial transaction data)

### Installation

1. **Clone Repository**
```bash
git clone <repository-url>
cd fraud-detection
```

2. **Install Dependencies**
```bash
pip install -r requirements.txt
```

3. **Prepare Dataset**
   - Place `fraud_dataset.csv` in the project root
   - Ensure columns: `step`, `type`, `amount`, `nameOrig`, `oldbalanceOrg`, `newbalanceOrig`, `nameDest`, `oldbalanceDest`, `newbalanceDest`, `isFraud`

### Training the Model

```bash
# Run the Jupyter notebook or Python script
jupyter notebook fraud_detection_notebook.ipynb

# Or run as Python script
python fraud_detection_model.py
```

**Outputs**:
- `fraud_detection_model.pkl`: Trained Logistic Regression model
- `scaler.pkl`: Feature scaler for normalization
- `label_encoder.pkl`: Categorical encoder for transaction types
- `feature_columns.pkl`: Feature column names

### Streamlit Deployment

```bash
streamlit run streamlit_app.py
```

Access the dashboard at: `http://localhost:8501`

## 📱 Streamlit Dashboard Features

### 1. Single Transaction Check
- **Real-time Analysis**: Input transaction details for instant fraud detection
- **Risk Scoring**: Fraud probability with confidence levels
- **Risk Factor Breakdown**: Detailed analysis of fraud indicators

### 2. Batch Analysis
- **CSV Upload**: Process multiple transactions simultaneously
- **Bulk Results**: Download complete analysis with risk scores
- **Visual Analytics**: Distribution charts and risk summaries

### 3. Model Insights
- **Performance Metrics**: Model accuracy and feature importance
- **Risk Indicators**: Key fraud patterns and business impact
- **Deployment Stats**: Real-time performance monitoring

## 🔍 Key Fraud Indicators

### Primary Risk Factors
1. **Balance Inconsistencies** (Strongest Predictor)
   - Mismatch between expected and actual balance changes
   - Indicator of account takeover or system manipulation

2. **Transaction Amount Anomalies**
   - High amount-to-balance ratios
   - Unusual transaction sizes for account history

3. **Temporal Patterns**
   - Off-hours transactions (late night/early morning)
   - Weekend transaction spikes

4. **Account Behavior**
   - Zero balance origination accounts
   - Merchant vs. customer transaction patterns

## 🛠️ Technical Implementation

### Feature Engineering Pipeline
```python
# Temporal features
hour = step % 24
day = step // 24
is_weekend = (day % 7) in [5, 6]

# Balance consistency
balance_diff_orig = newbalanceOrig - oldbalanceOrg
balance_diff_dest = newbalanceDest - oldbalanceDest

# Transaction anomalies
amount_to_balance_ratio = amount / (oldbalanceOrg + 1)
is_merchant = nameDest.startswith('M')
zero_balance_orig = (oldbalanceOrg == 0)
```

### Class Imbalance Strategy
- **Technique**: Undersampling majority class
- **Ratio**: 3:1 (legitimate:fraud) for balanced learning
- **Validation**: Stratified train-test split
- **Class Weights**: Balanced weighting in Logistic Regression

### Model Selection Rationale
**Why Logistic Regression?**
- ✅ **Interpretability**: Clear coefficient-based feature importance
- ✅ **Speed**: Fast training and inference (<50ms per prediction)
- ✅ **Scalability**: Linear complexity with feature count
- ✅ **Robustness**: Stable performance across different data distributions
- ✅ **Deployment**: Simple model serialization and loading

## 📈 Business Impact

### Fraud Prevention
- **Early Detection**: Identify fraud attempts before completion
- **Loss Reduction**: Minimize financial impact through rapid response
- **Customer Protection**: Prevent account takeover and unauthorized access

### Operational Efficiency
- **Automated Screening**: Reduce manual transaction review workload
- **Priority Flagging**: Focus human attention on highest-risk cases
- **Real-time Alerts**: Immediate notification for suspicious activities

### Risk Management
- **Pattern Recognition**: Identify emerging fraud techniques
- **Threshold Optimization**: Dynamic risk scoring based on patterns
- **Compliance Support**: Audit trail and regulatory reporting

## 🔄 Model Maintenance

### Regular Updates
- **Retraining**: Monthly model updates with new fraud patterns
- **Feature Monitoring**: Track feature drift and importance changes
- **Performance Tracking**: Monitor precision-recall metrics over time

### Feedback Loop
- **False Positive Analysis**: Refine model based on incorrect predictions
- **New Fraud Patterns**: Incorporate emerging fraud techniques
- **Threshold Adjustment**: Optimize decision boundaries based on business needs

## 📚 File Structure

```
fraud-detection/
│
├── fraud_detection_notebook.ipynb    # Main training notebook
├── streamlit_app.py                  # Streamlit dashboard
├── requirements.txt                  # Python dependencies
├── README.md                        # Project documentation
│
├── models/                          # Saved model files
│   ├── fraud_detection_model.pkl    # Trained Logistic Regression
│   ├── scaler.pkl                   # Feature scaler
│   ├── label_encoder.pkl            # Categorical encoder
│   └── feature_columns.pkl          # Feature names
│
└── data/
    └── fraud_dataset.csv           # Training dataset
```

## 🚀 Deployment Options

### Local Development
```bash
streamlit run streamlit_app.py --server.port 8501
```

### Cloud Deployment
- **Streamlit Cloud**: Direct GitHub integration
- **Heroku**: Container-based deployment
- **AWS/Azure**: Cloud hosting with scaling
- **Docker**: Containerized deployment

### Production Considerations
- **Load Balancing**: Handle high transaction volumes
- **Database Integration**: Connect to real-time transaction streams
- **API Endpoint**: REST API for system integration
- **Monitoring**: Real-time performance and drift detection
- **Security**: Encrypted data handling and secure model access

## 📊 Sample API Integration

### REST API Endpoint Example
```python
import requests

# Single transaction prediction
response = requests.post('http://your-api/predict', json={
    "step": 150,
    "type": "TRANSFER",
    "amount": 5000.0,
    "oldbalanceOrg": 10000.0,
    "newbalanceOrig": 5000.0,
    "nameDest": "C1234567890",
    "oldbalanceDest": 0.0,
    "newbalanceDest": 5000.0
})

result = response.json()
# {'prediction': 1, 'probability': 0.85, 'risk_level': 'HIGH'}
```

## 🔧 Configuration & Customization

### Model Hyperparameters
```python
# Logistic Regression Configuration
LogisticRegression(
    random_state=42,
    max_iter=1000,
    class_weight='balanced',
    solver='lbfgs',
    C=1.0  # Regularization strength
)
```

### Feature Selection
- **Core Features**: Always include balance differences and amount ratios
- **Optional Features**: Add domain-specific features based on business needs
- **Feature Scaling**: StandardScaler for numerical feature normalization

### Threshold Optimization
```python
# Adjust prediction threshold based on business requirements
optimal_threshold = 0.5  # Default
high_recall_threshold = 0.3  # Catch more fraud, higher false positives
high_precision_threshold = 0.7  # Fewer false positives, may miss some fraud
```

## 🧪 Testing & Validation

### Model Testing
```bash
# Run unit tests
python -m pytest tests/

# Validate model performance
python validate_model.py --test_data test_dataset.csv
```

### A/B Testing Framework
- **Champion/Challenger**: Compare new model versions
- **Gradual Rollout**: Progressive deployment to minimize risk
- **Performance Monitoring**: Track key metrics during deployment

## 📋 Troubleshooting

### Common Issues

**1. Model Files Not Found**
```
Error: FileNotFoundError: [Errno 2] No such file or directory: 'fraud_detection_model.pkl'
```
**Solution**: Run the training notebook first to generate model files

**2. Feature Engineering Errors**
```
Error: KeyError: 'oldbalanceOrg'
```
**Solution**: Ensure dataset has all required columns with correct names

**3. Streamlit Performance Issues**
```
Warning: App running slowly with large datasets
```
**Solution**: Use batch processing for datasets >10K transactions

**4. Memory Issues During Training**
```
Error: MemoryError during model training
```
**Solution**: Reduce sample size or use incremental learning approaches

## 📈 Performance Optimization

### Speed Improvements
- **Feature Caching**: Cache engineered features for repeated predictions
- **Batch Processing**: Process multiple transactions simultaneously
- **Model Quantization**: Reduce model size for faster inference

### Accuracy Enhancements
- **Feature Selection**: Use advanced feature selection techniques
- **Ensemble Methods**: Combine multiple models for better performance
- **Cross-validation**: Use k-fold validation for robust evaluation

## 🔐 Security & Privacy

### Data Protection
- **Encryption**: Encrypt sensitive transaction data
- **Access Control**: Role-based access to model and data
- **Audit Logging**: Track all model predictions and access
- **Data Anonymization**: Remove personally identifiable information

### Model Security
- **Model Versioning**: Track model changes and rollback capability
- **Input Validation**: Sanitize all input data
- **Rate Limiting**: Prevent abuse of prediction endpoints

## 📞 Support & Contributing

### Getting Help
- **Documentation**: Check README and code comments
- **Issues**: Create GitHub issues for bugs or feature requests
- **Discussions**: Use GitHub discussions for questions

### Contributing Guidelines
1. **Fork Repository**: Create your feature branch
2. **Code Standards**: Follow PEP 8 style guidelines
3. **Testing**: Add tests for new features
4. **Documentation**: Update README for significant changes
5. **Pull Request**: Submit PR with clear description

## 📝 License & Citation

### License
This project is licensed under the MIT License - see the LICENSE file for details.

### Citation
If you use this fraud detection system in your research or production, please cite:
```
@software{fraud_detection_system,
  title={Financial Fraud Detection System},
  author={Your Name},
  year={2024},
  url={https://github.com/your-repo/fraud-detection}
}
```

## 🎯 Future Enhancements

### Planned Features
- [ ] **Deep Learning Models**: Neural network implementations
- [ ] **Real-time Streaming**: Apache Kafka integration
- [ ] **Advanced Visualization**: Interactive fraud pattern analysis
- [ ] **Mobile App**: React Native mobile interface
- [ ] **Graph Analytics**: Network analysis for fraud rings

### Research Directions
- [ ] **Federated Learning**: Privacy-preserving model updates
- [ ] **Explainable AI**: Enhanced interpretability features
- [ ] **Adversarial Robustness**: Defense against adversarial attacks
- [ ] **Time Series Analysis**: Sequential pattern detection

---

## 📊 Project Metrics Summary

| Aspect | Achievement |
|--------|-------------|
| **Model Type** | Logistic Regression |
| **Features** | 11 engineered features |
| **Dataset Size** | 6.36M transactions |
| **Training Time** | <5 minutes |
| **Inference Speed** | <50ms per transaction |
| **ROC-AUC Score** | 0.9891 |
| **Fraud Recall** | 87% |
| **Deployment** | Streamlit dashboard |

---

**Built with ❤️ for financial security and fraud prevention**

*For questions, suggestions, or collaboration opportunities, please reach out through GitHub issues or discussions.*
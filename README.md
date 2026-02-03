# ML Lab

A comprehensive desktop application for exploring datasets, training machine learning models, and evaluating their performance. Built with Python and Flet for a beautiful, cross-platform UI.

## Features

- **Dataset Management**
  - Load and explore CSV files with intuitive data browsing
  - View dataset statistics and summary information
  - Edit individual cell values directly in the table
  - Rename and drop columns
  - Handle missing values (detect NaN rows, drop duplicates)
  - Export modified datasets with timestamps
  - Filter and search by row index or range
  - Replace values in selected columns
  - Change data types of columns

- **Data Visualization**
  - Interactive charts: scatter plots, histograms, bar charts, box plots, pie charts, heatmaps
  - Customizable visualizations based on selected columns
  - Real-time chart generation

- **Machine Learning Models**
  - **Classification Models**: Logistic Regression, Decision Tree, Random Forest, SVM, KNN, Gradient Boosting
  - **Regression Models**: Linear Regression, Decision Tree Regressor, Random Forest
  - **Clustering Models**: K-Means, DBSCAN, Hierarchical Clustering, Gaussian Mixture Models, Mean Shift, HDBSCAN, Affinity Propagation, MiniBatch K-Means
  - **Helper Tools**: Elbow locator for optimal cluster selection

- **Model Configuration**
  - Flexible hyperparameter tuning with validation
  - Multiple scaling options: Standard Scaler, MinMax Scaler, None
  - Configurable train/test split for supervised learning
  - Automatic feature selection

- **Model Evaluation**
  - Classification metrics: Accuracy, Precision, Recall, F1-Score, Confusion Matrix
  - Regression metrics: R² Score, Adjusted R² Score, Mean Squared Error, Mean Absolute Error
  - Clustering metrics: Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz Index
  - Detailed results with copy functionality

- **Data Profiling**
  - YData Profiling integration for comprehensive dataset reports
  - Export HTML reports for detailed analysis

## Installation

### Using uv (Recommended)

First, ensure you have [uv](https://github.com/astral-sh/uv) installed.

```bash
# Clone the repository
git clone https://github.com/farshadz1997/ML-Lab.git
cd "ML-Lab"

# Create virtual environment & Install dependencies
uv sync

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

cd src

# Run the app as a desktop application
uv run main.py
```

### Using pip

```bash
# Clone the repository
git clone https://github.com/farshadz1997/ML-Lab.git
cd "ML-Lab"

# Create a virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

cd src

# Run the app as a desktop application
python main.py
```

## Quick Start

1. **Load a Dataset**: Click "Open Dataset" and select a CSV file
2. **Explore Data**: Browse the dataset, view statistics, check for missing values
3. **Select Model**: Navigate to the Model Training tab
4. **Configure Hyperparameters**: Adjust model settings as needed
5. **Train & Evaluate**: Click "Train and evaluate model" to see results
6. **Visualize**: Use the Data Visualization tab to create charts

## Requirements

- Python 3.11+
- See `requirements.txt` for all dependencies

## Building for Distribution

### Windows

```bash
flet build windows -v
```

For more details on building Windows package, refer to the [Windows Packaging Guide](https://flet.dev/docs/publish/windows/).

## Project Structure

```
src/
├── main.py                 # Application entry point
├── core/
│   └── dataset.py         # Dataset handling and operations
│   └── data_preparation.py          # Main encoding orchestration
│       ├── prepare_data_for_training()       - Complete data prep pipeline
│       └── prepare_data_for_training_no_split() - Clustering variant
├── ui/
│   ├── layout.py          # Main UI layout
│   ├── data_visualization.py
│   ├── dataset_explorer.py # Dataset exploration interface
│   ├── model_factory.py    # Model training interface
│   ├── charts/            # Chart implementations
│   └── models/            # ML model wrappers
├── utils/
│    └── model_utils.py     # Model utilities and metrics
│       ├── detect_categorical_columns()
│       ├── validate_cardinality()
│       ├── create_categorical_encoders()
│       ├── apply_encoders()
│       ├── get_encoding_mappings()
│       ├── build_preprocessing_pipeline()
│       ├── compose_full_model_pipeline()
│       ├── get_categorical_encoding_info()
│       └── Data classes: EncodingError, CardinalityWarning, CategoricalEncodingInfo
```

## License

This project is licensed under the MIT License - see LICENSE file for details.

## Contributing

Contributions are welcome! Feel free to submit issues and pull requests.

# basepharyngitis
Here’s the updated README with the new folder structure and project details incorporated:

---

# BasePharyngitis  

**BasePharyngitis** is a machine learning project that classifies throat images into two categories:  
- **Bacterial Pharyngitis**  
- **Non-Bacterial Pharyngitis**  

This project uses deep learning models and k-fold cross-validation to ensure robust classification. It integrates throat image data and metadata for improved diagnostics and includes utilities for preprocessing, training, and evaluation.

---

## Features
- **Throat Image Classification**: Classifies images into bacterial and non-bacterial categories.  
- **K-Fold Cross-Validation**: Implements folder-based k-fold data splitting for model evaluation.  
- **Histogram Matching**: A preprocessing feature for standardizing image intensity distributions.  
- **Excel Metadata Integration**: Reads diagnostic data from Excel files.  
- **Predefined Densenet Model**: Uses DenseNet for deep feature extraction and classification.  

---

## Prerequisites
- Python 3.8 or later  
- Required Python packages (install them with `pip install -r requirements.txt`)  

---

## Setup Instructions  

1. **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/basepharyngitis.git
    cd basepharyngitis
    ```

2. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Configure Settings**:  
   Edit the `settings.py` file to set paths and constants:

    ```python
    # settings.py

    # Main directory for throat images
    MAIN_DIR = "path/to/images/folder"

    # Path to the Excel metadata file
    EXCEL_FILE = "path/to/excel/file.xlsx"

    # Other constants
    BACTERIAL = 1
    NON_BACTERIAL = 0
    IMAGE_WIDTH = 224
    IMAGE_HEIGHT = 224
    ```

4. **Run the program**:
    Use the provided scripts based on your workflow:
    - For dataset generation:  
      ```bash
      python DatasetGen.py
      ```
    - For k-fold cross-validation:  
      ```bash
      python main_file_k_fold_folder_based.py
      ```

---

## File Structure  

```
basepharyngitis/
│
├── DatasetGen.py                    # Dataset handling logic
├── main_file_k_fold_folder_based.py # K-fold cross-validation logic
├── densenet4.py                     # Predefined DenseNet model for classification
├── histogram_matching_phar.py       # Implements histogram matching for preprocessing
├── settings.py                      # Configuration file for paths and constants
├── util/                            # Utility scripts (e.g., preprocessing functions)
├── data/                            # Placeholder for example data
├── requirements.txt                 # Python dependencies
└── README.md                        # Project documentation
```

---

## Example Usage  

1. **Prepare your data**:  
   - Place throat images in the directory specified by `MAIN_DIR`.  
     Subdirectories should be named as specified in `settings.py` (e.g., `bact/` and `nonbact/`).  
   - Ensure the Excel metadata file is correctly formatted.  

2. **Generate and preprocess the dataset**:
    ```bash
    python DatasetGen.py
    ```

3. **Run k-fold cross-validation**:
    ```bash
    python main_file_k_fold_folder_based.py
    ```

4. **Customize your preprocessing**:  
   Modify `histogram_matching_phar.py` to apply histogram matching to your images for standardization.  

---

## Customization  

- **Interval Selection**:  
  Use `IntervalDatasetGenerator` in `DatasetGen.py` to define data intervals:
    ```python
    datasetTrain = IntervalDatasetGenerator(pathImageDirectory=MAIN_DIR, pathDatasetFile=EXCEL_FILE, 
                                            transform=get_train_preprocess((IMAGE_WIDTH, IMAGE_HEIGHT)))
    datasetTrain = datasetTrain.set_interval(10, 21)
    ```

- **Model Customization**:  
  Modify `densenet4.py` to fine-tune the DenseNet architecture as per your requirements.  

---

## Contributing  

We welcome contributions to enhance BasePharyngitis. To contribute:  
1. Fork the repository.  
2. Create a feature branch (`git checkout -b feature-name`).  
3. Commit your changes (`git commit -m 'Add feature'`).  
4. Push to the branch (`git push origin feature-name`).  
5. Open a pull request.  

---

## License  

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.  

---  

Let me know if you’d like additional refinements or adjustments!

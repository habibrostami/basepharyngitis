# BasePharyngitis  



<br>

This is the repository for the official implementation of [A publicly available pharyngitis dataset and baseline evaluations for bacterial or nonbacterial classification](https://doi.org/10.1038/s41597-025-05780-5) published in _Scientific Data 12, Article number: 1418 (2025)_ 

# 🔖 Citation
If you find this code helpful in your research, please cite the following paper and dataset:
```
@article{Shojaei2025,
  author    = {Negar Shojaei and Habib Rostami and Mohammad Barzegar and Shokooh Saadat Farzaneh and Zohreh Farrar and Majid Alimohammadi and Jahanbakhsh Keyvani and Mehdi Mirzad and Leila Gonbadi},
  title     = {A publicly available pharyngitis dataset and baseline evaluations for bacterial or nonbacterial classification},
  journal   = {Scientific Data},
  year      = {2025},
  volume    = {12},
  number    = {1},
  pages     = {1418},
  doi       = {10.1038/s41597-025-05780-5},
  url       = {https://doi.org/10.1038/s41597-025-05780-5},
  issn      = {2052-4463}
}
```

```
@misc{shojaei2025pharyngitis,
  author       = {Negar Shojaei and Habib Rostami and Mohammad Barzegar and Shokooh Saadat Farzaneh and Zohreh Farrar and Majid Alimohammadi and Jahanbakhsh Keyvani and Mehdi Mirzad and Leila Gonbadi},
  title        = {A publicly available pharyngitis dataset and baseline evaluations for bacterial or nonbacterial classification},
  year         = {2025},
  doi          = {10.6084/m9.figshare.28163513},
  url          = {https://doi.org/10.6084/m9.figshare.28163513},
  note         = {figshare}
}
```

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

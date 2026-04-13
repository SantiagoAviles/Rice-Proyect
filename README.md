# Classification of Peruvian Rice Grains Using Convolutional Neural Networks (CNN)

This project applies Deep Learning techniques for the automated classification of Peruvian rice grain quality. It evaluates and compares five state-of-the-art architectures to identify defects in rice grains, aiming to optimize and modernize agricultural quality control processes.

Additionally, this work explores the impact of transfer learning and data augmentation strategies in improving model generalization and robustness when working with real-world agricultural datasets.

---

## Dataset

The original images were obtained from Kaggle:

**Peruvian Rice Dataset**  
https://www.kaggle.com/datasets/cristhiansempertegui/dataset-de-arroz-peruano

The dataset contains labeled images of rice grains categorized based on their physical condition and visible defects.

---

## Preprocessing and Data Augmentation

To improve model robustness and mitigate overfitting, a Data Augmentation process was applied to the original dataset. This includes transformations such as:

- Rotation  
- Brightness adjustment  
- Zoom  
- Horizontal and vertical flipping  

These transformations resulted in a balanced dataset named `balanceado_aug`, which helps ensure that all classes are equally represented during training and prevents bias toward dominant classes.

---

## Classes

The dataset includes the following categories:

- **Whole**: Grain in optimal condition.  
- **Stained**: Grains with abnormal or fungal discoloration.  
- **Broken**: Fragmented grains.  
- **Chalky**: Grains with an opaque or chalk-like appearance.  

---

## Project Structure

The file organization follows reproducibility standards for computer vision projects:

```text
.
├── CODE/                           # Google Colab Notebooks
│   ├── MobileNetV2_Rice.ipynb      # Training and evaluation of MobileNetV2
│   ├── EfficientNetB0_Rice.ipynb   # Training and evaluation of EfficientNetB0
│   ├── ResNet50_Rice.ipynb         # Training and evaluation of ResNet50
│   ├── DenseNet121_Rice.ipynb      # Training and evaluation of DenseNet121
│   └── InceptionV3_Rice.ipynb      # Training and evaluation of InceptionV3
├── DATASET/                        # Data files (compressed and extracted)
│   ├── PeruvianRiceDataset.rar     # Original Kaggle Dataset
│   ├── balanceado_aug.rar          # Dataset after Data Augmentation
│   └── PeruvianRiceDataset/        # Extracted folder structure
│       ├── train/                  # (70%) Training folder (classes: entero, mancha, quebrado, tiza)
│       ├── val/                    # (15%) Validation folder (classes: entero, mancha, quebrado, tiza)
│       └── test/                   # (15%) Test folder (classes: entero, mancha, quebrado, tiza)
├── .gitattributes                  # Git LFS configuration
├── LICENSE                         # MIT License
└── README.md                       # Main documentation
```
---

## Methodology

### Evaluated Models

Five architectures were selected based on their proven performance on ImageNet:

- **MobileNetV2**: Optimized for mobile and computational efficiency.  
- **EfficientNetB0**: Balanced trade-off between accuracy and parameter count.  
- **ResNet50**: Uses residual connections to prevent vanishing gradients.  
- **DenseNet121**: Dense connections for maximum feature reuse.  
- **InceptionV3**: Multi-scale processing through Inception modules.  

All models leverage transfer learning with pre-trained ImageNet weights, followed by fine-tuning on the rice dataset.

---

### Dataset Split

A strict data partitioning strategy is used to ensure statistical validity:

- **Training (70%)**: Model weight optimization  
- **Validation (15%)**: Hyperparameter tuning and loss monitoring  
- **Test (15%)**: Final evaluation using unseen data  

---

## Evaluation and Results

Models are compared using the following metrics:

- **Accuracy**: Overall prediction correctness  
- **Loss**: Error during model training  
- **Confusion Matrix**: Class-wise performance visualization (whole, stained, broken, chalky)  

These metrics provide a comprehensive understanding of each model’s strengths and weaknesses.

---

## Learning Curve Comparison

The project generates comparative plots including:

- **Training Accuracy Comparison**: Overlapping curves of all models to observe convergence speed  
- **Test Accuracy Comparison**: Final performance evaluation on the test dataset  

These visualizations help identify the most efficient and accurate architecture.

---

## Reproducibility and Transparency

To replicate the results:

1. Upload the `DATASET` folder to your local environment or Google Drive  
2. Open the notebooks in the `CODE` folder using Google Colab  
3. Enable GPU runtime (T4 or higher recommended)  
4. Update dataset paths accordingly  
5. Execute all cells step by step  

---

## License

This project is licensed under the MIT License.

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

---

## Notes

- The use of `balanceado_aug.rar` is critical to prevent bias toward classes with more original samples.  
- All models are initialized with ImageNet pre-trained weights (Transfer Learning).  
- Proper preprocessing and dataset balancing significantly improve model generalization in real-world scenarios.

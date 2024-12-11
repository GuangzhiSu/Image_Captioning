# Image Captioning with CNN-LSTM and CNN-Transformer Architectures

This repository contains implementations and experiments for image captioning models using two different architectures: **CNN-LSTM** and **CNN-Transformer**. The project aims to evaluate and improve the performance of image captioning through innovative architectural designs and advanced evaluation techniques.

---


### **Files and Directories**
- **CNN-Transformer architecture/**: Contains code and scripts for the CNN-Transformer-based captioning model.
- **Image_captioning_CNNLSTM.ipynb**: A Jupyter Notebook that implements the CNN-LSTM image captioning model.

---

## **Models**
### **CNN-LSTM Architecture**
- **Encoder**: Uses a pre-trained ResNet model to extract image features.
- **Decoder**: A Long Short-Term Memory (LSTM) network generates captions word by word based on the extracted features.

### **CNN-Transformer Architecture**
- **Encoder**: Similar to the CNN-LSTM model, this uses a pre-trained ResNet model for image feature extraction.
- **Decoder**: A Transformer-based decoder introduces advanced components such as:
  - **Positional Encoding**: Captures sequence order information.
  - **Self-Attention**: Captures relationships between words in the generated sequence.
  - **Cross-Attention**: Dynamically conditions predictions on visual inputs.

---

## **Dataset**
The **Flickr8k dataset** is used for training and testing. It is split into:
- **80% Training Set**
- **20% Testing Set**

The dataset is preprocessed to generate a vocabulary, and image features are extracted using the ResNet encoder.

---

## **Evaluation**
### **Metrics**
- **BLEU Scores (BLEU-1 to BLEU-4)**:
  - Evaluate the quality of generated captions by comparing them to reference captions.
  - Higher BLEU scores indicate better alignment with reference captions.

### **Visualization**
- First 5 images from the evaluation set are saved with their corresponding predicted captions for qualitative analysis.

---

## **Requirements**
- Python 3.x
- PyTorch
- NumPy
- Matplotlib
- torchvision
- nltk (for BLEU score computation)

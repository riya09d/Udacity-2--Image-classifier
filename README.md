## Udacity-2--Image-classifier
Overview

This project uses PyTorch to train a deep learning model for image classification. The model is trained on the Oxford Flowers Dataset, which consists of 102 flower categories.

Requirements

- Python 3.8+
- PyTorch 1.9+
- Torchvision 0.10+
- NumPy 1.20+
- Matplotlib 3.4+
- Pillow 8.4+

Project Structure


image-classification/
data/
flowers/
train/
valid/
test/
models/
vgg16.pth
model.pth
notebooks/
Image Classification.ipynb
(link unavailable)
requirements.txt


Usage

Training the Model

1. Clone the repository: `git clone (link unavailable)
2. Install requirements: pip install -r requirements.txt
3. Train the model: python (link unavailable)

Making Predictions

1. Load the trained model: model = load_checkpoint('model.pth')
2. Make predictions: probabilities, classes = predict('image_path.jpg', model)

Files

- Trains the model using the Oxford Flowers Dataset.
- Makes predictions on new images using the trained model.
- models/vgg16.pth: Pre-trained VGG16 model.
- model.pth: Trained model checkpoint.
- Image Classification.ipynb: Jupyter Notebook for exploring the project.


Acknowledgments

- Oxford Flowers Dataset: (link unavailable)
- PyTorch: (link unavailable)

License

This project is licensed under the MIT License. See LICENSE for details.

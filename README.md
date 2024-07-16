# Detecting-and-Classifying-Chest-Disease
Training a Deep Learning model to to classify diseases using X-ray images.

# Problem Statement
The purpose of the project is to classify X-Ray images by applying DL models. The steps taking involves, understanding the dataset, preprocessing the data, building a nueral network and evaluating its performance

# Libaries
The following libraries were used for this project:
- TensorFlow
- NumPy
- OpenCV
- Matplotlib
- Seaborn
- Pandas

# Data Preprocessing
Images are normalized using an image generator, and 20% of the data is used for cross-validation.
`image_generator = ImageDataGenerator(rescale=1./255, validation_split=0.2)`

# Model Building
A ResNet50 architecture was used for this task. Layers include convolutional layers, pooling layers, batch normalization, and dense layers.
`from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten
input_tensor = Input(shape=(224, 224, 3))
base_model = ResNet50(include_top=False, weights='imagenet', input_tensor=input_tensor)
x = Flatten()(base_model.output)
output_tensor = Dense(1, activation='sigmoid')(x)
model = Model(inputs=input_tensor, outputs=output_tensor)`


# Model Training
The model is compiled and trained using appropriate optimizers and loss functions. Early stopping and learning rate reduction are applied to enhance training efficiency.
`model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(train_generator, epochs=50, validation_data=validation_generator,
          callbacks=[EarlyStopping(patience=5), ReduceLROnPlateau()])`

# Model Performance
When applied on the test set, the result showed the model achieved an accuracy of 58% and loss of 8.4 on the test sets, which is below the expected performance standard for a disease classification model.
The model was tested on 40 images with 4 classes.

# Areas of Improvement
- Experiment with different Architectures like EfficientNet, DenseNet, or VGG
- Fine tune ResNet model by adjusting number of layers and neurons and other hyperparameters used in training
- Experiment with different batch size and training epoch to create mode opportunities for the model to learn  
- Explore different optimizers like Adam, RMSprop, or SGD with momentum to see which works best for your model.
- Explore implementing k-fold cross-validation to ensure the model's robustness and ability to generalise.


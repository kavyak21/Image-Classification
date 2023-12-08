# Image Classification using Convolutional Neural Network

This task involves building a Convolutional Neural Network (CNN) to classify images of five different personalities: Lionel Messi, Maria Sharapova, Roger Federer, Serena Williams, and Virat Kohli.
The CNN model architecture consists of convolutional layers, max-pooling layers, a flattening layer, and dense layers. The model is compiled using the Adam optimizer and sparse categorical crossentropy as the loss function.

**Model Summary**

Accuracy: The model achieved an accuracy of 76.47% on the test set. This indicates the percentage of correctly predicted labels.

Classification Report: The classification report provides additional insights into precision, recall, and F1-score for each class.
The macro average F1-score is 0.77, suggesting a reasonable balance between precision and recall across classes.
The weighted average F1-score is 0.77, considering class imbalance.

**Training Process**

The model was trained for 50 epochs with a batch size of 32.

**Critical Findings**

The model successfully predicted the personalities of five sample images.
Class 0 (lionel Messi) has a perfect precision but a lower recall, indicating that some Messi images were misclassified as other classes.
Class 3 (serena williams) has a relatively low precision, suggesting that there are false positives in this class.

# Dog-vs-Cat-image-classification
This is simple mini project which predicts image of dog or cat with the help of machine learning algorithm called Convolution Neural Network (CNN).

The idea was taken from [https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition](url)

![catsdogs](https://user-images.githubusercontent.com/37996516/38716131-b5dda1aa-3efc-11e8-8d01-57740370809f.jpg)

# Model Implementation

This is simple model which uses 1 conv layer and 2 Fully connected layers at the last.
 
Layer Sequence
`
[Input (64, 64, 3)] -> [Conv (32)] -> [Pool]  -> [Full (128)] -> [Full (1)] -> [Output]
`
Convolution layer Filter size = 3 X 3
Max pool uses size of 2 X 2
learning rate = 0.001 
dropout = 0.2 from prevent overfitting

# Result
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_1 (Conv2D)            (None, 62, 62, 32)        896       
_________________________________________________________________
dropout_1 (Dropout)          (None, 62, 62, 32)        0         
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 31, 31, 32)        0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 30752)             0         
_________________________________________________________________
dense_1 (Dense)              (None, 128)               3936384   
_________________________________________________________________
dense_2 (Dense)              (None, 1)                 129       
=================================================================
Total params: 3,937,409
Trainable params: 3,937,409
```
 ### Loss : 0.3721 
### Accuracy : 0.8302 


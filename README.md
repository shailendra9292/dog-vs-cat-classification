# Dog-vs-Cat-image-classification
This is simple mini project which predicts image of dog or cat with the help of machine learning algorithm called Convolution Neural Network (CNN).

The idea was taken from [https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition](url)

![catsdogs](https://user-images.githubusercontent.com/37996516/38716131-b5dda1aa-3efc-11e8-8d01-57740370809f.jpg)

# Model Implementation

This is simple model which uses 2 conv layer, 2 Maxpool layer and 2 Fully connected layers at the last.
 
Layer Sequence
```
[Input (64, 64, 3)] -> [Conv (128)] -> [Pool]  ->[Conv (128)] -> [Pool]  ->  [Full (128)] -> [Full (1)] -> [Output]
```

- Convolution layer Filter size = 3 X 3
- Max pool uses size of 2 X 2
- learning rate = 0.001 
- dropout = 0.2 from prevent overfitting
- no_of_epochs = 50
- train_test_split = 80%-20%


-> Total time taken 25.48 hours for entire program to execute. Laptop configuration is :
- Microsoft Windows 10
- GPU- Nvidia Geforce 940-MX(4GB)
- RAM - 4GB
- Processor - Intel Core i5 6200 @ 2.8 - 3.0 GHz
- Tensorlfow version = 1.7
- Keras use as tensorflow backend

# Result
```
Loss : 0.2421 
Accuracy : 0.9526 

```

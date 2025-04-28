# Feature-Engineering
**Started as a fun/educational endeavor. Ended up learning a lot.**

**Note**: All the heatmaps, graphs, and predictions are in the python notebooks. :)

## Dataset

The dataset used for this project is the Keras mnist dataset.

## Overview

I began this project with the intention of **learning through implementation**. My goal was to build everything from scratch, avoiding libraries whenever possible, and hardcoding the models. In the process, I found myself, shall I say, "down the rabbit hole."

## Why I Started This Project

Ever since taking a Python course in high school during the COVID-19 lockdown, I’ve grown increasingly fascinated by machine learning, especially in how models perceive their environment through visual inputs. Building a model to recognize handwritten digits seemed like the perfect starting point to dive into this realm.

## Key Objective

- Develop image processing techniques to preprocess data and extract features that balance accuracy and efficiency. Why extract features? I mean, for a task as simple as recognizing digits, passing a flattened array of the image would be sufficient, although a lot of important information like spacial relationships will be lost. But a little thought would make it clear that for more complex, high-res, information rich images, it would be impractical to pass the entire image as the input, I think, and being able to extract features allows us to reduce the input dimension while being able to look "deeper" into the image and keep the important bits. Atleast that's what I learned about CNNs. But no CNNs in this project, well, its kinda there, maybe more manual and permanent than the conventional approaches where the convolutions are learnt through the dataset.

## What I Went Through

### Data Collection and Preprocessing

The MNIST dataset, containing 70000 labeled images, served as the foundation for this project. I explored two main approaches to preprocess the data:

1. **Feature Engineering with a DCT-II transformation matrix and PCA to extract the variation rich dimensions**  
   - Extracted frequency components of the images.
   - Initially, I used a 7x7 DCT-II transformation for each 7x7 grid on the image, just like in image compression use cases. While it yielded good performance, with some though I realised it completely misses information between each 7x7 grid of the image as they don't overlap. Therefore, I settled on using a DCT-II transformation matrix for the entire image(28x28). This, approach considerably lowered the validation loss of the model indicating more confidence in its predictions-better fit over the probability distributions for each number.
   - Used PCA by SVD to obtain the top **K** principal components of the dataset transformed into the frequency space so as to capture 90% of the variance in the dataset.
   - Resulted in a feature array of dimensions (70000, 57). (**Samples** = 70000, **K** = 57)

2. **Flattened 1D Array**  
   - Flattened the entire image into a 1D array. This approach captured pixel-wise information but increased computational costs.  

I also normalised the images and centered them based on the mean of the pixels with non-zero intensities.

### Building the Model

I initially started with a softmax regression model for multi-class classification. Before you start, yes I know it is a linear model and my task is highly non linear in nature. But for my purposes, I aimed for reasonable accuracy in predictions which, suffices to say, I did not get. 

Therefore, I decided to use neural networks that output a probability vector for the numbers 0-9 to accomplish my goal since they capture non-linearities well due to their non-linear activation functions. 

**Key observations**: 
- For this use case, I found that 2 layers with 64 and 32 nodes was sufficient to yield impressive accurcacy without any trade-off in efficiency for the image passed as a 1D array.
- But as I increased the complexity of the neural network, the training time grew exponentially for the image passed as a 1D array compared to the engineered features. Moreover, the model trained with engineered features had a notable increase in it's accuracy with an increase in the complexity of the neural network. 


### Model Evaluation and Improvement

- Throughout the process, I evaluated the model’s performance using metrics like accuracy and loss, and I fine-tuned parameters such as the learning rate and number of epochs to improve results.
- I also found out that a batch-size of 64 over a number of 8 epochs performed best on a neural network with 2 hidden layers consisting of 828 and 512 neurons. With more neurons, the validation loss kept going down, but I lost my patience.
- An increase in batch size led to a lower validation loss indicating higher confidence in predictions and lower training time per epoch, but slower convergence.
- In the end, for the DCT + PCA approach and the 1D array approach, there were minimal differences in the validation losses and accuracies. But, the training time for the 1D array approach was approximately 4x longer than the feature engineering approach.
- Also, for a reason still not yet known to me, for zeroes that were small, literally smaller, the DCT + PCA approach kept classifying them as 7s.

## Learning Outcome

This project gave me hands-on experience with:

- **Supervised Learning**: Training models on labeled data.  
- **Feature Extraction**: Engineering meaningful features for structured data.  
- **Overfitting and Underfitting**: Balancing model complexity for better generalization.  
- **TensorFlow & Keras**: Using pre-built tools to benchmark and validate my models.  
  *(Spoiler: My "built-from-scratch" models were slower but performed similarly in terms of validation losses and accuracies.)*  

## Key Learnings

- **Neural Networks**: Gained a deeper understanding of architectures, backpropagation, training dynamics and solidified my understanding of the math behind everything. Especially the part where it "captures non-linearities". It felt amazing when I understood that after every linear transformation-information flow from one layer to another-an activation function like ReLU warps the space and this repeats until a single hyperplane can split the different classes, bwoah!  
- **Hyperparameter Tuning**: Understood the impact of parameters like learning rate, batch size, and layer sizes on performance.  
- **Trade-offs**: Learned to balance computational efficiency with model accuracy.

## Challenges Faced

- **Data Preprocessing**: Handling noisy or large datasets posed unexpected challenges.  
- **Model Overfitting**: Required a lot of experimentation with hyper parameters.  
- **Debugging**: Understanding *why* models behaved the way they did was occasionally frustrating but **very** rewarding.
- **Overall**: A lot.

## Conclusion

This project has been an enriching experience, strengthening my knowledge of machine learning and feature engineering. I now feel confident tackling more complex challenges.

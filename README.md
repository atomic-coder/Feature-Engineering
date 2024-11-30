# Feature-Engineering
**Started as a fun/educational endeavor. Ended up learning a lot.**

## Dataset

The dataset used for this project can be found [here](https://www.kaggle.com/datasets/scolianni/mnistasjpg).

Test images have been included in the repository for the reader's convenience.

## Overview

I began this project with the intention of **learning through implementation**. My goal was to build everything from scratch, avoiding libraries whenever possible, and hardcoding the models. In the process, I found myself, shall I say, "down the rabbit hole."

## Why I Started This Project

Ever since taking a Python course in high school during the COVID-19 lockdown, I’ve grown increasingly fascinated by machine learning, especially in how models perceive their environment through visual inputs. Building a model to recognize handwritten digits seemed like the perfect starting point to dive into this realm.

## Key Objectives

- Develop image processing techniques to preprocess data and extract features that balance accuracy and efficiency.
- Build a machine learning model capable of recognizing handwritten digits using the **MNIST dataset**.

## What I Went Through

### Data Collection and Preprocessing

The MNIST dataset, containing thousands of labeled images, served as the foundation for this project. I explored two main approaches to preprocess the data:

1. **Feature Engineering with a 7x7 Grid**  
   - Extracted statistical features (e.g., spread, standard deviation, and covariance) relative to the centroid of white pixels.  **Concept: Covariance**
   - Used the Sobel operator to calculate mean of gradient magnitudes and directions for each grid. **Concept: Histogram of gradients**
   - Resulted in a feature array of dimensions (n, 245).

2. **Flattened 1D Array**  
   - Flattened the entire image into a 1D array. This approach captured more information but increased computational costs.  

I also converted the images to black and white as I realised grayscale images had too much noise for the features I'd engineered. However, black and white images removed considerable information about gradients and texture that approach 2 was utilizing. Therefore, I decided to use the type of processing that yielded the best results for both approaches.

### Building the Model

I initially started with a softmax regression model for multi-class classification. Before you start, yes I know it is a linear model and my task is highly non linear in nature. But for my purposes, I aimed for reasonable accuracy in predictions which, suffices to say, I did not get. 

Therefore, I decided to use neural networks that output a probability vector for the numbers 0-9 to accomplish my goal. Initially, I used a 3x3 grid on the images which did not yield good performance, therefore I passed the entire image as a flattened 1D array which worked well in terms of accuracy but bad in terms of training time. This led me to ask questions about the trade-off in efficiency and accuracy in both approaches

**Key observations**: 
- For this use case, I found that 2 layers with 64 and 32 nodes was sufficient to yield impressive accurcacy without any trade-off in efficiency for the image passed as a 1D array.
- But as I increased the complexity of the neural network, the training time grew exponentially for the image passed as a 1D array compared to the engineered features. Moreover, the model trained with engineered features had a notable increase in it's accuracy as well.  


### Model Evaluation and Improvement

- Throughout the process, I evaluated the model’s performance using metrics like accuracy and loss, and I fine-tuned parameters such as the learning rate and number of epochs to improve results.
- I also found out that a batch-size of 128 over a number of 50 epochs performed best on a neural network with 2 hidden layers consisting of 256 and 64 nodes.
- An increase in batch size led to a lower validation loss indicating higher confidence in predictions and lower training time per epoch, but slower convergence.
- In the end, for the 7x7 grid approach and the 1D array approach, there were no differences in the validation losses and accuracies. But, the training time for the 1D array approach was approximately 4x longer than the 7x7 grid approach.

## Learning Outcome

This project gave me hands-on experience with:

- **Supervised Learning**: Training models on labeled data.  
- **Feature Extraction**: Engineering meaningful features for structured data.  
- **Overfitting and Underfitting**: Balancing model complexity for better generalization.  
- **TensorFlow & Keras**: Using pre-built tools to benchmark and validate my models.  
  *(Spoiler: My "built-from-scratch" models were slower but performed similarly in terms of validation losses and accuracies.)*  

## Key Learnings

- **Neural Networks**: Gained a deeper understanding of architectures, backpropagation, training dynamics and solidified my understanding of the math behind everything.  
- **Hyperparameter Tuning**: Understood the impact of parameters like learning rate, batch size, and layer sizes on performance.  
- **Trade-offs**: Learned to balance computational efficiency with model accuracy.

## Challenges Faced

- **Data Preprocessing**: Handling noisy or large datasets posed unexpected challenges.  
- **Model Overfitting**: Required a lot of experimentation with hyper parameters.  
- **Debugging**: Understanding *why* models behaved the way they did was occasionally frustrating but rewarding.
- **Overall**: A lot.

## Conclusion

This project has been an enriching experience, strengthening my knowledge of machine learning and feature engineering. I now feel confident tackling more complex challenges, such as object detection or image segmentation. **The future is lookin goood!**

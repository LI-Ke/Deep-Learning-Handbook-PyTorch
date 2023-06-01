# Deep-Learning-Handbook-PyTorch

This is a handbook of Deep Learning by PyTorch. Most of the algorithms are realized from scratch. Let's start our journey by shallow neural networks. We will gradually dive into Deep Learning together. And we will learn more modern models with complex architecture and powerful funcitons. By the end, I hope
you will be able to solve the real-world problem by using your own Deep Learning model.

## Chapter 1: Linear NN for Regression

Regression problems pop up whenever we want to predict a numerical value. Common examples include predicting prices (of homes, stocks, etc.), predicting the length of stay (for patients in the hospital), forecasting demand (for retail sales), among countless others. Linear regression may be both the simplest and most popular among the standard tools for tackling regression problems. 

You can find linear regression code from [here](./D2L/models/linear_regression) where I provide the linear regression example that I've built from scratch and the concise examples that I've done by using pyTorch.

## Chapter 2: Linear NN for Classification 

Classification is defined as the process of recognition, understanding, and grouping of objects and ideas into preset categories a.k.a “sub-populations.” With the help of these pre-categorized training datasets, classification in machine learning programs leverage a wide range of algorithms to classify future datasets into respective and relevant categories.

The Softmax regression is probably the simplest classification algorithm. It a form of logistic regression that normalizes an input value into a vector of values that follows a probability distribution whose total sums up to 1. The output values are between the range [0,1] which not only allow us to do binary classification, but also allow us to accommodate as many classes or dimensions in our neural network model. This is why softmax is sometimes referred to as a multinomial logistic regression.

The provided Softmax regression examples can be found [here](./D2L/models/softmax_regression) which aim to deal with Multiclass Classification problems.

## Chapter 3: Multilayer Perceptrons

In this chapter, we will introduce your first truly deep network. The simplest deep networks are called multilayer perceptrons, and they consist of multiple layers of neurons each fully connected to those in the layer below (from which they receive input) and those above (which they, in turn, influence).

The MLP examples can be found [here](./D2L/models/mlp) which aim to deal with the same Multiclass Classification problems as aforementioned.

In this chapter, I will also introduce the notion of [Dropout](./D2L/models/dropout), whose idea is to inject noise while computing each internal layer during forward propagation, and it has become a standard regularization technique for training neural networks. 

## Chapter 4: Convolutional Neural Networks

This chapter introduces convolutional neural networks (CNNs), a powerful family of neural networks that are designed for dealing with image data.

You can find some classic CNN models and structures [here](./D2L/models/cnn) which includes AlexNet, VGG, Nin, GoogLeNet, ResNet, etc, where some of them have contributed to the prosperity of modern computer vision technology.

## Chapter 5: Recurrent Neural Networks

Recurrent neural networks (RNNs) are deep learning models that capture the dynamics of sequences via recurrent connections, which can be thought of as cycles in the network of nodes. Recurrent neural networks are unrolled across time steps (or sequence steps), with the same underlying parameters applied at each step. While the standard connections are applied synchronously to propagate each layer’s activations to the subsequent layer at the same time step, the recurrent connections are dynamic, passing information across adjacent time steps.

In this chapter, you will find some classic [RNN models](./D2L/models/rnn), such as LSTM and GRU. And finally you will be able to do the Machine Translation task by using the encoder-decoder architecture which is designed for sequence to sequence learning. 

## Chapter 6: Attention Mechanisms and Transformers

At the present moment, the dominant models for nearly all natural language processing tasks are based on the Transformer architecture. Given any new task in natural language processing, the default first-pass approach is to grab a large Transformer-based pretrained model, (e.g., BERT, ELECTRA, RoBERTa, or Longformer) adapting the output layers as necessary, and fine-tuning the model on the available data for the downstream task.

The core idea behind the Transformer model is the attention mechanism, an innovation that was originally envisioned as an enhancement for encoder-decoder RNNs applied to sequence-to-sequence applications, like machine translations.

The intuition behind attention is that rather than compressing the input, it might be better for the decoder to revisit the input sequence at every step. Moreover, rather than always seeing the same representation of the input, one might imagine that the decoder should selectively focus on particular parts of the input sequence at particular decoding steps.

In this chapter, we introduce [attention models](./D2L/models/attention), starting with the most basic intuitions and the simplest instantiations of the idea. We then work our way up to the Transformer architecture and the vision Transformer.

## Chapter 7: Computer Vision

In this chapter, we will introduce two methods that may improve model generalization, namely image augmentation and fine-tuning, and apply them to image classification. Since deep neural networks can effectively represent images in multiple levels, such layerwise representations have been successfully used in various computer vision tasks such as object detection, semantic segmentation, and style transfer. Following the key idea of leveraging layerwise representations in computer vision, we will begin with major components and techniques for object detection. Next, we will show how to use fully convolutional networks for semantic segmentation of images. Then we will explain how to use style transfer techniques to generate images.

The corresponding code can be found [here](./D2L/models/computer_vision).

## Chapter 8: Natural Language Processing

Natural language processing studies interactions between computers and humans using natural languages. In practice, it is very common to use natural language processing techniques to process and analyze text (human natural language) data, such as language models and machine translation models.

To understand text, we can begin by learning its representations. Leveraging the existing text sequences from large corpora, self-supervised learning has been extensively used to pretrain text representations, such as by predicting some hidden part of the text using some other part of their surrounding text. In this way, models learn through supervision from massive text data without expensive labeling efforts!

As we will see in this chapter, when treating each word or subword as an individual token, the representation of each token can be pretrained using word2vec, GloVe, or subword embedding models on large corpora. After pretraining, representation of each token can be a vector, however, it remains the same no matter what the context is. For instance, the vector representation of “bank” is the same in both “go to the bank to deposit some money” and “go to the bank to sit down”. Thus, many more recent pretraining models adapt representation of the same token to different contexts. Among them is BERT, a much deeper self-supervised model based on the Transformer encoder. In this chapter, we will focus on how to pretrain such representations for text and how to apply those (deep) representation learning of languages to addressing natural language processing problems, such as Sentiment Analysis and Natural Language Inference. 

In the end, we will introduce how to implement a BERT model for a wide range of natural language processing applications, such as on a sequence level (single text classification and text pair classification) and a token level (text tagging and question answering). Then we will fine-tune pre-trained BERT for natural language inference.

You can find all examples [here](./D2L/models/nlp).

## Chapter 9: Generative Adversarial Networks

A generative adversarial network (GAN) is a class of machine learning frameworks and a prominent framework for approaching generative AI.

Given a training set, this technique learns to generate new data with the same statistics as the training set. For example, a GAN trained on photographs can generate new photographs that look at least superficially authentic to human observers, having many realistic characteristics.

The core idea of a GAN is based on the "indirect" training through the discriminator, another neural network that can tell how "realistic" the input seems, which itself is also being updated dynamically. This means that the generator is not trained to minimize the distance to a specific image, but rather to fool the discriminator. This enables the model to learn in an unsupervised manner. 

Two examples are provided [here](./D2L/models/gan)

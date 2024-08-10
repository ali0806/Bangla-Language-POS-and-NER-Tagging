# Project Documents

This project focuses on developing a robust system for Named Entity Recognition (NER) and Parts of Speech (POS) tagging specifically tailored for the Bangla language. The primary objective is to create a unified model capable of accurately identifying and classifying entities (such as names of people, organizations, locations, etc.) and determining the grammatical categories (such as nouns, verbs, adjectives, etc.) of words in Bangla sentences




## Table of Contents

- [POS and NER Tagging](#project-documents)
  - [Table of Contents](#table-of-contents)
  - [Related work](#related-work-and-research-paper-analysis)
  - [Model](#model)
    - [Reason Behid Choosing BLSTM-CNN Architecture]()
  - [Project Pipeline](#project-pipeline)
    - [Training Pipeline](#training-pipeline)
    - [Inference Pipeline](#inference-pipeline)
  - [Pipeline Workflow](#Pipeline-workflow)
  

## Related Work and Research Paper Analysis


I conducted an extensive analysis of various research papers and related work in the field of Natural Language Processing (NLP) for the Bangla language. Through this review, I identified several key findings and insights that contribute to our understanding and development of NLP techniques for Bangla. Below, I present the most significant findings from my analysis from some papers.


- [**BanglaBERT: Language Model Pretraining and Benchmarks for Low-Resource Language Understanding Evaluation in Bangla**](https://arxiv.org/pdf/2101.00204v4)
  - The model was pretrained using a 27.5 GB Bangla text dataset ('Bangla2B+'), compiled by crawling 110 popular Bangla websites. It have 335M parameters and BangLUE Score 78.43
  - BanglaBERT supports a wide range of tasks, including sequence classification (e.g., document classification, sentiment analysis), token classification (e.g., NER, PoS tagging), and question answering (e.g., extractive, open-domain)

- [**Bangla-BERT: Transformer-Based Efficient Model for Transfer Learning and Language Understanding**](https://www.researchgate.net/publication/362574897_Bangla-BERT_Transformer-based_Efficient_Model_for_Transfer_Learning_and_Language_Understanding)
  - Bangla-BERT, a monolingual BERT model tailored for the Bangla language, was pretrained on the largest Bangla-specific dataset (40 GB of text)
  - Bangla-BERT achieved significant improvements in performance, surpassing previous state-of-the-art results by 3.52%, 2.2%, and 5.3% respectively


| Dataset                                      | Model                               | Accuracy | F1 Score |
|----------------------------------------------|-------------------------------------|----------|----------|
| **BanFakeNews [49]**                        | L+POS+E(F)+MP                       |   |    0.9100     |
|                                              | L+POS+E(N)+MP                       |    | 0.9100   |
|                                              | Bangla-BERT                         | 0.9925   | 0.9421   |
| **Data Set For Sentiment Analysis On Bengali News Comments [50]** | LSTM                                | 0.7474   | 0.7929   |
|                                              | Bangla-BERT                         | 0.8417   | 0.8104   |
| **Cross-lingual Sentiment Analysis in Bengali [48]** | TextBlob (Unsupervised approach) | 0.8279   | 0.7760   |
|                                              | SVM (Supervised Approach)           | 0.9300   | 0.9160   |
|                                              | Bangla-BERT                         | 0.9703   | 0.9621   |

- [**End to End Parts of Speech Tagging and Named Entity Recognition in Bangla Language**](https://www.researchgate.net/publication/338567187_End_to_End_Parts_of_Speech_Tagging_and_Named_Entity_Recognition_in_Bangla_Language)
  - The goal of the paper is to develop and evaluate a robust end-to-end deep learning model for automatic Parts of Speech (POS) tagging and Named Entity Recognition (NER) specifically for the Bangla language
  - POS Tagging

      | Model                     | Accuracy |
      |---------------------------|----------|
      | BLSTM-CRF [without ce, without pwe] | 90.04    |
      | BLSTM-CRF [without ec]    | 91.70    |
      | BLSTM-CRF                 | 92.29    |
      | BLSTM-CNN                 | 02.50    |
      | BLSTM-CNN-CRF             | 93.86    |

  - NER Tag F1 Score

    | Model                     | F1 Score |
    |---------------------------|----------|
    | BLSTM-CRF [without ce, without pwe] | 47.23    |
    | BLSTM-CRF [without ce]    | 59.22    |
    | BLSTM-CRF                 | 62.20    |
    | BLSTM-CNN                 | 60.40    |
    | BLSTM-CNN-CRF             | 62.84    |

- [**B-NER: A Novel Bangla Named Entity Recognition Dataset with Largest Entities and Its Baseline Evaluation**](https://www.researchgate.net/publication/370086368_B-NER_A_Novel_Bangla_Named_Entity_Recognition_Dataset_with_Largest_Entities_and_Its_Baseline_Evaluation)
  - The paper introduces B-NER, a comprehensive Bangla NER dataset with 22,144 manually annotated sentences and eight entity types, overcoming the previous limitations of recognizing only three entities. The dataset, validated with a Kappa score of 0.82, demonstrated superior performance in cross-dataset modeling and benchmarking, with fine-tuned IndicBERT achieving a Macro-F1 score of 86%.
  [screenshot]()
- [**Towards POS Tagging Methods for BengaliLanguage: A Comparative Analysis**](https://www.researchgate.net/publication/349152637_Towards_POS_Tagging_Methods_for_Bengali_Language_A_Comparative_Analysis)
  - The paper evaluates 16 POS tagging techniques for Bengali using a corpus of 7,390 sentences, including stochastic methods (e.g., Hidden Markov Model, Conditional Random Field) and transformation-based methods (e.g., Brill’s method). The combination of Brill’s method with Conditional Random Forest (CRF) achieved the highest accuracy, with 91.83% for an 11-tagset and 84.5% for a 30-tagset, demonstrating the effectiveness of this model combination for Bengali POS tagging.
- **Bangla-bert-base model which is a pretrained Bengali language model based on BERT's masked language modeling, using a corpus from Bengali CommonCrawl and Wikipedia**

## Model
For Named Entity Recognition (NER) and Parts of Speech (POS) tagging in the Bangla language, we employed a hybrid architecture combining a Bidirectional LSTM (BLSTM) with a Convolutional Neural Network (CNN). We have used CNN on top character embedding and BLSTM on top of the concatenation of word and character embedding after applying CNN on it.

![Screenshot](/media/model-architecture.png)


- **Word Embeddings**: The model begins with a word embedding layer initialized with pre-trained and non-trainable embeddings.

- **CNN for Feature Extraction** : This layer is crucial for extracting morphological features from sequences of characters, helping the model recognize prefixes, suffixes, and other character-level patterns that are important for POS tagging and NER tasks. NN applies 1D convolution operations across the character embeddings and whic kernel size of 5 and 30 filters.

- **Feature Combination** : The output from the CNN is combined with the word embeddings
- **BLSTM Layer** : The combined feature vectors are fed into a Bidirectional LSTM (BLSTM) layer. The BLSTM consists of 100 hidden units and processes the sequence in both forward and backward directions, capturing the context on both sides of each word. This dual context is essential for understanding the role of a word in a sentence, which is particularly useful in POS tagging and NER

- **Dropout Layer** : To prevent overfitting, a dropout layer with a rate of 0.5 is applied after the BLSTM layer

- **Fully Connected Layers**: The model has two separate fully connected layers, one for POS tagging and another for NER. Each layer takes the output of the BLSTM and maps it to the respective tag space, producing logits for POS and NER tags

## Reason Behind Choosing the BLSTM-CNN Model
After thoroughly analyzing various research papers, we considered three potential approaches for our model selection:
- **Transformer-based LLM Models**: Such as BanglaBERT or bangla-bert-base.
- **LSTM-based or Hybrid Models**: Including LSTM, LSTM-CNN, BLSTM-CNN, and BLSTM-CRF.
- **Stochastic Methods**: Such as Hidden Markov Models (HMM) and Conditional Random Fields (CRF).

### Why BLSTM-CNN?
- **Dataset Size Consideration**: Our dataset is relatively small, which makes Transformer-based models less effective. Large Language Models (LLMs) like BERT typically require vast amounts of data to achieve optimal performance. In contrast, the BLSTM-CNN model is better suited to work effectively with smaller datasets.
- **Performance Comparison**: The performance gap between Transformer-based LLMs and Hybrid models like BLSTM-CNN is relatively narrow. Specifically, the BLSTM-CNN model has demonstrated strong performance in both POS and NER tagging tasks, making it a competitive choice.

- **Computational Efficiency and Inference Speed**: Transformer-based LLMs come with a large number of parameters, leading to high computational costs during training. Additionally, their inference speed tends to be slower. On the other hand, the BLSTM-CNN model is lightweight in terms of parameters, resulting in faster training and inference times. This efficiency makes it a more practical choice for our project, where computational resources and speed are important considerations.

## Project Pipeline
### Training Pipeline
The model training pipeline involves several key steps: data cleaning to remove errors and inconsistencies, data processing to prepare and structure the data, tokenization and embedding to convert text into numerical representations, training the model using the prepared data, and evaluating the model's performance using various metrics.


![training pipeline](/media/training-pipeline.jpg)
      


#### Datasets
We provide a dataset named ```data.tsv```, and the dataset format is as follows:
```bash
<sentence>
<token> <POS-tag> <NER-tag>
<sentence>
<token> <POS-tag> <NER-tag>
```
#### Data Cleaning
In the data.tsv file, tokens are already separated from their sentences and assigned corresponding tags. However, there are inconsistencies in the dataset. For example, tokens like `(২৭` and `(ওসি)` are tagged as PUNCT, while others, such as `আগস্ট),বিশ্ব।,` are not . We have identified these inconsistencies and are checking for null values and other issues.



#### Data Processing, Data Tokenization and Embedding
During the exploratory data analysis (EDA) of the dataset, we observed an imbalance in tag frequencies. Some tags occur with high frequency, while others are relatively infrequent. The frequency distribution of some tags is as follows:

| POS Tag | POS Count | NER Tag | NER Count |
|---------|-----------|---------|-----------|
| NNC     | 30,660    | B-OTH   | 53,550    |
| NNP     | 12,977    | B-PER   | 6,158     |
| ADJ     | 8,076     | B-ORG   | 2,742     |
| PART      | 150     | I-T&T   | 77        |
| INTJ    | 36       | I-MISC  | 65        |
| OTH     | 113       | I-UNIT  | 20        |


In data processing section, involves transforming and structuring the cleaned data to make it suitable for training. We convert the data format as follows:

```bash
{"tokens":[token,token,token...], 'POS_tag':[pos_tag,pos_tag,pos_tag...], 'NER_Tag':[ner_tag,ner_tag,ner_tag...]}
```
For simplicity, we used the BanglaWord2Vec embedding technique.

#### Model Training and Evaluation
We selected a hybrid model architecture that combines Bidirectional Long Short-Term Memory (BLSTM) and Convolutional Neural Networks (CNN). In the research paper [End to End Parts of Speech Tagging and Named Entity Recognition in Bangla Language [2]]() they encourage parameters as follows and it provides good performance.

```bash
LSTM units: 100
Character embedding size: 30
CNN kernel size: 5
CNN filters: 30
Optimizer: Adam
Dropout rate: 0.5
```
After evaluation, we found performance as follows,
| Tag  | Accuracy | Precision | Recall | F1 Score |
|------|----------|-----------|--------|----------|
| POS  | 0.879    | 0.848     | 0.801  | 0.820    |
| NER  | 0.903    | 0.698     | 0.590  | 0.635    |

#### Trained Model Export 
Once the model is trained, it needs to be exported for future use. The export can be done in various formats, including `.pt, .pth, .onnx, & .bin.` Depending on the requirements, we have the flexibility to save either the entire model or just the weights. The choice of format and what to save depends on the specific needs of subsequent operations.

## Inference Pipeline
‍An inference pipeline is a program that takes input data, optionally transforms that data, then makes predictions on that input data using a model.

In our inference pipeline, there are two servers: a web server created with Flask and a model inference API, also known as the ML backend server, created with FastAPI.

### Pipeline Components
#### Model Inference API
The FastAPI service is designed to handle inference requests by processing text inputs and generating predictions for Parts of Speech (POS) and Named Entity Recognition (NER) tags.
- **Model Initialization:** Loads a pre-trained ONNX model and necessary mappings and parameters for POS and NER tasks
- **Preprocessing:** Cleaning,tokenization and converts input text into token indices using a pre-defined embeding.Also add Pads the token sequence to a fixed length to ensure consistency for model inference

- **Prediction**: Performs inference using the ONNX model to predict and Converts model outputs from indices to human-readable tags
- **API End Point** : Provides a POST endpoint (/predict) that accepts a JSON payload with text input, processes it, and returns the predicted POS and NER tags.

#### Web server
The Flask web application offers an interactive user interface that allows users to submit text for POS and NER tagging and view the results.

- **User Inferface**: Renders an HTML form where users can input text for processing
- **Request Handling**: For POST requests, retrieves the input text from the form and forwards it to the FastAPI inference service and receives predictions from the FastAPI service.

![training pipeline](/media/inference-pipeline.jpg)
### Pipeline Workflow
- **User Interaction**: The user accesses the Flask web application and inputs text into the provided form and the Flask app sends a POST request containing the text to the FastAPI inference service.
- **Inference Processing**: The FastAPI service processes the text, performs tokenization, converts tokens into indices, and applies the ONNX model to generate POS and NER predictions
- **Results Return:** The FastAPI service returns the predicted tags to the Flask application.'
- **Results Presentation:** The Flask application receives the results and renders them on a web page, displaying the input text along with the predicted POS and NER tags.

#### Deployment with Docker Compose:
Docker Compose orchestrates the deployment, ensuring that both the FastAPI and Flask services are correctly configured and communicate seamlessly within a containerized environment.
 







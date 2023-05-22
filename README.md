# paraphasing for Hindi and Telugu
## Models used
1. Lstm
1. Encoder-Decoder Lstm
2. Lstm with attention
4. GRU with attention
5. MTBT - LSTM
6. MTBT - GRU Attention

## work flow
* 1st sentences in english language is converted to Telugu and Hindi using google translate Api.
* preprocessing of data and spliting the sentences into input and outputs.
* Traning the models with train data
* saving the model using validation error
* Testing the performance using METEOR and test data
## structure of files
* main - folder where all the .py files are located
* graphs - graphs of some experiments
* report - detailed report on project
* models - drive link in README file

## Loading the model
* models are stored in .pth files,which can we loaded in the files using torch.load().
* input for the model is a sequence of words in hindi or telugu based on the model which we are using and output is the another sentence with has similar meaning to the input.

## METEOR score in order for hindi language: 
* GRU ATTENTION > LSTM ATTENTION > ENCODER-DECODER ATTENTION > LSTM

### MACHINE TRANSLATION for Telugu language:
* Machine Translation
* Back Translation

LINK FOR PRE-TRAINED MODELS:
https://drive.google.com/drive/folders/15-GxkDwWaTxgzGy573OBhU27Nzlyk2Mu?usp=sharing

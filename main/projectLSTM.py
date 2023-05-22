import json
from operator import itemgetter
from itertools import groupby
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
import torch
from tensorflow.keras.preprocessing.sequence import pad_sequences
import torch.nn as nn
import torch.optim as optim

                            # loading the json file of hindi translations
f = open('/content/drive/MyDrive/annotations/final_hindi_data_train.json', 'r')
x = f.read()
x = json.loads(x)

                            # sorting all sentences and grouping sentences with same id

sortkeyfn = itemgetter('image_id')
x['annotations'].sort(key=sortkeyfn)
captions = []
for key,valuesiter in groupby(x['annotations'], key=sortkeyfn):
    captions.append(dict(type=key, items=list(v['caption'] for v in valuesiter)))


                            # preprocessing the sentences using built-in tokenizer and finding the max length of sentence


sentences=[]
for x in range(len(captions)):
  sentence = []
  for y in range(len(captions[x]['items'][0])):
    sentence.append(captions[x]['items'][0][y])
  sentences.append(sentence)
h = np.array(sentences).flatten()
print(len(h),h[1])
tokenizer = Tokenizer()
tokenizer.fit_on_texts(h)
print(tokenizer.texts_to_sequences(sentences[0]))
sequences = []
print(len(sentences))
for i in range(len(sentences)):
  sequences.append(tokenizer.texts_to_sequences(sentences[i]))
max = -1
for i in range(len(sequences)):
  for j in range(len(sequences[i])):
    if max < len(sequences[i][j]):
      max = len(sequences[i][j])

print(max)

                                     # padding the sequences to make them equal length

padded_sequences = []
for i in range(len(sequences)):
  padded_sequences.append(pad_sequences(sequences[i], maxlen=max, padding="post", truncating="post"))

                                    # spliting the data into input and output for model


inputs = []
outputs = []
for i in range(len(padded_sequences)):
  inputs.append(padded_sequences[i][0])
  inputs.append(padded_sequences[i][2])
  outputs.append(padded_sequences[i][1])
  outputs.append(padded_sequences[i][3])
inputs = torch.LongTensor(inputs)
outputs = torch.LongTensor(outputs)

                                    # making input and output into batchs

batch_wiseinput=torch.split(inputs,128)
batch_wiseoutput=torch.split(outputs,128)
batchs=[]
for i in range(0,len(batch_wiseinput)):
  batchs.append((batch_wiseinput[i],batch_wiseoutput[i]))
print(len(batchs))
print(len(batchs[0]))

                                      # lstm model
class LSTM(nn.Module):
  def __init__(self, vocab_size,pos_size):
    super(LSTM,self).__init__()
    self.hidden_size=200
    self.emb_dim=300
    self.embedding = nn.Embedding(vocab_size,self.emb_dim)
    self.lstm = nn.LSTM(input_size=self.emb_dim, hidden_size=self.hidden_size)
    self.answer = nn.Linear(self.hidden_size,pos_size)
    
  def forward(self,context):
    embed_output=self.embedding(context)
    out,state=self.lstm(embed_output)
    final=self.answer(out)
    # print(final)
    return final
  
                                 # training the model
model = LSTM(len(tokenizer.word_index)+1, len(tokenizer.word_index)+1)
import math
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=0.01)
initial_loss=-math.inf
for i in range(5):
  actual=[]
  predict=[]
  model.train()
  loss_for_epoch=0
  print("entered")
  batc=0
  for batch,(words,prediction) in enumerate(batchs):
    optimizer.zero_grad() # set the gradients to zero before starting to do backpropragation 
    predicted_one=model(words)
    predicted_one=predicted_one.view(-1,predicted_one.shape[2])
    prediction=prediction.view(-1)
    print(len(prediction),len(predicted_one))
    loss=criterion(predicted_one,prediction)
    loss.backward()
    optimizer.step()
    loss_for_epoch+=loss.item()
    batc+=1
  loss_for_epoch=loss_for_epoch/batc
  initial_loss=loss_for_epoch
  print(loss_for_epoch)

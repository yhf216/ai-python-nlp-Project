import torch
import torch.nn as nn
import torch.nn.functional as F

class TextClassifier(nn.Module):
    def __init__(self,vocab_size,embedding_dim,hidden_dim,output_dim,bidirectional=True,dropout=0.4):
        super().__init__()
        self.bidirectional=bidirectional
        #嵌入层
        self.embedding=nn.Embedding(vocab_size,embedding_dim)
        #LSTM层
        self.lstm=nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            bidirectional=bidirectional,
            batch_first=True,
        )
        lstm_output_dim=hidden_dim*2 if bidirectional else hidden_dim
        #线性层
        self.fc=nn.Linear(lstm_output_dim,output_dim)
        #随机失活避免过拟合
        self.dropout=nn.Dropout(dropout)

    def forward(self,text):
        embedded=self.dropout(self.embedding(text))
        output,(hidden,cell)=self.lstm(embedded)

        #处理双向LSTM的隐藏状态
        if self.bidirectional:
            hidden = torch.cat([hidden[-2], hidden[-1]], dim=1) 
        else:
            hidden = hidden[-1]  

        hidden=self.dropout(hidden)
        hidden=self.fc(hidden)
        return hidden

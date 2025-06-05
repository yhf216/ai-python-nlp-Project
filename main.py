from torch.utils.data import DataLoader
from torch.optim import Adam
from sklearn.metrics import classification_report

import pandas as pd
import numpy as np
import torch 
import torch.nn as nn
import gradio as gr
import json

from utils import TextDataset,ModelCheckpoint
from model import TextClassifier
from trainer import Trainer
from graph import ROC,loss


def main(
        lr:float=0.002,
        batch_size:int=32,
        bidirectional:bool=True,
        dropout:float=0.4,
        weight_decay: float = 1e-5,
        epochs:int=100,
        embedding_dim:int=100,
        hidden_dim:int=192,
        max_length:int=50,
):
    #加载数据
    df1=pd.read_excel("datasets\\train.xlsx")
    df2=pd.read_excel("datasets\\test.xlsx")

    train_texts=df1['text'].apply(lambda x: list(str(x))).values
    train_labels=df1['label'].map({'positive':0,'neutral':1,'negative':2}).values
    val_texts=df2['text'].apply(lambda x: list(str(x))).values
    val_labels=df2['label'].map({'positive':0,'neutral':1,'negative':2}).values

    #构建词汇表
    vocab={'<padded>':0,'<unknown>':1}
    for text in train_texts:
        for char in text:
            if char not in vocab:
                vocab[char]=len(vocab)
    
    # 保存词汇表
    with open("vocab\\vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False)

    #创建数据集和数据加载器
    train_dataset=TextDataset(train_texts,train_labels,vocab,max_length)
    val_dataset=TextDataset(val_texts,val_labels,vocab,max_length)
    
    train_loader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
    val_loader=DataLoader(val_dataset,batch_size=batch_size)

    
    #模型初始化
    model=TextClassifier(
        vocab_size=len(vocab),
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        output_dim=3,
        bidirectional=bidirectional,
        dropout=dropout,
    )

    model_config = {
        'vocab_size': len(vocab),
        'embedding_dim': embedding_dim,
        'hidden_dim': hidden_dim,
        'output_dim': 3,
        'bidirectional': bidirectional,
        'dropout': dropout,
    }

    device=torch.device('cuda'if torch.cuda.is_available() else 'cpu')

    #选用交叉熵损失函数，Adam优化算法
    criterion=nn.CrossEntropyLoss()
    optimizer=Adam(model.parameters(),lr=lr,weight_decay=weight_decay)
    trainer=Trainer(model,device,criterion, optimizer)

    model_checkpoint=ModelCheckpoint(save_path="models\\best_model.pth",model_config=model_config)

    train_losses=[]
    train_accs=[]
    val_losses=[]
    val_accs=[]

    for epoch in range(epochs):
        print(f"{epoch+1}/{epochs}")
        train_loss,train_acc=trainer.train_epoch(train_loader)
        val_loss,val_acc,_,_=trainer.evaluate(val_loader)
            
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(f"train loss:{train_loss:.4f} train accuracy:{train_acc:.4f}")
        print(f"val loss:{val_loss:.4f} val accuracy:{val_acc:.4f}")

        model_checkpoint(model,val_acc)
    
    #最终评估
    val_loss,val_acc,all_preds,all_labels=trainer.evaluate(val_loader)
    y_pred=np.argmax(all_preds,axis=1)

    #生成报告
    report=classification_report(
        all_labels,y_pred,
        target_names=['positive','neutral','negative'],
        output_dict=True,
    )

    # 将报告转换为DataFrame形式
    report_df=pd.DataFrame(report).transpose()
    report_df=report_df.reset_index().rename(columns={'index': 'metric'})
    report_df=report_df.round(4)

    #生成ROC曲线
    ROC(all_labels=all_labels,all_preds=all_preds)

    #生成损失曲线
    loss(train_losses=train_losses,val_losses=val_losses)
    
    return max(val_accs),report_df,'graphs\\roc.png','graphs\\loss.png'

iface=gr.Interface(
    fn=main,
    inputs=[
        gr.Slider(0.0001,0.01,value=0.002,label="学习率"),
        gr.Dropdown([16,32,64,128,256],value=32,label="批次大小"),
        gr.Dropdown([True,False],value=True,label="双向LSTM"),
        gr.Slider(0,1,value=0.4,step=0.01,label="Dropout"),
        gr.Slider(0,1e-3,value=0,step=1e-9,label="L2正则化系数"),
        gr.Slider(10,2000,value=50,step=10,label="训练轮数"),
        gr.Slider(50,200,value=100,step=50,label="词向量维数"),
        gr.Slider(64,512,value=192,step=64,label="LSTM隐藏层维数"),
        gr.Slider(20,100,value=50,step=10,label="最大文本长度"),
    ],
    outputs=[
        gr.Textbox(label="Best_val_accuracy"),
        gr.Dataframe(label="分类报告",wrap=True),
        gr.Image(label="ROC曲线"),
        gr.Image(label="训练曲线"),
    ]
)

if __name__=="__main__":
    iface.launch()


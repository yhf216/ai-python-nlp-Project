from torch.utils.data import DataLoader
from torch.optim import Adam
from sklearn.metrics import classification_report

import pandas as pd
import numpy as np
import torch 
import torch.nn as nn
import gradio as gr
import json

from utils import TextDataset
from model import TextClassifier
from trainer import Trainer
from graph import ROC,loss


def predict(
        model_path:str,
        vocab_path:str="vocab\\vocab.json",
        max_length:int=50,
        batch_size:int=32,
):
    #加载词汇表
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    
    #加载测试数据
    df = pd.read_excel("datasets\\test.xlsx")
    test_texts = df['text'].apply(lambda x: list(str(x))).values
    test_labels = df['label'].map({'positive': 0, 'neutral': 1, 'negative': 2}).values
    
    # 创建测试数据集和数据加载器
    test_dataset = TextDataset(test_texts, test_labels, vocab, max_length)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    device=torch.device('cuda'if torch.cuda.is_available() else 'cpu')

    #加载模型
    checkpoint=torch.load(model_path)
    model=TextClassifier(**checkpoint['config'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    #创建训练器用于评估
    criterion=nn.CrossEntropyLoss()
    trainer=Trainer(model,device,criterion, None)

    #在测试集上评估
    test_loss,test_acc,all_preds,all_labels=trainer.evaluate(test_loader)
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
    
    return test_acc,report_df,'roc.png',

iface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Textbox(label="模型路径", value="models\\best_model.pth"),
        gr.Textbox(label="词汇表路径", value="vocab\\vocab.json"),
        gr.Slider(20, 100, value=50, step=10, label="最大文本长度"),
        gr.Dropdown([16, 32, 64, 128, 256], value=32, label="批次大小"),
    ],
    outputs=[
        gr.Textbox(label="测试集准确率"),
        gr.Dataframe(label="分类报告", wrap=True),
        gr.Image(label="ROC曲线"),
    ]
)


if __name__=="__main__":
    iface.launch()

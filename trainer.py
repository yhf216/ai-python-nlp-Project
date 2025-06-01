import torch
from tqdm import tqdm

#训练器
class Trainer:
    def __init__(self,model,device,criterion,optimizer):
        self.model=model.to(device)
        self.device=device
        self.criterion=criterion
        self.optimizer=optimizer

    def train_epoch(self,loader):
        self.model.train()
        epoch_loss=0
        epoch_accuracy=0

        #训练进度条
        progress_bar=tqdm(loader,leave=False)
        for batch in progress_bar:
            texts=batch['text'].to(self.device)
            labels=batch['label'].to(self.device)

            #清空梯度
            self.optimizer.zero_grad()

            #预测
            predictions=self.model(texts)

            #计算损失与准确度
            loss=self.criterion(predictions,labels)
            accuracy=self.accuracy(predictions,labels)

            loss.backward()
            self.optimizer.step()
            
            epoch_loss+=loss.item()
            epoch_accuracy+=accuracy.item()

            progress_bar.set_postfix({
                'loss':loss.item(),
                'accuracy':accuracy.item(),
            })

        return epoch_loss/len(loader),epoch_accuracy/len(loader)
    
    def evaluate(self,loader):
        self.model.eval()
        epoch_loss=0
        epoch_accuracy=0
        all_preds=[]
        all_labels=[]

        #禁用梯度计算
        with torch.no_grad():
            for batch in tqdm(loader,leave=False):
                texts=batch['text'].to(self.device)
                labels=batch['label'].to(self.device)

                predictions=self.model(texts)
                loss=self.criterion(predictions,labels)
                accuracy=self.accuracy(predictions,labels)

                epoch_loss+=loss.item()
                epoch_accuracy+=accuracy.item()
                
                #用cpu完成.numpy()方法
                all_preds.append(predictions.cpu())
                all_labels.append(labels.cpu())

        all_preds=torch.cat(all_preds).numpy()
        all_labels=torch.cat(all_labels).numpy()

        return (epoch_loss/len(loader),
                epoch_accuracy/len(loader),
                all_preds,
                all_labels,)
    
    @staticmethod
    def accuracy(preds,y):

        #获取预测概率最大的标签索引
        max_preds=preds.argmax(dim=1,keepdim=True)

        correct=max_preds.squeeze(1).eq(y)
        return correct.sum()/torch.FloatTensor([y.shape[0]]).to(y.device)
    


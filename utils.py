from torch.utils.data import Dataset
import torch

#数据预处理
class TextDataset(Dataset):
    def __init__(self,texts,labels,vocab,max_length):
        super().__init__()
        self.texts=texts
        self.labels=labels
        self.vocab=vocab
        self.max_length=max_length

    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, index):
        text=self.texts[index]
        label=self.labels[index]

        num=[self.vocab.get(word,1) for word in text[:self.max_length]]
        padded=num+[0]*(self.max_length-len(num))

        return{
            'text':torch.tensor(padded,dtype=torch.long),
            'label':torch.tensor(label,dtype=torch.long),
        }
    
#模型保存
class ModelCheckpoint():
    def __init__(self,save_path,model_config):
        self.save_path=save_path
        self.model_config=model_config
        self.best_acc=0
        
    def __call__(self,model,current_acc):
        if current_acc>self.best_acc:
            self.best_acc=current_acc
            torch.save({
                'model_state_dict':model.state_dict(),
                'config':self.model_config},
                self.save_path)

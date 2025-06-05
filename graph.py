from sklearn.metrics import  roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt

#指定中文字体（根据系统安装的字体选择）
plt.rcParams['font.sans-serif']=['WenQuanYi Zen Hei','SimHei','Microsoft YaHei']
#解决负号显示问题
plt.rcParams['axes.unicode_minus']=False

#ROC曲线绘制
def ROC(all_labels,all_preds):
    y_test_bin=label_binarize(all_labels,classes=[0,1,2])
    fpr,tpr,roc_auc={},{},{}
    for i in range(3):
       fpr[i],tpr[i],__=roc_curve(y_test_bin[:,i],all_preds[:,i])
       roc_auc[i]=auc(fpr[i],tpr[i])
    
    plt.figure()
    colors=['blue', 'red', 'green']
    for i,color in zip(range(3),colors):
        plt.plot(fpr[i],tpr[i],color=color,
                 label=f'类别 {i} (AUC = {roc_auc[i]:.2f})')
    plt.plot([0,1], [0,1], 'k--')
    plt.xlabel('假阳性率')
    plt.ylabel('真阳性率')
    plt.title('多分类ROC曲线')
    plt.legend()
    plt.savefig('graphs\\roc.png')

#绘制训练曲线
def loss(train_losses,val_losses):
    plt.figure()
    plt.plot(train_losses, label='训练损失')
    plt.plot(val_losses, label='验证损失')
    plt.xlabel('训练轮数')
    plt.ylabel('损失值')
    plt.title('训练过程损失曲线')
    plt.legend()
    plt.savefig('graphs\\loss.png')
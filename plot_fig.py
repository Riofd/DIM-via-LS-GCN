import os
import numpy as np
import matplotlib.pyplot as plt
import mat4py
# os.chdir('./result/label_smoothing/NEW')
os.chdir('./result/CE/')
x = range(1, 501)  # numpy.linspace(开始，终值(含终值))，个数)
val_loss = mat4py.loadmat('val_loss.mat')
train_loss = mat4py.loadmat('train_loss.mat')
best_loss = mat4py.loadmat('best_loss.mat')
val_acc = mat4py.loadmat('val_acc.mat')
y1 = val_loss['val_loss']
y2 = train_loss['train_loss']
y3 = best_loss['best_loss']
y4 = val_acc['val_acc']
# y4 = (np.array(y4)-0.5).tolist()
# 画图
plt.title('Loss compare')  #标题
# plt.plot(x,y)
# 常见线的属性有：color,label,linewidth,linestyle,marker等
plt.plot(x, y1, color='blue', marker='.', linestyle='-', label='val loss')
plt.plot(x, y2, color='red', marker='.', linestyle='-', label='train loss')
plt.plot(x, y3, color='black', marker='.', linestyle='-', label='best loss')# 'b'指：color='blue'
plt.legend()  # 显示上面的label
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.xlim(0, 500)
# plt.axis([0, 500, 0.3, 1])  # 设置坐标范围axis([xmin,xmax,ymin,ymax])
#  plt.ylim(-1,1)  # 仅设置y轴坐标范围
plt.show()
plt.plot(x, y4, marker='.', linestyle='-', label='val acc')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.xlim(0, 500)
plt.ylim(82, 100)
plt.show()

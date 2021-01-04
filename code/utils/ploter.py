
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
def plot_confusion_matrix(title, y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred)
    # np.newaxis的作用就是在这一位置增加一个一维，这一位置指的是np.newaxis所在的位置，比较抽象，需要配合例子理解。
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    # print (cm, '\n\n', cm_normalized)
    # [[1 0 0 0 0]                            #  [[1. 0. 0. 0. 0.]    
    #  [0 1 0 0 0]                            #  [0 1 0 0 0]                            
    #  [0 0 1 0 0]                            #  [0. 0. 1. 0. 0.]
    #  [0 0 0 1 0]                            #  [0. 0. 0. 1. 0.]
    #  [0 0 0 0 1]]                           #  [0. 0. 0. 0. 1.]]
    tick_marks = np.array(range(len(labels))) + 0.5
    #  [0.5 1.5 2.5 3.5 4.5 5.5]
    np.set_printoptions(precision=2)
    plt.figure(figsize=(10, 8), dpi=120)
    x_list, y_list = np.meshgrid(np.arange(len(labels)), np.arange(len(labels)))
    #  [[0 1 2 3 4 5]
    #  [0 1 2 3 4 5]
    #  [0 1 2 3 4 5]
    #  [0 1 2 3 4 5]
    #  [0 1 2 3 4 5]
    #  [0 1 2 3 4 5]] 

    #  [[0 0 0 0 0 0]
    #  [1 1 1 1 1 1]
    #  [2 2 2 2 2 2]
    #  [3 3 3 3 3 3]
    #  [4 4 4 4 4 4]
    #  [5 5 5 5 5 5]]
    intFlag = 0 # 标记在图片中对文字是整数型还是浮点型
    for x, y in zip(x_list.flatten(), y_list.flatten()):
        # plt.text()函数用于设置文字说明。
        if intFlag:
            c = cm[y][x]
            plt.text(x, y, "%d" % (c,), color='red', fontsize=8, va='center', ha='center')
        else:
            c = cm_normalized[y][x]
            if (c > 0.01):
                plt.text(x, y, "%0.2f" % (c,), color='red', fontsize=7, va='center', ha='center')
            else:
                plt.text(x, y, "%d" % (0,), color='red', fontsize=7, va='center', ha='center')
    if(intFlag):
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.binary)
    else:
        plt.imshow(cm_normalized, interpolation='nearest', cmap=cmap)
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.colorbar()
    plt.xticks(np.array(range(len(labels))), labels, rotation=90)
    plt.yticks(np.array(range(len(labels))), labels)
    plt.title(title)
    plt.ylabel('Index of True Classes')
    plt.xlabel('Index of Predict Classes')
    plt.savefig('confusion_matrix.jpg', dpi=300)
    plt.show()


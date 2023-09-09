import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def print_args(args, file):
    args_str = "************************ args **************************" + '\n'
    for k in args.__dict__:
        args_str += k + ": " + str(args.__dict__[k]) + '\n'
    args_str += "********************************************************" + '\n'
    f = open(file, 'a')
    print(args_str, file=f)
    f.close()
    print(args_str)



# # 绘制loss曲线
def show_log(losses, epochs, path, name):
    losses = np.array(losses)

    fig = plt.figure(figsize=(40, 10))    # 创建一个画布
    ax = fig.add_subplot(1, 1, 1)
    ax.get_yaxis().get_major_formatter().set_scientific(False)
    ax.plot(losses, label=name)
    plt.title(name + '_' + str(epochs))
    ax.legend()    # 添加图例
    plt.savefig(path + '/' +name + '_' + str(epochs)+'.png')
    plt.close()
    # plt.show()



# 将loss写入文件
def save_log(losses, epochs, path, name):
    losses = np.array(losses)

    # 将losses写入文件保存
    np.set_printoptions(suppress=False, precision=5)
    # np.save(path + '/' +name + '_' + str(epochs) + ".npy", losses)
    np.savetxt(path + '/' +name + '_' + str(epochs) + ".txt", losses, fmt='%.05f')



def read_log(filename):
    loss = np.loadtxt(filename, delimiter='\n')
    return loss



def show_log_all(train_loss, val_loss, train_name, val_name, name, epochs, path):
    losses = []
    losses.append(train_loss)
    losses.append(val_loss)
    losses = np.array(losses)

    fig = plt.figure(figsize=(40, 10))  # 创建一个画布
    ax = fig.add_subplot(1, 1, 1)
    ax.get_yaxis().get_major_formatter().set_scientific(False)
    ax.plot(losses[0], label=train_name)
    ax.plot(losses[1], label=val_name)
    plt.title(name + '_' + str(epochs))
    ax.legend()  # 添加图例
    plt.savefig(path + '/' + name + '_' + str(epochs) + '.png')
    plt.close()
    # plt.show()



# 将loss写入文件
def save_log_all(train_loss, val_loss, epochs, path, name):
    losses = []
    losses.append(train_loss)
    losses.append(val_loss)
    losses = np.array(losses)

    # 将losses写入文件保存
    np.set_printoptions(suppress=False, precision=5)
    # np.save(path + '/' +name + '_' + str(epochs) + ".npy", losses)
    np.savetxt(path + '/' + name + '_' + str(epochs) + ".txt", losses.T, fmt='%.05f')



def save_matrix(matrix):
    matrix = np.array(matrix.cpu())

    # 将losses写入文件保存
    np.set_printoptions(suppress=False, precision=5)
    # np.save(path + '/' +name + '_' + str(epochs) + ".npy", losses)
    np.savetxt("W.txt", matrix, fmt='%.05f')




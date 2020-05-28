import matplotlib.pyplot as plt

def Loss(avg_train_loss, avg_valid_loss, FolderName):
    """ visualize the loss as the network trained   """
    fig = plt.figure(figsize=(10,8))
    plt.plot(range(1,len(avg_train_loss)+1),avg_train_loss, label='Training Loss')
    plt.plot(range(1,len(avg_valid_loss)+1),avg_valid_loss,label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.xlim(0, len(avg_train_loss)+1) # consistent scale
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FolderName, 'Regularizedpooling_loss.jpeg'))
    plt.close()
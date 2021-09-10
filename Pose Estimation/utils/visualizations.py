import matplotlib.pyplot as plt

'''
Plot loss curve while training
Params:
    train_history: train loss so far
    valid_history: validation loss so far
    curr_epoch: current epoch
    n_epochs: total epochs
Returns:
'''
def plot_loss_curves(train_history, valid_history, curr_epoch, n_epochs):
    plt.plot(train_history, 'g', label='Training Loss')
    plt.plot(valid_history, 'b', label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    #plt.ylim([0,1.75])
    plt.title(f"Curr epoch: {curr_epoch} / {n_epochs}")
    plt.legend()
    plt.show()

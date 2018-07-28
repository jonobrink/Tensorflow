

def training_graphs(test_accuracy_list, train_accuracy_list, test_loss_list, train_loss_list, iteration_list, zoom = True, target_accuracy = None, separate_zoom = False):
    plt.subplot(2,1,1)
    if not target_accuracy == None:
        plt.plot(iteration_list, [target_accuracy for x in range(len(iteration_list))], c='r', linestyle = 'dotted' )
    plt.plot(iteration_list, test_accuracy_list)
    plt.plot(iteration_list, train_accuracy_list)
    plt.grid(True)
    plt.title('Test/Training Accuracy')
    plt.subplot(2,1,2)
    plt.plot(iteration_list, train_loss_list)
    plt.plot(iteration_list, test_loss_list)
    plt.title('Test/Training Loss')
    plt.grid(True)
    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=1,
        wspace=0.35)
    plt.show()

import matplotlib.pyplot as plt
help(plt.plot)

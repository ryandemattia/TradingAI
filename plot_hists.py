import matplotlib.pyplot as plt
import pandas as pd  

def pull_in_file(f_name):
    return pd.read_csv('./champs_histd3/'+f_name)


if __name__ == '__main__':
    thist = pull_in_file('trade_histthot-checkpoint-33.txt')
    print(list(thist))
    print(thist.head())

    plt.plot(thist['date'], thist['current_balance '])
    plt.show()

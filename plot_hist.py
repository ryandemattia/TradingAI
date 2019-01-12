import matplotlib.pyplot as plt
import pandas as pd  

def pull_in_file(f_name):
    return pd.read_csv('./champs_hist/'+f_name)


if __name__ == '__main__':
    thist = pull_in_file('trade_histperpetual_champion_3.pkl.txt')
    print(list(thist))
    print(thist.head())

    plt.plot(thist['Date'], thist['current_balance '])
    plt.plot(thist['Date'], thist['price'])
plt.show()
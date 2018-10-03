import hist_service as hs



class Cryptolution:

    def __init__(generations): #pass number of generations
        self.histworker = hs.HistWorker()
        self.num_gens = generations
    
    def one_gen(self):
        for x in range(len(self.histworker.currentHists[0])):
            for symbol in self.histworker.currentHists.keys():
                dataIn = self.histworker.currentHists[symbol]
                substrate_in = len(dataIn.columns.values)


    def data_loop(self):
        for i in self.num_gens:
            return i


class CryptoFolio:

    ledger = {}
    start = 0
    def __init__(self, start_amount):
        ledger['BTC'] = start_amount
        start = start_amount

    def buy_coin(self, c_name, amount, price, fee):
        if(amount*price > self.ledger['BTC']):
            return
        else:
            self.ledger['BTC'] -= (amount * price) + fee
            self.ledger[c_name] += amount
            return
    def sell_coin(self, price, amount, c_name):
        self.ledger[c_name] -= amount
        self.ledger['BTC'] += (amount * price) - fee
    
    def get_full_btc_value(self, hs, date):
        
        for c in self.ledger:
            if ledger[c] != 0:
                current_price = hs.currentHists[c][date]['Close']
                ledger['BTC'] += ledger[c] * current_price
                ledger[c] = 0
        return ledger['BTC']

class CryptoEval:

    def __init__(self, start_btc, population):
        self.port = CryptoFolio(start_btc)
        self.start_amnt = start_btc
        self.pop = population

    def evaluate(self):
        self.end_amnt = self.port.get_total_btc
        perf = self.start_amnt - self.end_amnt
        return perf


class EvoNode:
    fitnessScore = 0
    lastScore = 0
    numInNodes = 0
    def __init__(self, parent1, parent2, startAmt, numNodes):
        buyPower = startAmt
        numInNodes = inNodes
        numOutNodes = numOutNodes
        generation = 0 
        fitness = 0
        weights = {}

        def learn(self, data):
            for i in range(0, len(data)):

class Account:
    def __init__(self, balance=10000, trading_fee=0.01):
        self._balance = balance
        self._shares = 0
        self._trading_fee = trading_fee

    def buy(self, amount, price):
        trading_fee = self._trading_fee * amount
        if self._balance >= amount + trading_fee:
            change_in_shares = amount / price
            change_in_balance = -amount

            self._shares += change_in_shares
            self._balance += change_in_balance - trading_fee
        else:
            raise ValueError('Not enough balance')

    def sell(self, amount, price):
        if self._shares * price >= amount:
            change_in_shares = -amount / price
            change_in_balance = amount

            self._balance += change_in_balance - self._trading_fee * amount
            self._shares += change_in_shares
        else:
            raise ValueError('Not enough shares')

    def get_balance(self):
        return self._balance

    def get_shares(self):
        return self._shares

    def get_value(self, price):
        return self._balance + self._shares * price

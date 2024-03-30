import unittest

from Account import Account


class AccountTest(unittest.TestCase):
    def test_buy(self):
        account = Account(balance=10000, trading_fee=0.01)
        account.buy(1000, 10)
        self.assertEqual(account.get_balance(), 9000 - 0.01 * 1000)
        self.assertEqual(account.get_shares(), 1000 / 10)
        self.assertEqual(account.get_value(10), 9000 -
                         0.01 * 1000 + (1000 / 10) * 10)

    def test_sell(self):
        account = Account(balance=10000, trading_fee=0.01)
        account.buy(1000, 10)
        self.assertEqual(account.get_balance(), 9000 - 0.01 * 1000)
        account.sell(500, 10)
        self.assertEqual(account.get_balance(),
                         9000 - 0.01 * 1000 +
                         500 - 0.01 * 500)
        self.assertEqual(account.get_shares(), 500 / 10)
        self.assertEqual(account.get_value(10),
                         9000 - 0.01 * 1000 +
                         500 - 0.01 * 500 +
                         (500 / 10) * 10)
        self.assertEqual(account.get_value(11),
                         9000 - 0.01 * 1000 +
                         500 - 0.01 * 500 +
                         (500 / 10) * 11)

    def test_buy_not_enough_balance(self):
        account = Account(balance=10000, trading_fee=0.01)
        with self.assertRaises(ValueError):
            account.buy(10001, 10)

    def test_sell_not_enough_shares(self):
        account = Account(balance=10000, trading_fee=0.01)
        account.buy(1000, 10)
        with self.assertRaises(ValueError):
            account.sell(1001, 10)


unittest.main()

prices = [7,1,5,3,6,4]


def max_profit(prices):
    if not prices:
        return 0
    
    min_price = prices[0]  # track the minimum price so far
    max_profit = 0         # track the maximum profit
    
    for price in prices[1:]:
        # calculate profit if sold today
        profit = price - min_price
        # update max profit
        max_profit = max(max_profit, profit)
        # update min price so far
        min_price = min(min_price, price)
        
    return max_profit



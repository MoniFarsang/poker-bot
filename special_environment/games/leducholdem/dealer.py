from rlcard.core import Card
from rlcard.games.limitholdem import Dealer

class LeducholdemDealer(Dealer):

    def __init__(self, np_random):
        ''' Initialize a leducholdem dealer class
        '''
        self.np_random = np_random
        self.deck = [Card('S', 'J'), Card('H', 'J'), Card('S', 'Q'), Card('H', 'Q'), Card('S', 'K'), Card('H', 'K'), Card('S', 'A'), Card('H', 'A')]
        self.shuffle()
        self.pot = 0

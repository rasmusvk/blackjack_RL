import numpy as np
import pickle
from matplotlib import pyplot as plt
import random
from datetime import datetime
import scipy.stats as stats


class BlackJackSolution:
    def __init__(self, learning_rate=0.05, exploration_rate=1.0, verbose=True, num_decks=6,  exploration_rate_end=0.3):

        self.learning_rate = learning_rate
        self.exploration_rate = exploration_rate

        self.actions = [1, 0]  # 1: HIT  0: STAND
        self.player_state_action = []
        self.end = False

        # initial state (player value, show card, usable ace, count)
        self.state = (0, 0, False, 0)

        self.verbose = verbose
        self.exploration_rate_start = exploration_rate
        self.exploration_rate_end = exploration_rate_end
        self.decay_rate = exploration_rate/exploration_rate_end
        self.exponential_values = self.exponential_decay(
            exploration_rate, exploration_rate_end)

        self.num_decks = num_decks
        self.count = 0
        self.cards_left = [1, 2, 3, 4, 5, 6, 7,
                           8, 9, 10, 10, 10, 10]*4*num_decks
        self.range_count = self.calc_range_count()
        self.countlist = []

        # key: [(player_value, show_card, usable_ace, count)][action] = value
        self.player_Q_Values = {}
        # initialise Q values
        for i in range(12, 22):
            for j in range(1, 11):
                for k in [True, False]:
                    for l in range(self.range_count[0], self.range_count[1]+1):
                        self.player_Q_Values[(i, j, k, l)] = {}
                        for a in [1, 0]:  # initialize actions
                            self.player_Q_Values[(i, j, k, l)][a] = 0

    # give card

    def give_card(self):
        # 1 stands for ace, return a random card from the cards left

        # if there are no cards left, refill it and set the count to zero
        if self.cards_left == []:
            self.cards_left = [1, 2, 3, 4, 5, 6,
                               7, 8, 9, 10, 10, 10, 10]*4*self.num_decks
            self.count = 0
        card = random.choice(self.cards_left)
        self.cards_left.remove(card)
        self.add_count(card)
        return card

    # this calculates the range of the count
    def calc_range_count(self):
        max = 4*self.num_decks*(len([2, 3, 7])*1+len([4, 5, 6])*2)
        min = -4*self.num_decks*(len([9])*1+len([10, 10, 10, 10])*2)
        return [min, max]

    def add_count(self, card):

        self.countlist.append(self.count)
        if card in [2, 3, 7]:
            self.count += 1
        elif card in [4, 5, 6]:
            self.count += 2
        elif card in [9]:
            self.count += -1
        elif card in [10]:
            self.count += -2
        # print(self.count)

        return

    # returns [value, usable ace, *show card]

    def dealer_policy(self, current_value, usable_ace, is_end):
        if current_value > 21:  # if dealer is bust
            if usable_ace:
                current_value -= 10
                usable_ace = False
            else:
                return current_value, usable_ace, True
        # HIT17
        if current_value >= 17:  # if 17 or over, stay
            return current_value, usable_ace, True
        else:
            card = self.give_card()
            if card == 1:
                if current_value <= 10:
                    return current_value + 11, True, False
                return current_value + 1, usable_ace, False
            else:
                return current_value + card, usable_ace, False

    def choose_action(self):
        # if current value <= 11, always hit
        current_value = self.state[0]
        if current_value <= 11:
            return 1

        if np.random.uniform(0, 1) <= self.exploration_rate:
            action = np.random.choice(self.actions)
        #             print("random action", action)
        else:
            # greedy action
            v = -999
            action = 0
            for a in self.player_Q_Values[self.state]:
                if self.player_Q_Values[self.state][a] > v:
                    action = a  # 0 or 1
                    v = self.player_Q_Values[self.state][a]
        #             print("greedy action", action)
        return action

    # one can only have 1 usable ace
    # return next state
    def next_state_player(self, action):
        current_value = self.state[0]
        show_card = self.state[1]
        usable_ace = self.state[2]
        count = self.state[3]
        if action:
            # action hit
            card = self.give_card()
            if card == 1:
                if current_value <= 10:
                    current_value += 11
                    usable_ace = True
                else:
                    current_value += 1
            else:
                current_value += card
        else:
            # action stand
            self.end = True
            return (current_value, show_card, usable_ace, count)

        if current_value > 21:
            if usable_ace:
                current_value -= 10
                usable_ace = False
            else:
                self.end = True
                return (current_value, show_card, usable_ace, count)

        return (current_value, show_card, usable_ace, count)

    def reward(self, player_value, dealer_value):  # this gives the reward of the game
        # player 1 | draw 0 | dealer -1
        reward = 0
        if player_value > 21:
            if dealer_value > 21:
                # draw
                reward = 0
            else:
                reward = -1
        else:
            if dealer_value > 21:
                reward = 1
            else:
                if player_value < dealer_value:
                    reward = -1
                elif player_value > dealer_value:
                    reward = 1
                else:
                    # draw
                    reward = 0
        return reward

    def update_q_value(self, player_value, dealer_value):
        reward = self.reward(player_value, dealer_value)
        #episode_reward = reward
        # backpropagate reward
        # print(reward)
        for s in reversed(self.player_state_action):
            state, action = s[0], s[1]
            reward = self.player_Q_Values[state][action] + \
                self.learning_rate * \
                (reward - self.player_Q_Values[state][action])
            self.player_Q_Values[state][action] = round(reward, 4)

    def reset(self):
        self.player_state_action = []
        self.state = (0, 0, False)  # initial state
        self.end = False

    def deal_two_cards(self, show=False):
        # return value after 2 cards and usable ace
        value, usable_ace = 0, False
        cards = [self.give_card(), self.give_card()]
        if 1 in cards:
            value = sum(cards) + 10
            usable_ace = True
        else:
            value = sum(cards)
            usable_ace = False

        if show:
            return value, usable_ace, cards[0]
        else:
            return value, usable_ace

    def play(self, rounds=10000):
        ii = 0
        for i in range(rounds):
            if i % int(rounds/100) == 0:
                print("round", i)
                self.exploration_rate = self.exponential_values[ii]
                ii += 1
                print(self.exploration_rate)

            count_before_round = self.count
            # give 2 cards
            dealer_value, d_usable_ace, show_card = self.deal_two_cards(
                show=True)
            player_value, p_usable_ace = self.deal_two_cards(show=False)

            self.state = (player_value, show_card,
                          p_usable_ace, count_before_round)

            # judge reward after 2 cards
            if player_value == 21 or dealer_value == 21:
                # game end
                next
            else:
                while True:
                    action = self.choose_action()  # state -> action
                    if self.state[0] >= 12:
                        state_action_pair = [self.state, action]
                        self.player_state_action.append(state_action_pair)
                    # update next state
                    self.state = self.next_state_player(action)
                    if self.end:
                        break

                        # dealer's turn
                is_end = False
                while not is_end:
                    dealer_value, d_usable_ace, is_end = self.dealer_policy(
                        dealer_value, d_usable_ace, is_end)

                # judge reward
                # give reward and update Q value
                player_value = self.state[0]
                if self.verbose:
                    print("player value {} | dealer value {}".format(
                        player_value, dealer_value))
                self.update_q_value(player_value, dealer_value)

            self.reset()

    def save_policy(self, file="policy"):
        fw = open(file, 'wb')
        pickle.dump(self.player_Q_Values, fw)
        fw.close()

    def load_policy(self, file="policy"):
        fr = open(file, 'rb')
        self.player_Q_Values = pickle.load(fr)
        fr.close()

    # trained robot play against dealer
    def evaluation(self, rounds=10000, filename="policy"):
        self.reset()
        self.load_policy(f"policy_{filename}")
        self.exploration_rate = 0

        result = np.zeros(3)  # player [win, draw, lose]
        for _ in range(rounds):
            # hit 2 cards each
            # give 2 cards
            count_before_round = self.count
            dealer_value, d_usable_ace, show_card = self.deal_two_cards(
                show=True)
            player_value, p_usable_ace = self.deal_two_cards(show=False)

            self.state = (player_value, show_card,
                          p_usable_ace, count_before_round)

            # judge reward after 2 cards
            if player_value == 21 or dealer_value == 21:
                if player_value == dealer_value:
                    result[1] += 1
                elif player_value > dealer_value:
                    result[0] += 1
                else:
                    result[2] += 1
            else:
                # player's turn
                while True:
                    action = self.choose_action()
                    # update next state
                    self.state = self.next_state_player(action)
                    if self.end:
                        break

                        # dealer's turn
                is_end = False
                while not is_end:
                    dealer_value, d_usable_ace, is_end = self.dealer_policy(
                        dealer_value, d_usable_ace, is_end)

                # judge
                player_value = self.state[0]
                # print("player value {} | dealer value {}".format(player_value, dealer_value))
                w = self.reward(player_value, dealer_value)
                if w == 1:
                    result[0] += 1
                elif w == 0:
                    result[1] += 1
                else:
                    result[2] += 1
            self.reset()
        return result

    def decay_exp(self, part):
        self.exploration_rate = self.exploration_rate_start - \
            (self.exploration_rate_start-self.exploration_rate_end)*(part)
        if self.verbose:
            print(self.exploration_rate)

    def exponential_decay(self, start_value, end_value, num_steps=100):
        # Generate an array of values from 0 to 1
        x = np.linspace(0, 1, num_steps)
        # Calculate the decay factor
        decay_factor = -np.log(end_value / start_value)

        # Calculate the exponentially decaying values
        y = start_value * np.exp(-decay_factor * x)

        return y


def plotter(data, count):
    #data = b.player_Q_Values
    player_values = np.unique([key[0] for key in data.keys()])
    show_cards = np.unique([key[1] for key in data.keys()])
    usable_aces = np.unique([key[2] for key in data.keys()])
    X, Y = np.meshgrid(player_values, show_cards)
    fig, axs = plt.subplots(len(usable_aces), 1, figsize=(
        6, 3 * len(usable_aces)), sharex=True, sharey=True)

    for i, ace in enumerate(usable_aces):
        ax = axs[i]
        ax.set_title("Usable Ace: {}".format(ace))
        Z = np.zeros_like(X, dtype=float)

        for row in range(X.shape[0]):
            for col in range(X.shape[1]):
                player_value = X[row, col]
                show_card = Y[row, col]
                values = data.get((player_value, show_card, ace, count))
                #print(player_value, show_card, values.get(1))
                if values:
                    # print(max(values.values()))
                    #max_value = max(values.values())
                    #max_key = max(values, key=values.get)
                    #Z[col][row] = abs(abs(values.get(1))-abs(values.get(0)))
                    if values.get(1) > values.get(0):
                        Z[col][row] = 1
                    else:
                        Z[row][col] = 0
                # print(Z)
                #Z[row, col] = max(values.values())
        # print(Z)
        im = ax.imshow(Z, cmap="jet", vmin=0, vmax=1)
        ax.set_xticks(np.arange(len(show_cards)))
        ax.set_yticks(np.arange(len(player_values)))
        ax.set_xticklabels(show_cards)
        ax.set_yticklabels(player_values)
        ax.set_xlabel("Show Card")
        ax.set_ylabel("Player Value")
        ax.grid(False)

    plt.tight_layout()
    cbar = fig.colorbar(im, ax=axs)
    cbar.set_label("Value")

    plt.savefig(f"final__{desired_count}.png")


def obtain_plotting_data(desired_count, Q_tensor):
    # Create empty lists to store the filtered keys and values
    filtered_data = {}

    # Iterate over the dictionary entries
    for key, value in Q_tensor.items():
        _, _, _, count = key  # Extract the count from the key
        if count == desired_count:
            filtered_data[key] = value

    # print(filtered_data)
    return filtered_data


def plot_gaussian(listan, filename):
    mu = np.mean(listan)
    sigma = np.std(listan)
    x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
    plt.plot(x, stats.norm.pdf(x, mu, sigma))
    plt.savefig(f"normaldist_{filename}.png")
    print(round(mu, 3), round(sigma, 3))


def count_ratio(data, range):
    plt.hist(data, bins=20)
    plt.xlabel('Count')
    plt.ylabel('Frequency')
    plt.title('Histogram of Count Frequency')
    plt.xlim(range[0], range[1])
    print(min(data), max(data), np.mean(data))
    # Display the plot
    plt.savefig("histogram_count")


if __name__ == "__main__":
    # training (make sure you change the filename if you want to run the training)
    filename = "main_long_training_n=6lr01"
    train = False
    O = BlackJackSolution(
        verbose=False, learning_rate=0.01, exploration_rate=1, exploration_rate_end=0.3, num_decks=1)
    if train:

        O.play(1000000)
        print("Done training")
        #count_ratio(b.countlist, b.range_count)
        # save policy
        #current_datetime = datetime.now()
        #formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H:%M")
        # f"policy_omega{formatted_datetime}"
        O.save_policy(f"policy_{filename}")
    if not train:
        O.load_policy("policy_main_long_training_n=6lr01")

    plot = False
    if plot:
        desired_count = 4
        data = obtain_plotting_data(desired_count, O.player_Q_Values)
        plotter(data, desired_count)
    # print(b.player_Q_Values)
    listan = []
    play = True
    if play:
        for i in range(1, 100):
            # play
            result = O.evaluation(rounds=10000, filename=filename)
            # print(result)
            listan.append(result[0]/100)

        #plot_gaussian(listan, filename)
        print(round(sum(listan)/len(listan), 3))

import pandas as pd
import matplotlib.pyplot as plt
import re
import pickle


def print_df():
    """Print the full dataset"""
    print(df)


def rarity_cnt():
    """Returns how many drops of each tier"""
    rarity_df = df.groupby('Rarity').count()
    rarity_df = rarity_df.reindex(rarity_order)
    rarity_df.rename(columns={'Reward': 'Count'}, inplace=True)
    return rarity_df


def rarity_bar_plot():
    """Bar plot for rarity_cnt()"""
    rarity_df = rarity_cnt()
    ax = rarity_df.plot.bar(title=f"Rarity bar plot of {df.shape[0]} drops")
    ax.set_xticklabels(rarity_df.index, rotation=45, ha='right')
    for i, v in enumerate(rarity_df['Count']):
        ax.text(i, v, str(v), ha='center', va='bottom')
    plt.tight_layout()


def rarity_pie_plot():
    """Pie plot for rarity_cnt()"""
    rarity_df = rarity_cnt()
    ax = rarity_df.plot.pie(y='Count', legend=False, autopct='%1.1f%%', figsize=(10, 10), startangle=45,
                            title=f"Rarity pie chart of {df.shape[0]} drops",
                            colors=["limegreen", "dodgerblue", "mediumorchid", "red", "yellow"])
    ax.yaxis.set_visible(False)


def rarity_drops(rarity):
    """Only the drops for a certain tier: Rare, Super Rare, Epic, Mythic, Legendary"""
    return df[df['Rarity'] == rarity]


def all_rarity_drops():
    for rarity in rarity_order:
        print(rarity_drops(rarity))


def convert_reward_to_color(rewards_list):
    """Helper function to convert reward to color"""
    convert = {
        'Coins': 'gold',
        'Power Points': 'fuchsia',
        'Credits': 'deepskyblue',
        'Token Doublers': 'cyan',
        'Bling': 'mistyrose',
        'Brawlers': 'lavender',
        'Gadgets': 'lime',
        'Star Powers': 'yellow',
        'Sprays': 'lightcyan',
        'Pins': 'khaki',
        'Icons': 'wheat',
        'Skins': 'peachpuff'
    }
    colors = [convert[reward] for reward in rewards_list]
    return colors


def rewards_cnt(df=None):
    """How many times each reward got dropped"""
    if df is None:
        df = globals().get('df')
    drops_cnt = []
    for ind, reward in enumerate(reward_order):
        if ind <= 4:
            drops_cnt.append(len(reward_total(reward, df)))
        else:
            drops_cnt.append(len(reward_list(reward, df)))
    rewards_df = pd.DataFrame({'Reward': reward_order, 'Count': drops_cnt})
    rewards_df['Reward'] = reward_format
    rewards_df = rewards_df.set_index('Reward')
    return rewards_df


def rewards_bar_plot():
    """Bar plot for rewards_cnt()"""
    rewards_df = rewards_cnt()
    rewards_df.sort_values('Count', ascending=False, inplace=True)
    ax = rewards_df.plot.bar(title=f"Reward bar plot of {df.shape[0]} drops")
    ax.set_xticklabels(rewards_df.index, rotation=45, ha='right')
    for i, v in enumerate(rewards_df['Count']):
        ax.text(i, v, str(v), ha='center', va='bottom')
    plt.tight_layout()


def rewards_pie_plot(df=None, rarity="", size=(10, 10), ax=None):
    """Pie plot for rewards_cnt()"""
    if df is None:
        df = globals().get('df')
    rewards_df = rewards_cnt(df)
    rewards_df.sort_values('Count', ascending=False, inplace=True)
    colors = convert_reward_to_color(rewards_df.index.tolist())
    title_string = f"Reward pie chart of {df.shape[0]} drops"
    if rarity != "":
        title_string += f" ({rarity})"
    print(rewards_df)
    if ax is None:
        ax = rewards_df.plot.pie(y='Count', legend=False, autopct=lambda p: '{:.1f}%'.format(p) if p > 0 else '',
                                title=title_string,
                                startangle=45, colors=colors,
                                figsize=size)
        ax.yaxis.set_visible(False)
    else:
        labels = [label if count > 0 else '' for label, count in zip(rewards_df.index, rewards_df['Count'])]
        ax.pie(rewards_df['Count'], labels=labels, autopct=lambda p: '{:.1f}%'.format(p) if p > 0 else '',
                            startangle=45, colors=colors)
        ax.set_title(title_string)


def rewards_pie_plot_by_rarity(rarity, size=(10, 10), ax=None):
    """Pie plot for rewards_cnt for only a certain rarity"""
    df = rarity_drops(rarity)
    rewards_pie_plot(df=df, rarity=rarity, size=size, ax=ax)


def all_rewards_pie_plot_by_rarity(size=(10, 10)):
    """Calls rewards_pie_plot_by_rarity for every single rarity and plots in one figure"""
    rows, cols = (2, 3)
    fig, axs = plt.subplots(2, 3, figsize=(size[0]*2, size[1]*3))
    index_row, index_col = (0, 0)
    for rarity in rarity_order:
        rewards_pie_plot_by_rarity(rarity, size, axs[index_row, index_col])
        index_col += 1
        if index_col == cols:
            index_col = 0
            index_row += 1
    axs[1, 2].set_visible(False)


def reward_drops(reward):
    """Only the drops for a certain reward:
    pp, sp, pin, bling, skin, coins, credits, td, brawler, gadget, spray, icon"""
    return df[df['Reward'].str.contains(reward)]


def reward_list(reward, df=None):
    """Returns a list containing all the entries containing said reward:
    pp, sp, pin, bling, skin, coins, credits, td, brawler, gadget, spray, icon"""
    if df is None:
        df = globals().get('df')
    pattern = r"\b" + re.escape(reward) + r"\b"
    return df.Reward[df['Reward'].str.contains(pattern, regex=True)].values.tolist()


def reward_total(reward, df=None):
    """Returns an INT list of a certain reward: pp, bling, coins, credits, td"""
    if df is None:
        df = globals().get('df')
    filtered_df = df[df['Reward'].str.contains(' ' + reward)]
    pattern = rf'(\d+) {reward}'
    numbers = filtered_df['Reward'].str.extract(pattern, expand=False).astype(int)
    return numbers


def reward_total_by_rarity(rarity, reward):
    """Returns an INT list of a certain reward for a certain rarity: pp, bling, coins, credits, td"""
    rarity_df = df[df['Rarity'] == rarity]
    filtered_df = rarity_df[rarity_df['Reward'].str.contains(' ' + reward)]
    pattern = rf'(\d+) {reward}'
    numbers = filtered_df['Reward'].str.extract(pattern, expand=False).astype(int)
    return numbers


def rarity_range_table():
    """Makes a 2D array with rarity and rewards numbers"""
    table = []
    ls = []
    for reward in reward_order[:5]:
        for rarity in rarity_order:
            ls_reward = reward_total_by_rarity(rarity, reward)
            ls.append((ls_reward.min(), ls_reward.max()))
        table.append(ls)
        ls = []
    return table


def summary():
    """An overview of all the rewards from all drops"""
    check_cnt = 0
    print("Summary of", df.shape[0], 'drops:')
    for ind, reward in enumerate(reward_order):
        if ind <= 4:
            numbers = reward_total(reward)
            print(str(numbers.sum()) + ' ' + reward_format[ind] + ' (' + str(len(numbers)) + ' drop', end='')
            if len(numbers) != 1:
                print('s', end='')
            print(')')
            check_cnt += len(numbers)
        else:
            lst = reward_list(reward)
            print(f"{reward_format[ind]} ({len(lst)}):")
            for elem in lst:
                print('-', elem)
            check_cnt += len(lst)
    if check_cnt != df.shape[0]:
        logs.append(f"Error when checking dataframe: expected {df.shape[0]}, counted {check_cnt}")
    elif check_cnt != global_row_cnt:
        logs.append(f"Error when checking file rows: expected {global_row_cnt}, counted {check_cnt}")
    else:
        logs.append("No error when checking")


def rarity_range_summary():
    """Returns the range of the reward of each rarity: pp, bling, coins, credits, td"""
    print("Rarity reward ranges:")
    table = rarity_range_table()
    for i in range(len(rarity_order)):
        print(f"{rarity_order[i]}:")
        for j in range(len(reward_order[:5])):
            print(f"• {reward_format[j]}: {table[j][i][0]} - {table[j][i][1]}")


def check_valid_reward(reward):
    """Checks if reward is valid"""
    valid_reward = False
    for reward_ in reward_order:
        reward_ = ' ' + reward_
        if reward_ in reward:
            valid_reward = True
    return valid_reward


def check_valid_rarity(rarity):
    """Checks if rarity is valid"""
    valid_rarity = False
    for rarity_ in rarity_order:
        if rarity_ in rarity:
            valid_rarity = True
    return valid_rarity


def save_table():
    """Updates the rarity range table"""
    with open("rarity_reward_range.pkl", "wb") as pickle_file:
        pickle.dump(rarity_range_table(), pickle_file)


def display_logs():
    """Displays all the errors and comments that gathered during execution"""
    for log in logs:
        print(log)


def split_dash(line):
    """Processes the input if it has a dash"""
    global global_row_cnt
    rarity, reward = line.split(" - ")
    rarity, reward = str(rarity).lower().title(), str(reward).lower()
    valid_rarity, valid_reward = check_valid_rarity(rarity), check_valid_reward(reward)
    if not valid_rarity:
        logs.append(f"Invalid rarity on line {global_row_cnt}:\n{line}")
    if not valid_reward:
        logs.append(f"Invalid reward on line {global_row_cnt}:\n{line}")
    else:
        rarities.append(rarity)
        rewards.append(reward)
        global_row_cnt += 1


def split_no_dash(line):
    """Processes the input if it doesn't have a dash and assigns a rarity"""
    global global_row_cnt
    reward = str(line).lower()
    index_reward = -1
    for ind, reward_ in enumerate(reward_order):
        reward_ = ' ' + reward_
        if reward_ in reward:
            index_reward = ind
    if index_reward == -1:
        logs.append(f"Invalid reward on line {global_row_cnt}:\n{line}")
        return line
    else:
        with open("rarity_reward_range.pkl", "rb") as pickle_file:
            table = pickle.load(pickle_file)
        ls = table[index_reward]
        num, _ = reward.split(" ")
        num = int(num)
        index_rarity = -1
        for i in range(len(ls)):
            if pd.isna(ls[i][0]):
                continue
            if ls[i][0] <= num <= ls[i][1]:
                index_rarity = i
        if index_rarity == -1:
            logs.append(f"Unable to identify rarity on line {global_row_cnt}:\n{line}")
            return line
        else:
            rarity = rarity_order[index_rarity]
            logs.append(f"Successfully assigned rarity {rarity} to {reward} on line {global_row_cnt}")
            rarities.append(rarity)
            rewards.append(reward)
            global_row_cnt += 1
            return f"{rarity} - {reward}"


def read_file(file_path):
    """Reads the rewards from text files and makes a dataframe and autofill missing rarities"""
    with open(file_path, 'r') as file:
        lines = [line.strip() for line in file.readlines() if line.strip()]
    lines_corrected = []
    for line in lines:
        # print(line)
        if " - " in line:
            split_dash(line)
        else:
            line = split_no_dash(line)
        lines_corrected.append(line)
    with open(file_path, 'w') as file:
        for line in lines_corrected:
            file.write(line + '\n')


rarities = []
rewards = []
logs = []
rarity_order = ['Rare', 'Super Rare', 'Epic', 'Mythic', 'Legendary']
reward_order = ['coins', 'pp', 'credits', 'td', 'bling', 'brawler', 'gadget', 'sp', 'spray', 'pin', 'icon', 'skin']
reward_format = ['Coins', 'Power Points', 'Credits', 'Token Doublers', 'Bling', 'Brawlers',
                                    'Gadgets', 'Star Powers', 'Sprays', 'Pins', 'Icons', 'Skins']

global_row_cnt = 0


read_file('sandry_data')
read_file('beshan_data')
read_file('luza_data')
# read_file('temp_data')

# print(rarities, rewards, sep='\n')
df = pd.DataFrame({'Rarity': rarities, 'Reward': rewards})
# Rows with NaN rewards
nan_df = df[df['Reward'].str.contains('NaN')]

# Commands:
# print_df()
# print(rarity_cnt())
# rarity_bar_plot()
# rewards_bar_plot()
# rarity_pie_plot()
# rewards_pie_plot()
# rewards_pie_plot_by_rarity('Mythic')
all_rewards_pie_plot_by_rarity(size=(10, 10))
# print(rarity_drops('Mythic'))
# all_rarity_drops()
# print(reward_drops('pin'))
# print(reward_list('td'))
# print(reward_total('td'))
summary()
# rarity_range_summary()
# print(rarity_range_table())
plt.show()
save_table()
display_logs()

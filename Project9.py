# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 17:22:20 2022

@author: anato
"""

#librairies
import pandas as pd
import re
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

#import the data
data = pd.read_csv(r"C:\Users\anato\Documents\IRONHACK\games_wgm.csv")

openings = pd.read_csv(r"C:\Users\anato\Documents\IRONHACK\IronFrandre\Project 9 Chess games of woman grandmasters\openings.csv")

#Extract the code for the opening from the text of the game in column pgn
def open_code(s) :
    if type(s) != str :
        return ""
    else :
        matches = re.findall("ECO \"[A-Z]\d\d\"", s)
        if matches == [] :
            return ""
        else :
            return matches[0][5:8]

data['opening_code'] = data["pgn"].apply(open_code)

#calculate the score of the white player from his result
def white_points(s) :
    if s == 'win' :
        return 1
    elif s in ['repetition', 'insufficient', 'stalemate', 'agreed', 'timevsinsufficient', '50move'] :
        return 0.5
    elif s in ['timeout', 'checkmated', 'resigned', 'abandoned']:
        return 0
    else :
        return -1
    
data['white_points'] = data['white_result'].apply(white_points)

#Grouping the games by opening, naming the openings and sort the openings by average score of the white player
opening_rate = data.groupby(by = 'opening_code', as_index = False).agg({'white_points' : 'mean', 'game_url' : 'count'}).sort_values(by = 'white_points')

def opening_name(code) :
    return openings.loc[openings['ECO'] == code, 'Opening Names'].min()

opening_rate['name'] = opening_rate['opening_code'].apply(opening_name)

#Extract only the moves from the text in png and fix the errors that can be fixed
def game_str(s) :
    last = re.split(r'\n', s)[-1]
    return re.sub(r'{\[[a-zA-Z0-9 %:\.]*\]} ', '', last)

def moves_ECO(ECO) :
    moves = [j for i in re.split(r'1 | \d | \d+$', openings.loc[openings['ECO'] == ECO, 'Moves'].min()) for j in re.split(' ', i) if j != '']
    numbers = ['1. ', ' 1... ', ' 2. ', ' 2...', ' 3. ', ' 3... ', ' 4. ', ' 4... ', ' 5. ', ' 5... ', ' 6. ', ' 6... ', ' 7. ', ' 7... ', ' 8. ', ' 8... ', ' 9. ', ' 9... ', ' 10. ', ' 10... ', ' 11. ', ' 11... ', ' 12. ', ' 12... ', ' 13. ', ' 13... ', ' 14. ']
    return [j for i in range(len(moves)) for j in [numbers[i], moves[i]]] + [numbers[len(moves)]]

def full_game(s, ECO) :
    if s[0:3] in ['1. ', '1-0', '0-1'] :
        return s
    else :
        moves_start = []
        stop = re.match(r'\d+\.\.?\.? ', s)
        if stop == None :
            return ''
        else :
            stop = stop.group(0)
        for item in moves_ECO(ECO) :
            if item == ' ' + stop :
                break
            moves_start.append(item)
        else :
            return 'e1'
        return ''.join(moves_start + [' '] + [s])
    
def try_game(s) :
    try :
        return 'e2' if type(s) != str else full_game(game_str(s), open_code(s))
    except :
        return 'e3'
    
data['full_game'] = data['pgn'].apply(try_game)

#Dropping the games with alternative rules or no referenced opening
index_to_drop = data.loc[(data['rules'] != 'chess') | (data['opening_code'] == '')].index

data.drop(index = index_to_drop, inplace = True)

#Calculating the number of the last turn
def num_moves(s) :
    white_num = re.findall(' \d+\. | \d+ ', s)
    if white_num == [] :
        return 1
    else :
        return int(re.findall('\d+', white_num[-1])[0])
    
data['num_moves'] = data['full_game'].apply(num_moves)

#Dropping short and problematic games
short_game = data.loc[data['num_moves'] < 6]

data.drop(index = short_game.index, inplace = True)

#Implementing the moves on a position
initial_pos = 'wRa1wNb1wBc1wQd1wKe1wBf1wNg1wRh1wpa2wpb2wpc2wpd2wpe2wpf2wpg2wph2bpa7bpb7bpc7bpd7bpe7bpf7bpg7bph7bRa8bNb8bBc8bQd8bKe8bBf8bNg8bRh8'

columns = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']

rows = ['1', '2', '3', '4', '5', '6', '7', '8']

def col_to_num(c) :
    dict = { 'a' : 1, 'b' : 2, 'c' : 3, 'd' : 4, 'e' : 5, 'f' : 6, 'g' : 7, 'h' : 8}
    return dict[c]

def num_to_col(n) :
    dict = { 1 : 'a', 2 : 'b', 3 : 'c', 4 : 'd', 5 : 'e', 6 : 'f', 7 : 'g', 8 : 'h'}
    return dict[n]

#function that tests if the line between 2 squares is obstructed in a position, or if the line is not straight.
def line_obstr_or_not_straight(square1, square2, pos) :
    if square2[0] < square1[0] or (square2[0] == square1[0] and square2[1] < square1[1]) :
        return line_obstr_or_not_straight(square2, square1, pos)
    #same column
    elif square1[0] == square2[0] :
        if [re.search(square1[0] + str(i), pos) for i in range(int(square1[1]) + 1, int(square2[1]))] == [None for i in range(int(square1[1]) + 1, int(square2[1]))] :
            return False
        else :
            return True
    #same row
    elif square1[1] == square2[1] :
        if [re.search(num_to_col(i) + square1[1], pos) for i in range(col_to_num(square1[0]) + 1, col_to_num(square2[0]))] == [None for i in range(col_to_num(square1[0]) + 1, col_to_num(square2[0]))] :
            return False
        else :
            return True
    #same ascending diagonal
    elif col_to_num(square1[0]) - int(square1[1]) == col_to_num(square2[0]) - int(square2[1]) :
        if [re.search(num_to_col(i) + str(i + int(square1[1]) - col_to_num(square1[0])), pos) for i in range(col_to_num(square1[0]) + 1, col_to_num(square2[0]))] == [None for i in range(col_to_num(square1[0]) + 1, col_to_num(square2[0]))] :
            return False
        else :
            return True
    #same descending diagonal
    elif col_to_num(square1[0]) + int(square1[1]) == col_to_num(square2[0]) + int(square2[1]) :
        if [re.search(num_to_col(i) + str(int(square1[1]) - i + col_to_num(square1[0])), pos) for i in range(col_to_num(square1[0]) + 1, col_to_num(square2[0]))] == [None for i in range(col_to_num(square1[0]) + 1, col_to_num(square2[0]))] :
            return False
        else :
            return True
    else :
        return True
    
line_obstr_or_not_straight('e4', 'e1', 'e3')

def w_move(pos, move) :
    if pos == '' :
        return ''
    else :
        #castle
        if move in ['O-O', 'O-O+'] :
            if re.search('wKe1', pos) == None or re.search('wRh1', pos) == None :
                return ''
            else :
                return pos.replace('wKe1', 'wKg1').replace('wRh1', 'wRf1')
        elif move in ['O-O-O', 'O-O-O+']:
            if re.search('wKe1', pos) == None or re.search('wRa1', pos) == None :
                return ''
            else :
                return pos.replace('wKe1', 'wKc1').replace('wRa1', 'wRd1')
        #move by a pawn
        elif move[0] in columns and move[1] in ['3', '5', '6', '7'] :
            if re.search('wp' + move[0] + str(int(move[1]) - 1), pos) == None :
                return ''
            else :
                return pos.replace('wp' + move[0] + str(int(move[1]) - 1), 'wp' + move[0:2])
        elif move[0] in columns and move[1] == '4' :
            third_row_content = re.search('..' + move[0] + '3', pos)
            if third_row_content != None :
                if third_row_content.group(0) != 'wp' + move[0] + '3' :
                    return ''
                else :
                    return pos.replace('wp' + move[0] + '3', 'wp' + move[0] + '4')
            else :
                second_row_content = re.search('..' + move[0] + '2', pos)
                if second_row_content == None or second_row_content.group(0) != 'wp' + move[0] + '2' :
                    return ''
                else :
                    return pos.replace('wp' + move[0] + '2', 'wp' + move[0] + '4')
            #capture by a pawn
        elif move[0] in columns and move[1] == 'x' and move[3] in ['3', '4', '5', '6', '7'] :
            start = re.search('wp' + move[0] + str(int(move[3]) - 1), pos)
            arrival = re.search('b.' + move[2:4], pos)
                #Capture "en passant"
            if arrival == None and move[3] == '6':
                arrival = re.search('bp' + move[2] + '5', pos)
            if start == None or arrival == None :
                return ''
            else :
                return pos.replace(start.group(0), 'wp' + move[2:4]).replace(arrival.group(0), '')
            #promotion of a pawn
        elif move[0] in columns and move[1:3] == '8=' :
            if re.search('wp' + move[0] + '7', pos) == None :
                return ''
            else :
                return pos.replace('wp' + move[0] + '7', 'w' + move[3] + move[0] + '8')
            #promotion with capture
        elif move[0] in columns and move[1] == 'x' and move[2] in columns and move[3:5] == '8=' :
            start = re.search('wp' + move[0] + '7', pos)
            arrival = re.search('b.' + move[2:4], pos)
            if start == None or arrival == None :
                return ''
            else :
                return pos.replace(start.group(0), 'w' + move[5] + move[2:4]).replace(arrival.group(0), '')
            #move of the King
        elif move[0] == 'K' and move[1] in columns :
            start = re.search('wK..', pos)
            return pos.replace(start.group(0), 'w' + move[0:3])
            #capture by the king
        elif move[0:2] == 'Kx' :
            start = re.search('wK..', pos)
            arrival = re.search('b.' + move[2:4], pos)
            if arrival == None :
                return ''
            else :
                return pos.replace(start.group(0), 'wK' + move[2:4]).replace(arrival.group(0), '')
            #move of a bishop
        elif move[0] == 'B' and move[1] in columns :
            bishops = re.findall('wB..', pos)
            if bishops == [] :
                return ''
            else :
                color_match = (int(move[2]) + int(bishops[0][3]) + col_to_num(move[1]) + col_to_num(bishops[0][2])) %2
                if color_match == 1 and len(bishops) == 1 :
                    return ''
                elif color_match == 0 :
                    return pos.replace(bishops[0], 'w' + move[0:3])
                else :
                    return pos.replace(bishops[1], 'w' + move[0:3])
            #Capture by a bishop
        elif move[0] == 'B' and move[1] == 'x' :
            bishops = re.findall('wB..', pos)
            arrival = re.search('b.' + move[2:4], pos)
            if bishops == [] or arrival == None :
                return ''
            else :
                color_match = (int(move[3]) + int(bishops[0][3]) + col_to_num(move[2]) + col_to_num(bishops[0][2])) %2
                if color_match == 1 and len(bishops) == 1 :
                    return ''
                elif color_match == 0 :
                    return pos.replace(bishops[0], 'wB' + move[2:4]).replace(arrival.group(0), '')
                else :
                    return pos.replace(bishops[1], 'wB' + move[2:4]).replace(arrival.group(0), '')
            #Move of a knight
        elif move[0] == 'N' and len(move) >= 5 and move[4] in rows and move[2] != 'x' :
            start = re.search('wN' + move[1:3], pos)
            arrival = re.search(move[3:5], pos)
            if start == None or arrival != None :
                return ''
            else :
                return pos.replace(start.group(0), 'wN' + move[3:5])
        elif move[0] == 'N' and len(move) >=4 and move[3] in rows and move[1] != 'x' :
            if move[1] in rows :
                start = re.search('wN.' + move[1], pos)
            elif move[1] in columns :
                start = re.search('wN' + move[1] + '.', pos)
            else :
                return ''
            arrival = re.search(move[2:4], pos)
            if start == None or  arrival != None :
                return ''
            else :
                return pos.replace(start.group(0), 'wN' + move[2:4])
        elif move[0] == 'N' and move.find('x') == -1 :
            if move[1] not in columns or move[2] not in rows :
                return ''
            else :
                knights = re.findall('wN..', pos)
                start = [k for k in knights if {abs(col_to_num(k[2]) - col_to_num(move[1])), abs(int(k[3]) - int(move[2]))} == {1, 2}]
                arrival = re.search(move[1:3], pos)
                if len(start) != 1 or arrival != None :
                    return ''
                else :
                    return pos.replace(start[0], 'wN' + move[1:3])
            #Capture by a knight
        elif move[0] == 'N' and len(move) >= 6 and move[5] in rows and move[3] == 'x' :
            start = re.search('wN' + move[1:3], pos)
            arrival = re.search('b.' + move[4:6], pos)
            if start == None or arrival == None :
                return ''
            else :
                return pos.replace(start.group(0), 'wN' + move[4:6]).replace(arrival.group(0), '')
        elif move[0] == 'N' and len(move) >=5 and move[4] in rows and move[2] == 'x' :
            if move[1] in rows :
                start = re.search('wN.' + move[1], pos)
            elif move[1] in columns :
                start = re.search('wN' + move[1] + '.', pos)
            else :
                return ''
            arrival = re.search('b.' + move[3:5], pos)
            if start == None or  arrival == None :
                return ''
            else :
                return pos.replace(start.group(0), 'wN' + move[3:5]).replace(arrival.group(0), '')
        elif move[0:2] == 'Nx' :
            if move[2] not in columns or move[3] not in rows :
                return ''
            else :
                knights = re.findall('wN..', pos)
                start = [k for k in knights if {abs(col_to_num(k[2]) - col_to_num(move[2])), abs(int(k[3]) - int(move[3]))} == {1, 2}]
                arrival = re.search('b.' + move[2:4], pos)
                if len(start) != 1 or arrival == None :
                    return ''
                else :
                    return pos.replace(start[0], 'wN' + move[2:4]).replace(arrival.group(0), '')
            #Move of a rook
        elif move[0] == 'R' and len(move) >=4 and move[3] in rows and move[1] != 'x' :
            if move[1] in rows :
                start = re.search('wR.' + move[1], pos)
            elif move[1] in columns :
                start = re.search('wR' + move[1] + '.', pos)
            else :
                return ''
            arrival = re.search(move[2:4], pos)
            if start == None or  arrival != None :
                return ''
            else :
                return pos.replace(start.group(0), 'wR' + move[2:4])
        elif move[0] == 'R' and move.find('x') == -1 :
             if move[1] not in columns or move[2] not in rows :
                 return ''
             else :
                 rooks = re.findall('wR..', pos)
                 start = [r for r in rooks if (r[2] == move[1] or r[3] == move[2]) and not line_obstr_or_not_straight(r[2:4], move[1:3], pos)]
                 arrival = re.search(move[1:3], pos)
                 if len(start) != 1 or arrival != None :
                     return ''
                 else :
                     return pos.replace(start[0], 'wR' + move[1:3])
            #Capture by a rook
        elif move[0] == 'R' and len(move) >=5 and move[4] in rows and move[2] == 'x' :
            if move[1] in rows :
                start = re.search('wR.' + move[1], pos)
            elif move[1] in columns :
                start = re.search('wR' + move[1] + '.', pos)
            else :
                return ''
            arrival = re.search('b.' + move[3:5], pos)
            if start == None or  arrival == None :
                return ''
            else :
                return pos.replace(start.group(0), 'wR' + move[3:5]).replace(arrival.group(0), '')
        elif move[0] == 'R' and move[1] == 'x' :
             if move[2] not in columns or move[3] not in rows :
                 return ''
             else :
                 rooks = re.findall('wR..', pos)
                 start = [r for r in rooks if (r[2] == move[2] or r[3] == move[3]) and not line_obstr_or_not_straight(r[2:4], move[2:4], pos)]
                 arrival = re.search('b.' + move[2:4], pos)
                 if len(start) != 1 or arrival == None :
                     return ''
                 else :
                     return pos.replace(start[0], 'wR' + move[2:4]).replace(arrival.group(0), '')
            #Move of the queen
        elif move[0] == 'Q' and len(move) >=4 and move[3] in rows and move[1] != 'x' :
            if move[1] in rows :
                start = [q for q in re.findall('wQ.' + move[1], pos) if not line_obstr_or_not_straight(q[2:4], move[2:4], pos)]
            elif move[1] in columns :
                start = [q for q in re.findall('wQ' + move[1] + '.', pos) if not line_obstr_or_not_straight(q[2:4], move[2:4], pos)]
            else :
                return ''
            arrival = re.search(move[2:4], pos)
            if len(start) != 1 or  arrival != None :
                return ''
            else :
                return pos.replace(start[0], 'wQ' + move[2:4])
        elif move[0] == 'Q' and move.find('x') == -1 :
             if move[1] not in columns or move[2] not in rows :
                 return ''
             else :
                 start = [q for q in re.findall('wQ..', pos) if not line_obstr_or_not_straight(q[2:4], move[1:3], pos)]
                 arrival = re.search(move[1:3], pos)
                 if len(start) != 1 or arrival != None :
                     return ''
                 else :
                     return pos.replace(start[0], 'wQ' + move[1:3])
            #Capture by the queen
        elif move[0] == 'Q' and len(move) >=5 and move[4] in rows and move[2] == 'x' :
            if move[1] in rows :
                start = [q for q in re.findall('wQ.' + move[1], pos) if not line_obstr_or_not_straight(q[2:4], move[3:5], pos)]
            elif move[1] in columns :
                start = [q for q in re.findall('wQ' + move[1] + '.', pos) if not line_obstr_or_not_straight(q[2:4], move[3:5], pos)]
            else :
                return ''
            arrival = re.search('b.' + move[3:5], pos)
            if len(start) != 1 or  arrival == None :
                return ''
            else :
                return pos.replace(start[0], 'wQ' + move[3:5]).replace(arrival.group(0), '')
        elif move[0] == 'Q' and move[1] == 'x' :
             if move[2] not in columns or move[3] not in rows :
                 return ''
             else :
                 start = [q for q in re.findall('wQ..', pos) if not line_obstr_or_not_straight(q[2:4], move[2:4], pos)]
                 arrival = re.search('b.' + move[2:4], pos)
                 if len(start) != 1 or arrival == None :
                     return ''
                 else :
                     return pos.replace(start[0], 'wQ' + move[2:4]).replace(arrival.group(0), '')
        else :
            return pos
        
        

data['promotion'] = data['full_game'].apply(lambda s : False if re.search('=', s) == None else True)

data['strange_prom'] = data['full_game'].apply(lambda s : False if re.search('=[B]', s) == None else True)
weird = data.loc[data['strange_prom']]

w_move('wpe2bNc3', 'exc3')
w_move('wpe7bQa8', 'exa8=Q')
w_move('wKf2bQf3', 'Kxf3')
w_move('wBa1wBb1', 'Bh7')
w_move('wBa1wBb1bQh8', 'Bxh8')
w_move('wNe3wNg3', 'Nf5')
w_move('wNe3wNd7bQf5', 'Nxf5')
w_move('wRe1bQe4', 'Rexe4')
w_move('wpe5bpa5', 'exa6')
w_move('wQe4bpe5bRe8', 'Qxe8')

def b_move(pos, move) :
    if pos == '' :
        return ''
    else :
        #castle
        if move in ['O-O', 'O-O+'] :
            if re.search('bKe8', pos) == None or re.search('bRh8', pos) == None :
                return ''
            else :
                return pos.replace('bKe8', 'bKg8').replace('bRh8', 'bRf8')
        elif move in ['O-O-O', 'O-O-O+']:
            if re.search('bKe8', pos) == None or re.search('bRa8', pos) == None :
                return ''
            else :
                return pos.replace('bKe8', 'bKc8').replace('bRa8', 'bRd8')
        #move by a pawn
        elif move[0] in columns and move[1] in ['2', '3', '4', '6'] :
            if re.search('bp' + move[0] + str(int(move[1]) + 1), pos) == None :
                return ''
            else :
                return pos.replace('bp' + move[0] + str(int(move[1]) + 1), 'bp' + move[0:2])
        elif move[0] in columns and move[1] == '5' :
            sixth_row_content = re.search('..' + move[0] + '6', pos)
            if sixth_row_content != None :
                if sixth_row_content.group(0) != 'bp' + move[0] + '6' :
                    return ''
                else :
                    return pos.replace('bp' + move[0] + '6', 'bp' + move[0] + '5')
            else :
                seventh_row_content = re.search('..' + move[0] + '7', pos)
                if seventh_row_content == None or seventh_row_content.group(0) != 'bp' + move[0] + '7' :
                    return ''
                else :
                    return pos.replace('bp' + move[0] + '7', 'bp' + move[0] + '5')
            #capture by a pawn
        elif move[0] in columns and move[1] == 'x' and move[3] in ['2', '3', '4', '5', '6'] :
            start = re.search('bp' + move[0] + str(int(move[3]) + 1), pos)
            arrival = re.search('w.' + move[2:4], pos)
                #Capture "en passant"
            if arrival == None and move[3] == '3':
                arrival = re.search('wp' + move[2] + '4', pos)
            if start == None or arrival == None :
                return ''
            else :
                return pos.replace(start.group(0), 'bp' + move[2:4]).replace(arrival.group(0), '')
            #promotion of a pawn
        elif move[0] in columns and move[1:3] == '1=' :
            if re.search('bp' + move[0] + '2', pos) == None :
                return ''
            else :
                return pos.replace('bp' + move[0] + '2', 'b' + move[3] + move[0] + '1')
            #promotion with capture
        elif move[0] in columns and move[1] == 'x' and move[2] in columns and move[3:5] == '1=' :
            start = re.search('bp' + move[0] + '2', pos)
            arrival = re.search('w.' + move[2:4], pos)
            if start == None or arrival == None :
                return ''
            else :
                return pos.replace(start.group(0), 'b' + move[5] + move[2:4]).replace(arrival.group(0), '')
            #move of the King
        elif move[0] == 'K' and move[1] in columns :
            start = re.search('bK..', pos)
            return pos.replace(start.group(0), 'b' + move[0:3])
            #capture by the king
        elif move[0:2] == 'Kx' :
            start = re.search('bK..', pos)
            arrival = re.search('w.' + move[2:4], pos)
            if arrival == None :
                return ''
            else :
                return pos.replace(start.group(0), 'bK' + move[2:4]).replace(arrival.group(0), '')
            #move of a bishop
        elif move[0] == 'B' and move[1] in columns :
            bishops = re.findall('bB..', pos)
            if bishops == [] :
                return ''
            else :
                color_match = (int(move[2]) + int(bishops[0][3]) + col_to_num(move[1]) + col_to_num(bishops[0][2])) %2
                if color_match == 1 and len(bishops) == 1 :
                    return ''
                elif color_match == 0 :
                    return pos.replace(bishops[0], 'b' + move[0:3])
                else :
                    return pos.replace(bishops[1], 'b' + move[0:3])
            #Capture by a bishop
        elif move[0] == 'B' and move[1] == 'x' :
            bishops = re.findall('bB..', pos)
            arrival = re.search('w.' + move[2:4], pos)
            if bishops == [] or arrival == None :
                return ''
            else :
                color_match = (int(move[3]) + int(bishops[0][3]) + col_to_num(move[2]) + col_to_num(bishops[0][2])) %2
                if color_match == 1 and len(bishops) == 1 :
                    return ''
                elif color_match == 0 :
                    return pos.replace(bishops[0], 'bB' + move[2:4]).replace(arrival.group(0), '')
                else :
                    return pos.replace(bishops[1], 'bB' + move[2:4]).replace(arrival.group(0), '')
            #Move of a knight
        elif move[0] == 'N' and len(move) >= 5 and move[4] in rows and move[2] != 'x' :
            start = re.search('bN' + move[1:3], pos)
            arrival = re.search(move[3:5], pos)
            if start == None or arrival != None :
                return ''
            else :
                return pos.replace(start.group(0), 'bN' + move[3:5])
        elif move[0] == 'N' and len(move) >=4 and move[3] in rows and move[1] != 'x' :
            if move[1] in rows :
                start = re.search('bN.' + move[1], pos)
            elif move[1] in columns :
                start = re.search('bN' + move[1] + '.', pos)
            else :
                return ''
            arrival = re.search(move[2:4], pos)
            if start == None or  arrival != None :
                return ''
            else :
                return pos.replace(start.group(0), 'bN' + move[2:4])
        elif move[0] == 'N' and move.find('x') == -1 :
            if move[1] not in columns or move[2] not in rows :
                return ''
            else :
                knights = re.findall('bN..', pos)
                start = [k for k in knights if {abs(col_to_num(k[2]) - col_to_num(move[1])), abs(int(k[3]) - int(move[2]))} == {1, 2}]
                arrival = re.search(move[1:3], pos)
                if len(start) != 1 or arrival != None :
                    return ''
                else :
                    return pos.replace(start[0], 'bN' + move[1:3])
            #Capture by a knight
        elif move[0] == 'N' and len(move) >= 6 and move[5] in rows and move[3] == 'x' :
            start = re.search('bN' + move[1:3], pos)
            arrival = re.search('w.' + move[4:6], pos)
            if start == None or arrival == None :
                return ''
            else :
                return pos.replace(start.group(0), 'bN' + move[4:6]).replace(arrival.group(0), '')
        elif move[0] == 'N' and len(move) >=5 and move[4] in rows and move[2] == 'x' :
            if move[1] in rows :
                start = re.search('bN.' + move[1], pos)
            elif move[1] in columns :
                start = re.search('bN' + move[1] + '.', pos)
            else :
                return ''
            arrival = re.search('w.' + move[3:5], pos)
            if start == None or  arrival == None :
                return ''
            else :
                return pos.replace(start.group(0), 'bN' + move[3:5]).replace(arrival.group(0), '')
        elif move[0:2] == 'Nx' :
            if move[2] not in columns or move[3] not in rows :
                return ''
            else :
                knights = re.findall('bN..', pos)
                start = [k for k in knights if {abs(col_to_num(k[2]) - col_to_num(move[2])), abs(int(k[3]) - int(move[3]))} == {1, 2}]
                arrival = re.search('w.' + move[2:4], pos)
                if len(start) != 1 or arrival == None :
                    return ''
                else :
                    return pos.replace(start[0], 'bN' + move[2:4]).replace(arrival.group(0), '')
            #Move of a rook
        elif move[0] == 'R' and len(move) >=4 and move[3] in rows and move[1] != 'x' :
            if move[1] in rows :
                start = re.search('bR.' + move[1], pos)
            elif move[1] in columns :
                start = re.search('bR' + move[1] + '.', pos)
            else :
                return ''
            arrival = re.search(move[2:4], pos)
            if start == None or  arrival != None :
                return ''
            else :
                return pos.replace(start.group(0), 'bR' + move[2:4])
        elif move[0] == 'R' and move.find('x') == -1 :
             if move[1] not in columns or move[2] not in rows :
                 return ''
             else :
                 rooks = re.findall('bR..', pos)
                 start = [r for r in rooks if (r[2] == move[1] or r[3] == move[2]) and not line_obstr_or_not_straight(r[2:4], move[1:3], pos)]
                 arrival = re.search(move[1:3], pos)
                 if len(start) != 1 or arrival != None :
                     return ''
                 else :
                     return pos.replace(start[0], 'bR' + move[1:3])
            #Capture by a rook
        elif move[0] == 'R' and len(move) >=5 and move[4] in rows and move[2] == 'x' :
            if move[1] in rows :
                start = re.search('bR.' + move[1], pos)
            elif move[1] in columns :
                start = re.search('bR' + move[1] + '.', pos)
            else :
                return ''
            arrival = re.search('w.' + move[3:5], pos)
            if start == None or  arrival == None :
                return ''
            else :
                return pos.replace(start.group(0), 'bR' + move[3:5]).replace(arrival.group(0), '')
        elif move[0] == 'R' and move[1] == 'x' :
             if move[2] not in columns or move[3] not in rows :
                 return ''
             else :
                 rooks = re.findall('bR..', pos)
                 start = [r for r in rooks if (r[2] == move[2] or r[3] == move[3]) and not line_obstr_or_not_straight(r[2:4], move[2:4], pos)]
                 arrival = re.search('w.' + move[2:4], pos)
                 if len(start) != 1 or arrival == None :
                     return ''
                 else :
                     return pos.replace(start[0], 'bR' + move[2:4]).replace(arrival.group(0), '')
            #Move of the queen
        elif move[0] == 'Q' and len(move) >=4 and move[3] in rows and move[1] != 'x' :
            if move[1] in rows :
                start = [q for q in re.findall('bQ.' + move[1], pos) if not line_obstr_or_not_straight(q[2:4], move[2:4], pos)]
            elif move[1] in columns :
                start = [q for q in re.findall('bQ' + move[1] + '.', pos) if not line_obstr_or_not_straight(q[2:4], move[2:4], pos)]
            else :
                return ''
            arrival = re.search(move[2:4], pos)
            if len(start) != 1 or  arrival != None :
                return ''
            else :
                return pos.replace(start[0], 'bQ' + move[2:4])
        elif move[0] == 'Q' and move.find('x') == -1 :
             if move[1] not in columns or move[2] not in rows :
                 return ''
             else :
                 start = [q for q in re.findall('bQ..', pos) if not line_obstr_or_not_straight(q[2:4], move[1:3], pos)]
                 arrival = re.search(move[1:3], pos)
                 if len(start) != 1 or arrival != None :
                     return ''
                 else :
                     return pos.replace(start[0], 'bQ' + move[1:3])
            #Capture by the queen
        elif move[0] == 'Q' and len(move) >=5 and move[4] in rows and move[2] == 'x' :
            if move[1] in rows :
                start = [q for q in re.findall('bQ.' + move[1], pos) if not line_obstr_or_not_straight(q[2:4], move[3:5], pos)]
            elif move[1] in columns :
                start = [q for q in re.findall('bQ' + move[1] + '.', pos) if not line_obstr_or_not_straight(q[2:4], move[3:5], pos)]
            else :
                return ''
            arrival = re.search('w.' + move[3:5], pos)
            if len(start) != 1 or  arrival == None :
                return ''
            else :
                return pos.replace(start[0], 'bQ' + move[3:5]).replace(arrival.group(0), '')
        elif move[0] == 'Q' and move[1] == 'x' :
             if move[2] not in columns or move[3] not in rows :
                 return ''
             else :
                 start = [q for q in re.findall('bQ..', pos) if not line_obstr_or_not_straight(q[2:4], move[2:4], pos)]
                 arrival = re.search('w.' + move[2:4], pos)
                 if len(start) != 1 or arrival == None :
                     return ''
                 else :
                     return pos.replace(start[0], 'bQ' + move[2:4]).replace(arrival.group(0), '')
        else :
            return pos

def iter_moves(pos, moves) :
    if moves == [] :
        return pos
    elif len(moves) == 1 :
        return w_move(pos, moves[0])
    else :
        try :
            test = b_move(w_move(pos, moves[0]), moves[1])
        except :
            #print(moves[0], moves[1])
            return ''
        if test == '' :
            #print(moves[0], moves[1])
            return ''
        else :
            return iter_moves(test, moves[2:])
            #return iter_moves(b_move(w_move(pos, moves[0]), moves[1]), moves[2:])
    
data.info()

def move_list_from_string(s) :
    return [j for i in re.split(r'^1 |^1. | \d+ | \d+\. | \d+... | 0-1| 1-0| 1/2-1/2', s) for j in re.split(' ', i) if j != '']

data.at[0, 'full_game']
move_list_from_string(data.at[0, 'full_game'])
iter_moves(initial_pos, move_list_from_string(data.at[0, 'full_game']))

data['final_pos'] = data['full_game'].apply(lambda s : iter_moves(initial_pos, move_list_from_string(s)))
(data['final_pos'] == '').sum()
prob = data.loc[data['final_pos'] == '']
iter_moves(initial_pos, move_list_from_string(data.at[32, 'full_game']))
data.at[32, 'full_game']
data.loc[data['num_moves'] < 11].shape

data.drop(axis = 1, columns = ['game_id', 'game_url', 'pgn', 'rules'], inplace = True)
data['time_class'].value_counts()

def score(piece) :
    dict = {'p' : 1, 'K' : 0, 'Q' : 9, 'B' : 3, 'N' : 3, 'R' : 5}
    return dict[piece]


def material(color, pos) :
    sc = 0
    for i in range(len(pos)//4) :
        if pos[4*i] == color :
            sc += score(pos[4 * i + 1])
    return sc

data['final_material'] = data['final_pos'].apply(lambda s : min([material('w', s), material('b', s)]))
data['final_balance'] = data['final_pos'].apply(lambda s : material('w', s) - material('b', s))

draws = data.loc[(data['white_points'] == 0.5) & (abs(data['final_balance']) > 1)]
data.loc[(data['white_points'] == 0.5)].shape[0]
bad_beats = data.loc[((data['white_points'] == 1) & (data['final_balance'] < -1)) | ((data['white_points'] == 0) & (data['final_balance'] > 1))]

data.to_csv(r'C:\Users\anato\Documents\IRONHACK\chess_cleaned.csv')
sample = data.sample(n = 30000, axis = 0, random_state = 17)
sample.to_csv(r'C:\Users\anato\Documents\IRONHACK\IronFrandre\Project 9 Chess games of woman grandmasters\samplechess.csv')

#Choosing the features.
x = data[['black_rating', 'num_moves']]
x['difference_of_rating'] = data['white_rating'] - data['black_rating']
x['bullet'] = data['time_class'].apply(lambda s : 1 if s == 'bullet' else 0)
x['rapid'] = data['time_class'].apply(lambda s : 1 if s == 'rapid' else 0)
x['blitz'] = data['time_class'].apply(lambda s : 1 if s == 'blitz' else 0)
x['daily'] = data['time_class'].apply(lambda s : 1 if s == 'daily' else 0)
x['wgm_is_white'] = (data['wgm_username'] == data['white_username'].apply(lambda s : s.lower())).apply(lambda b : 1 if b else 0)
y = 2 * data['white_points']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=5)

#Implementing the machine learning
#First, decision tree classifier.
dtc = DecisionTreeClassifier(max_depth = 5, random_state = 1)
dtc.fit(x_train, y_train)
y_DT = dtc.predict(x_test)
print(classification_report(y_test, y_DT), sep = "\n")
print(confusion_matrix(y_test, y_DT))

plt.figure(figsize = (20,15))
fn = ["black_rating", "num_moves", "difference_of_rating", "bullet", "rapid", "blitz", "daily", "wgm_is_white"]
cn = ['black_wins', 'draw', 'white_wins']
plot_tree(dtc, feature_names = fn, class_names = cn, filled = True);

#Random forest classifier.
rfc = RandomForestClassifier(max_depth = 7, random_state = 1)
rfc.fit(x_train, y_train)
y_RF = rfc.predict(x_test)
print(classification_report(y_test, y_RF), sep = "\n")
print(confusion_matrix(y_test, y_RF))

#K nearest neighbours classifier.
knn = KNeighborsClassifier(n_neighbors=40)
knn.fit(x_train, y_train)
y_KN = knn.predict(x_test)
print(classification_report(y_test, y_KN), sep = "\n")
print(confusion_matrix(y_test, y_KN))

#Figures
pie1 = [data.loc[data['white_points'] == 1].shape[0], data.loc[data['white_points'] == 0].shape[0], data.loc[data['white_points'] == 0.5].shape[0]]
labels1 = ['White', 'Black', 'Draw']

fig1 = plt.figure(figsize = (15, 8))
plt.title("Proportion of wins")
plt.pie(pie1, labels = labels1, autopct='%1.1f%%')
plt.show()

data['difference_bins'] = (data['white_rating'] - data['black_rating']).apply(lambda x : round(x/100, 0) * 100)
def regroup(x) :
    if x < -500 :
        return -500
    elif x > 500 :
        return 500
    else :
        return x
data['difference_bins'] = data['difference_bins'].apply(regroup)
df2 = data[['full_game', 'white_points', 'difference_bins']].groupby(by = ['difference_bins', 'white_points'], as_index = False).agg('count')
fig2 = plt.figure(figsize = (15, 8))
sns.barplot(data = df2, x = 'difference_bins', y = 'full_game', hue = 'white_points')
plt.xlabel('Rating of white minus rating of black')
plt.ylabel('Number of games')
plt.show()

pie3 = [draws.shape[0], bad_beats.shape[0], data.shape[0] - draws.shape[0] - bad_beats.shape[0]]
label3 = ['Draws with large material unbalance', 'Wins with material disadvantage', 'Other']
fig3 = plt.figure(figsize = (15, 8))
plt.title("Surprising results relative to material balance")
plt.pie(pie3, labels = label3, autopct='%1.1f%%')
plt.show()

fig4 = plt.figure(figsize = (15, 8))
sns.violinplot(x = (data['white_rating'] - data['black_rating']).apply(regroup))
plt.title('Repartition of the difference between the ratings of the white and the black player')
plt.show()

fig5 = plt.figure()
sns.violinplot(x = data.loc[data['white_points'] == 1, 'final_balance'])
plt.title('Violin plot of the final material balance in games won by white')
plt.show()

fig6 = plt.figure()
sns.violinplot(x = data.loc[data['white_points'] == 0, 'final_balance'])
plt.title('Violin plot of the final material balance in games won by black')
plt.show()

fig7 = plt.figure()
sns.violinplot(x = data.loc[data['white_points'] == 0.5, 'final_balance'])
plt.title('Violin plot of the final material balance in games ended as a draw')
plt.show()

df = pd.read_csv(r"C:\Users\anato\Documents\IRONHACK\games_wgm.csv")
df.at[0, 'pgn']

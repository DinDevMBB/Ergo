import pandas as pd
listA =[{'Index': 111, 'Score': 1}, {'Index': 112, 'Score': 2}, {'Index': 113, 'Score': 2}, {'Index': 114, 'Score': 3}, {'Index': 115, 'Score': 4}, {'Index': 121, 'Score': 2}, {'Index': 122, 'Score': 3}, {'Index': 123, 'Score': 4}, {'Index': 124, 'Score': 5}, {'Index': 125, 'Score': 6}, {'Index': 131, 'Score': 3}, {'Index': 132, 'Score': 4}, {'Index': 133, 'Score': 5}, {'Index': 134, 'Score': 6}, {'Index': 135, 'Score': 7}, {'Index': 141, 'Score': 4}, {'Index': 142, 'Score': 5}, {'Index': 143, 'Score': 6}, {'Index': 144, 'Score': 7}, {'Index': 145, 'Score': 8}, {'Index': 211, 'Score': 1}, {'Index': 212, 'Score': 3}, {'Index': 213, 'Score': 4}, {'Index': 214, 'Score': 5}, {'Index': 215, 'Score': 6}, {'Index': 221, 'Score': 2}, {'Index': 222, 'Score': 4}, {'Index': 223, 'Score': 5}, {'Index': 224, 'Score': 6}, {'Index': 225, 'Score': 7}, {'Index': 231, 'Score': 3}, {'Index': 232, 'Score': 5}, {'Index': 233, 'Score': 6}, {'Index': 234, 'Score': 7}, {'Index': 235, 'Score': 8}, {'Index': 241, 'Score': 4}, {'Index': 242, 'Score': 6}, {'Index': 243, 'Score': 7}, {'Index': 244, 'Score': 8}, {'Index': 245, 'Score': 9}, {'Index': 311, 'Score': 3}, {'Index': 312, 'Score': 4}, {'Index': 313, 'Score': 5}, {'Index': 314, 'Score': 6}, {'Index': 315, 'Score': 7}, {'Index': 321, 'Score': 3}, {'Index': 322, 'Score': 5}, {'Index': 323, 'Score': 6}, {'Index': 324, 'Score': 7}, {'Index': 325, 'Score': 8}, {'Index': 331, 'Score': 5}, {'Index': 332, 'Score': 6}, {'Index': 333, 'Score': 7}, {'Index': 334, 'Score': 8}, {'Index': 335, 'Score': 9}, {'Index': 341, 'Score': 6}, {'Index': 342, 'Score': 7}, {'Index': 343, 'Score': 8}, {'Index': 344, 'Score': 9}, {'Index': 345, 'Score': 9}]
tblA =pd.DataFrame(listA)
tblA.set_index('Index',inplace =True)

def GetA(idx):
    ScoreA =tblA.loc[idx]['Score']
    return ScoreA

listB =[{'Index': 111, 'Score': 1}, {'Index': 112, 'Score': 2}, {'Index': 113, 'Score': 2}, {'Index': 121, 'Score': 1}, {'Index': 122, 'Score': 2}, {'Index': 123, 'Score': 3}, {'Index': 211, 'Score': 1}, {'Index': 212, 'Score': 2}, {'Index': 213, 'Score': 3}, {'Index': 221, 'Score': 2}, {'Index': 222, 'Score': 3}, {'Index': 223, 'Score': 4}, {'Index': 311, 'Score': 3}, {'Index': 312, 'Score': 4}, {'Index': 313, 'Score': 5}, {'Index': 321, 'Score': 4}, {'Index': 322, 'Score': 5}, {'Index': 323, 'Score': 5}, {'Index': 411, 'Score': 4}, {'Index': 412, 'Score': 5}, {'Index': 413, 'Score': 5}, {'Index': 421, 'Score': 5}, {'Index': 422, 'Score': 6}, {'Index': 423, 'Score': 7}, {'Index': 511, 'Score': 6}, {'Index': 512, 'Score': 7}, {'Index': 513, 'Score': 8}, {'Index': 521, 'Score': 7}, {'Index': 522, 'Score': 8}, {'Index': 523, 'Score': 8}, {'Index': 611, 'Score': 7}, {'Index': 612, 'Score': 8}, {'Index': 613, 'Score': 8}, {'Index': 621, 'Score': 8}, {'Index': 622, 'Score': 9}, {'Index': 623, 'Score': 9}]
tblB =pd.DataFrame(listB)
tblB.set_index('Index',inplace =True)

def GetB(idx):
    ScoreB =tblB.loc[idx]['Score']
    return ScoreB

listC =[{'A': 1, 'B': 1, 'Score': 1}, {'A': 1, 'B': 2, 'Score': 1}, {'A': 1, 'B': 3, 'Score': 1}, {'A': 1, 'B': 4, 'Score': 2}, {'A': 1, 'B': 5, 'Score': 3}, {'A': 1, 'B': 6, 'Score': 3}, {'A': 1, 'B': 7, 'Score': 4}, {'A': 1, 'B': 8, 'Score': 5}, {'A': 1, 'B': 9, 'Score': 6}, {'A': 1, 'B': 10, 'Score': 7}, {'A': 1, 'B': 11, 'Score': 7}, {'A': 1, 'B': 12, 'Score': 7}, {'A': 2, 'B': 1, 'Score': 1}, {'A': 2, 'B': 2, 'Score': 2}, {'A': 2, 'B': 3, 'Score': 2}, {'A': 2, 'B': 4, 'Score': 3}, {'A': 2, 'B': 5, 'Score': 4}, {'A': 2, 'B': 6, 'Score': 4}, {'A': 2, 'B': 7, 'Score': 5}, {'A': 2, 'B': 8, 'Score': 6}, {'A': 2, 'B': 9, 'Score': 6}, {'A': 2, 'B': 10, 'Score': 7}, {'A': 2, 'B': 11, 'Score': 7}, {'A': 2, 'B': 12, 'Score': 8}, {'A': 3, 'B': 1, 'Score': 2}, {'A': 3, 'B': 2, 'Score': 3}, {'A': 3, 'B': 3, 'Score': 3}, {'A': 3, 'B': 4, 'Score': 3}, {'A': 3, 'B': 5, 'Score': 4}, {'A': 3, 'B': 6, 'Score': 5}, {'A': 3, 'B': 7, 'Score': 6}, {'A': 3, 'B': 8, 'Score': 7}, {'A': 3, 'B': 9, 'Score': 7}, {'A': 3, 'B': 10, 'Score': 8}, {'A': 3, 'B': 11, 'Score': 8}, {'A': 3, 'B': 12, 'Score': 8}, {'A': 4, 'B': 1, 'Score': 3}, {'A': 4, 'B': 2, 'Score': 4}, {'A': 4, 'B': 3, 'Score': 4}, {'A': 4, 'B': 4, 'Score': 4}, {'A': 4, 'B': 5, 'Score': 5}, {'A': 4, 'B': 6, 'Score': 6}, {'A': 4, 'B': 7, 'Score': 7}, {'A': 4, 'B': 8, 'Score': 8}, {'A': 4, 'B': 9, 'Score': 8}, {'A': 4, 'B': 10, 'Score': 9}, {'A': 4, 'B': 11, 'Score': 9}, {'A': 4, 'B': 12, 'Score': 9}, {'A': 5, 'B': 1, 'Score': 4}, {'A': 5, 'B': 2, 'Score': 4}, {'A': 5, 'B': 3, 'Score': 4}, {'A': 5, 'B': 4, 'Score': 5}, {'A': 5, 'B': 5, 'Score': 6}, {'A': 5, 'B': 6, 'Score': 7}, {'A': 5, 'B': 7, 'Score': 8}, {'A': 5, 'B': 8, 'Score': 8}, {'A': 5, 'B': 9, 'Score': 9}, {'A': 5, 'B': 10, 'Score': 9}, {'A': 5, 'B': 11, 'Score': 9}, {'A': 5, 'B': 12, 'Score': 9}, {'A': 6, 'B': 1, 'Score': 6}, {'A': 6, 'B': 2, 'Score': 6}, {'A': 6, 'B': 3, 'Score': 6}, {'A': 6, 'B': 4, 'Score': 7}, {'A': 6, 'B': 5, 'Score': 8}, {'A': 6, 'B': 6, 'Score': 8}, {'A': 6, 'B': 7, 'Score': 9}, {'A': 6, 'B': 8, 'Score': 9}, {'A': 6, 'B': 9, 'Score': 10}, {'A': 6, 'B': 10, 'Score': 10}, {'A': 6, 'B': 11, 'Score': 10}, {'A': 6, 'B': 12, 'Score': 10}, {'A': 7, 'B': 1, 'Score': 7}, {'A': 7, 'B': 2, 'Score': 7}, {'A': 7, 'B': 3, 'Score': 7}, {'A': 7, 'B': 4, 'Score': 8}, {'A': 7, 'B': 5, 'Score': 9}, {'A': 7, 'B': 6, 'Score': 9}, {'A': 7, 'B': 7, 'Score': 9}, {'A': 7, 'B': 8, 'Score': 10}, {'A': 7, 'B': 9, 'Score': 10}, {'A': 7, 'B': 10, 'Score': 11}, {'A': 7, 'B': 11, 'Score': 11}, {'A': 7, 'B': 12, 'Score': 11}, {'A': 8, 'B': 1, 'Score': 8}, {'A': 8, 'B': 2, 'Score': 8}, {'A': 8, 'B': 3, 'Score': 8}, {'A': 8, 'B': 4, 'Score': 9}, {'A': 8, 'B': 5, 'Score': 10}, {'A': 8, 'B': 6, 'Score': 10}, {'A': 8, 'B': 7, 'Score': 10}, {'A': 8, 'B': 8, 'Score': 10}, {'A': 8, 'B': 9, 'Score': 10}, {'A': 8, 'B': 10, 'Score': 11}, {'A': 8, 'B': 11, 'Score': 11}, {'A': 8, 'B': 12, 'Score': 11}, {'A': 9, 'B': 1, 'Score': 9}, {'A': 9, 'B': 2, 'Score': 9}, {'A': 9, 'B': 3, 'Score': 9}, {'A': 9, 'B': 4, 'Score': 10}, {'A': 9, 'B': 5, 'Score': 10}, {'A': 9, 'B': 6, 'Score': 10}, {'A': 9, 'B': 7, 'Score': 11}, {'A': 9, 'B': 8, 'Score': 11}, {'A': 9, 'B': 9, 'Score': 11}, {'A': 9, 'B': 10, 'Score': 12}, {'A': 9, 'B': 11, 'Score': 12}, {'A': 9, 'B': 12, 'Score': 12}, {'A': 10, 'B': 1, 'Score': 10}, {'A': 10, 'B': 2, 'Score': 10}, {'A': 10, 'B': 3, 'Score': 10}, {'A': 10, 'B': 4, 'Score': 11}, {'A': 10, 'B': 5, 'Score': 11}, {'A': 10, 'B': 6, 'Score': 11}, {'A': 10, 'B': 7, 'Score': 12}, {'A': 10, 'B': 8, 'Score': 12}, {'A': 10, 'B': 9, 'Score': 12}, {'A': 10, 'B': 10, 'Score': 12}, {'A': 10, 'B': 11, 'Score': 12}, {'A': 10, 'B': 12, 'Score': 12}, {'A': 11, 'B': 1, 'Score': 11}, {'A': 11, 'B': 2, 'Score': 11}, {'A': 11, 'B': 3, 'Score': 11}, {'A': 11, 'B': 4, 'Score': 12}, {'A': 11, 'B': 5, 'Score': 12}, {'A': 11, 'B': 6, 'Score': 12}, {'A': 11, 'B': 7, 'Score': 12}, {'A': 11, 'B': 8, 'Score': 12}, {'A': 11, 'B': 9, 'Score': 12}, {'A': 11, 'B': 10, 'Score': 12}, {'A': 11, 'B': 11, 'Score': 12}, {'A': 11, 'B': 12, 'Score': 12}, {'A': 12, 'B': 1, 'Score': 12}, {'A': 12, 'B': 2, 'Score': 12}, {'A': 12, 'B': 3, 'Score': 12}, {'A': 12, 'B': 4, 'Score': 12}, {'A': 12, 'B': 5, 'Score': 12}, {'A': 12, 'B': 6, 'Score': 12}, {'A': 12, 'B': 7, 'Score': 12}, {'A': 12, 'B': 8, 'Score': 12}, {'A': 12, 'B': 9, 'Score': 12}, {'A': 12, 'B': 10, 'Score': 12}, {'A': 12, 'B': 11, 'Score': 12}, {'A': 12, 'B': 12, 'Score': 12}]

tblC =pd.DataFrame(listC)


def GetC(a,b):

    ScoreC = tblC.loc[(tblC['A']==a )& (tblC['B']==b ), 'Score'].values[0]
    return ScoreC
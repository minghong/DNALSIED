# 00-A  10-C  01-T  11-G
import math
# 00-A  10-C  01-T  11-G    新添代码
def data_DNA(input_data):
    if input_data == "00":
        return 'A'
    elif input_data == "10":
        return 'C'
    elif input_data == "01":
        return 'T'
    elif input_data == "11":
        return 'G'
    else:
        ValueError('please input correct value!')


def confirm_dna_seq_value(input_data):
    if input_data == 'A':
        return '00'
    elif input_data == 'C':
        return '10'
    elif input_data == 'T':
        return '01'
    elif input_data == 'G':
        return '11'
    else:
        ValueError('please input correct value!')

# DNA转为二进制
def convert_dna_to_data_sequence(code):  # 11 01 11 01 11 10 00 01 00 10
    dna_seq = []
    str1 = ""
    for i in range(len(code)):
        a = code[i]
        value = confirm_dna_seq_value(a)
        str1 = str1 + str(value)
    return str1

def convert_dna_to_data_sequence1(Odd , Mata):  # 11 01 11 01 11 10 00 01 00 10
    str1 = ""
    list1 = Odd
    list2 = Mata
    for i in range(len(list1)):
        a = list1[i] + list2[i]
        value = data_DNA(a)
        str1 = str1 + str(value)
    return str1

# 输入 文本，获得 数据频率  后七位数据  以及 对应十位均衡数据
def confirm_Seven_Pro_Ten_list(input_data):
    f = open(input_data)
    data = f.read()
    data = data.replace('\n', '')
    f.close()  # 关闭文件
    i = 0
    sumSep = 0
    # 将十进制转为7位二进制
    seq_Sev = []
    for x in range(0, 128):
        a = x
        a = bin(x)  # 将数据转化为二进制的函数
        a = a[2:]  # 从第三位开始取数据
        seq_Sev.append(a.zfill(7))
    a = [0] * 128
    while i <= len(data):
        for t in range(128):
            if seq_Sev[t] == str(data[i + 10: i + 17]):
                a[t] = int(a[t]) + 1
        i = i + 17
    dictionary = list()
    for i in range(len(seq_Sev)):
        dictionary.append([int(seq_Sev[i]), int(a[i])])
    dictionary.sort(key=lambda x: x[1], reverse=True)
    # 读取含有约束的均衡码
    with open("01Blance.txt", "r") as file:
        data = file.readlines()
    # 将01均衡码纳入列表中
    data1 = list()
    for i in range(len(data)):
        if i % 2 != 0:
            data1.append(int(data[i][:10]))
    for i in range(len(dictionary)):
        dictionary[i].append(data1[i])
    return dictionary


#  传参，将11个数据传过来，字符串类型
##  情况1， 出现连续四个的情况
#   情况2，出现连续三个0的情况
#   情况3  平常情况，进行删除就可以
#   传出的数据是正确数据（String）和索引（int）
def Insert_correct(Insert_String_Group ,dictionary):
    Insert_Group = ""
    Index_Correct = 0
    Index_000_Correct = list()
    String_Group_Cache_list = []
    Insert_Group = Insert_String_Group
    Correct_Group = []
    Index_Correct = []
    """
    开始  针对插入1的情况
    case 1：出现连续四个1的情况
    case 2：出现三个连续0的情况000
    case 3：任意情况
    """
    if Insert_Group.count("0") == 5:
        #  情况1， 出现连续四个的情况  1111
        for i in range(len(Insert_Group) - 3):
            if Insert_Group[i: i + 4] == "1111":
                # 三种情况判断
                w = 0
                for w in range(4):
                    if i == 0:
                        Correct_Group.append(Insert_Group[1: 11])
                        Index_Correct.append(i)
                    elif i == 7:
                        Correct_Group.append(Insert_Group[0: 10])
                        Index_Correct.append(i + w)
                    else:
                        Correct_Group.append(Insert_Group[0: i + w] + Insert_Group[i + w + 1: len(Insert_Group)])
                        Index_Correct.append(i + w)

                for j in range(len(Correct_Group)):
                    for q in range(len(dictionary)):
                        if int(Correct_Group[j]) == int(dictionary[q][2]) and dictionary[q][1] > 0:
                            String_Group_Cache_list.append([Index_Correct[j], Correct_Group[j], dictionary[q][1]])
                String_Group_Cache_list.sort(key=lambda x: x[2], reverse=True)
                #return String_Group_Cache_list[0][0], String_Group_Cache_list[0][1]
                if len(String_Group_Cache_list) != 0:
                    return String_Group_Cache_list[0][0], String_Group_Cache_list[0][1]
                if len(String_Group_Cache_list) == 0:
                    data_part = "1001100101"
                    data_MinIndex = 3
                    return data_MinIndex, data_part

        #  情况2    出现三个连续相等的0情况
        #  1. 判断有几个000情况   如何记住索引   2. 去除掉这些不合适的索引   3进一步排除，不满足概率和表格的   4  选取概率最大的输出
        #  Insert_Cache_list  储存所有1 索引的列表
        Insert_Cache_list = list()
        for i in range(len(Insert_Group)):
            if Insert_Group[i] == "1":
                Insert_Cache_list.append(i)
        for i in range(len(Insert_Group) - 2):
            if Insert_Group[i: i + 3] == "000":
                if i == 0 or i == 8:
                    Index_000_Correct.append(i)
                else:
                    Index_000_Correct.append(i - 1)
                    Index_000_Correct.append(i + 3)
        Index_String_list = list()
        t = []
        for i in range(len(Insert_Cache_list)):
            t.append(Insert_Cache_list[i])
            if t[i] == 0:
                String_Group_Cache_list.append(Insert_Group[1: 11])
            elif t[i] == 10:
                String_Group_Cache_list.append(Insert_Group[0: 10])
            else:
                String_Group_Cache_list.append(Insert_Group[0: t[i]] + Insert_Group[t[i] + 1: 11])
            Index_String_list.append([int(t[i]), String_Group_Cache_list[i]])
        #  进一步去除不满足概率 和 表格的字符串
        Pop = []
        for i in range(len(Index_String_list)):
            for j in range(len(Index_000_Correct)):
                if int(Index_String_list[i][0]) == int(Index_000_Correct[j]):
                    Pop.append(i)
        if len(Pop) == 1:
            Index_String_list.pop(Pop[0])
        if len(Pop) == 2:
            Index_String_list.pop(Pop[0])
            Index_String_list.pop(Pop[1] - 1)
        #  last_Index_String_Pro_List      最后包含索引  概率最大的数据   概率
        last_Index_String_Pro_List = list()
        #  进一步去除不满足概率 和 表格的字符串, 最后筛选，
        Prob = []
        ac = 0
        for i in range(len(Index_String_list)):
            for j in range(len(dictionary)):
                if int(Index_String_list[i][1]) == int(dictionary[j][2]) and dictionary[j][1] > 0:
                    Prob.append(dictionary[j][1])
                    last_Index_String_Pro_List.append([Index_String_list[i][0], Index_String_list[i][1], Prob[ac]])
                    ac = ac + 1
        #  根据概率的高低，选出最合适的，，并且输出删除位置的索引
        last_Index_String_Pro_List.sort(key=lambda x: x[2], reverse=True)
        
        
        if len(last_Index_String_Pro_List) != 0:
            Correct_Group = last_Index_String_Pro_List[0][1]
        
            Index_Correct = last_Index_String_Pro_List[0][0]
            return last_Index_String_Pro_List[0][0], last_Index_String_Pro_List[0][1]
        if len(last_Index_String_Pro_List) == 0:
            data_part = "1001100101"
            data_MinIndex = 3
            return data_MinIndex, data_part

    elif Insert_Group.count("0") == 5:
        #  情况1， 出现连续四个的情况
        for i in range(len(Insert_Group) - 3):
            if Insert_Group[i: i + 4] == "0000":
                # 三种情况判断
                w = 0
                for w in range(4):
                    if i == 0:
                        Correct_Group.append(Insert_Group[1: 11])
                        Index_Correct.append(i)
                    elif i == 7:
                        Correct_Group.append(Insert_Group[0: 10])
                        Index_Correct.append(i + w)
                    else:
                        Correct_Group.append(Insert_Group[0: i + w] + Insert_Group[i + w + 1: len(Insert_Group)])
                        Index_Correct.append(i + w)

                for j in range(len(Correct_Group)):
                    for q in range(len(dictionary)):
                        if int(Correct_Group[j]) == int(dictionary[q][2]) and dictionary[q][1] > 0:
                            String_Group_Cache_list.append([Index_Correct[j], Correct_Group[j], dictionary[q][1]])
                String_Group_Cache_list.sort(key=lambda x: x[2], reverse=True)
                if len(String_Group_Cache_list) != 0:
                    return String_Group_Cache_list[0][0], String_Group_Cache_list[0][1]
                if len(String_Group_Cache_list) == 0:
                    data_part = "1001100101"
                    data_MinIndex = 3
                    return data_MinIndex, data_part
        #  情况2    出现三个连续相等的0情况
        #  1. 判断有几个000情况   如何记住索引   2. 去除掉这些不合适的索引   3进一步排除，不满足概率和表格的   4  选取概率最大的输出
        #  Insert_Cache_list  储存所有1 索引的列表
        Insert_Cache_list = list()
        for i in range(len(Insert_Group)):
            if Insert_Group[i] == "0":
                Insert_Cache_list.append(i)
        for i in range(len(Insert_Group) - 2):
            if Insert_Group[i: i + 3] == "111":
                if i == 0 or i == 8:
                    Index_000_Correct.append(i)
                else:
                    Index_000_Correct.append(i - 1)
                    Index_000_Correct.append(i + 3)
        Index_String_list = list()
        t = []
        for i in range(len(Insert_Cache_list)):
            t.append(Insert_Cache_list[i])
            if t[i] == 0:
                String_Group_Cache_list.append(Insert_Group[1: 11])
            elif t[i] == 10:
                String_Group_Cache_list.append(Insert_Group[0: 10])
            else:
                Slipe = Insert_Group[0: t[i]] + Insert_Group[t[i] + 1: 11]
                String_Group_Cache_list.append(Slipe)
            Index_String_list.append([int(t[i]), String_Group_Cache_list[i]])
        #  进一步去除不满足概率 和 表格的字符串
        Index_String_list1 = Index_String_list
        Pop = []
        for i in range(len(Index_String_list)):
            for j in range(len(Index_000_Correct)):
                if int(Index_String_list[i][0]) == int(Index_000_Correct[j]):
                    Pop.append(i)
        if len(Pop) == 1:
            Index_String_list.pop(Pop[0])
        if len(Pop) == 2:
            Index_String_list.pop(Pop[0])
            Index_String_list.pop(Pop[1] - 1)
        #  last_Index_String_Pro_List      最后包含索引  概率最大的数据   概率
        last_Index_String_Pro_List = list()
        #  进一步去除不满足概率 和 表格的字符串, 最后筛选，
        Prob = []
        ac = 0
        for i in range(len(Index_String_list)):
            for j in range(len(dictionary)):
                if dictionary[j][1] > 0:
                    Prob.append(dictionary[j][1])
                    last_Index_String_Pro_List.append([Index_String_list[i][0], Index_String_list[i][1], Prob[ac]])
                    ac = ac + 1
        #  根据概率的高低，选出最合适的，，并且输出删除位置的索引
        last_Index_String_Pro_List.sort(key=lambda x: x[2], reverse=True)
        Correct_Group = last_Index_String_Pro_List[0][1]
        Index_Correct = last_Index_String_Pro_List[0][0]
        # return Index_Correct, Correct_Group
        if len(last_Index_String_Pro_List) != 0:
            # return last_Index_String_Pro_List[0][1], last_Index_String_Pro_List[0][0]
            return last_Index_String_Pro_List[0][0] , last_Index_String_Pro_List[0][1]
        if len(last_Index_String_Pro_List) == 0:
            data_part = "1001100101"
            data_MinIndex = 3
            return data_MinIndex, data_part
    else:
        Index_Correct = 3
        Correct_Group = "0011001011"
        return Index_Correct, Correct_Group


#    删除错误
# case 1： 出现连续五个相等的情况
# case 2： 出现四个连续相等的情况
# case 3： 正常的情况

def Delete_correct(Delete_String_Group ,dictionary ):
    Delete_Group = list()
    Delete_Group.append(Delete_String_Group)
    Correct_Group = []
    Index = []
    temp = 0
    Index_String_Pro_List = list()
    last_Index_String_Pro_List = list()
    Delete_Correct = 0
    Delete_000_Correct = list()
    String_Group_Cache_list = []

    Delete_Correct = []
    if Delete_Group[0].count("0") == 5:
        #  情况1， 出现连续五个的情况   #  (1)找出索引 以及可能正确的数据块
        for i in range(len(Delete_String_Group) - 4):
            if Delete_Group[0][i: i + 5] == "00000":
                temp = temp + 1
                Index_00000 = Delete_Group[0].index("00000")
                for j in range(2):
                    Correct_Group.append(Delete_Group[0][0:Index_00000 + 2 + j] + "1" + Delete_Group[0][
                                                                                        Index_00000 + 2 + j: len(
                                                                                            Delete_String_Group)])
                    Index.append(Index_00000 + 2 + j)
                    Index_String_Pro_List.append([Correct_Group[j], Index[j]])
                # 对数据块进行约束     构建可能争取字符快   索引  和 概率
        if temp > 0:
            for j in range(len(Index_String_Pro_List)):
                for k in range(len(dictionary)):
                    if int(Index_String_Pro_List[j][0]) == int(dictionary[k][2]) and dictionary[k][1] > 0:
                        last_Index_String_Pro_List.append(
                            [Index_String_Pro_List[j][0], Index_String_Pro_List[j][1], dictionary[k][1]])
            #   进行排序
            last_Index_String_Pro_List.sort(key=lambda x: x[2], reverse=True)
            if len(last_Index_String_Pro_List)  != 0:
                return last_Index_String_Pro_List[0][1], last_Index_String_Pro_List[0][0]
            if len(last_Index_String_Pro_List)  == 0:
                data_part = "1001100101"
                data_MinIndex = 3
                return data_MinIndex, data_part

        # 情况2，出现四个连续相同的0
        for i in range(len(Delete_String_Group) - 3):
            if Delete_Group[0][i: i + 4] == "0000":
                temp = temp + 1
                Index_00000 = Delete_Group[0].index("0000")
                for j in range(3):
                    Correct_Group.append(Delete_Group[0][0:Index_00000 + 1 + j] + "1" + Delete_Group[0][
                                                                                        Index_00000 + j + 1: len(
                                                                                            Delete_String_Group)])
                    Index.append(Index_00000 + 1 + j)
                    Index_String_Pro_List.append([Correct_Group[j], Index[j]])
                # 对数据块进行约束     构建可能争取字符快   索引  和 概率
        if temp > 0:
            for j in range(len(Index_String_Pro_List)):
                for k in range(len(dictionary)):
                    if int(Index_String_Pro_List[j][0]) == int(dictionary[k][2]) and dictionary[k][1] > 0:
                        last_Index_String_Pro_List.append([Index_String_Pro_List[j][0], Index_String_Pro_List[j][1],
                                                           dictionary[k][1]])
            last_Index_String_Pro_List.sort(key=lambda x: x[2], reverse=True)
            #return last_Index_String_Pro_List[0][1], last_Index_String_Pro_List[0][0]
            if len(last_Index_String_Pro_List)  != 0:
                return last_Index_String_Pro_List[0][1], last_Index_String_Pro_List[0][0]
            if len(last_Index_String_Pro_List)  == 0:
                data_part = "1001100101"
                data_MinIndex = 3
                return data_MinIndex, data_part

        # 情况三 ：无特殊情况
        Iedex_Case3 = []
        for i in range(len(Delete_String_Group)):
            if i == 0:
                Correct_Group.append("1" + Delete_String_Group)
                Iedex_Case3.append(i)
            elif i == 8:
                Correct_Group.append(Delete_String_Group + "1")
                Iedex_Case3.append(i + 1)
            else:
                Correct_Group.append(Delete_Group[0][0: i + 1] + "1" + Delete_Group[0][i + 1: len(Delete_Group[0])])
                Iedex_Case3.append(i + 1)
        for i in range(len(Correct_Group)):
            Index_String_Pro_List.append([Correct_Group[i], Iedex_Case3[i]])
            # 对数据块进行约束     构建可能争取字符快   索引  和 概率
        for i in range(len(Index_String_Pro_List)):
            for j in range(len(dictionary)):
#                if int(Index_String_Pro_List[i][0]) == dictionary[j][2] and dictionary[j][1] > 0:
                if dictionary[j][1] > 0:
                    last_Index_String_Pro_List.append([Index_String_Pro_List[i][0], Index_String_Pro_List[i][1],
                                                       dictionary[j][1]])
        #   进行排序
        last_Index_String_Pro_List.sort(key=lambda x: x[2], reverse=True)
        if len(last_Index_String_Pro_List) != 0:
            return last_Index_String_Pro_List[0][1], last_Index_String_Pro_List[0][0]
        if len(last_Index_String_Pro_List) == 0:
            data_part = "1001100101"
            data_MinIndex = 3
            return data_MinIndex, data_part

    elif Delete_Group[0].count("1") == 5:
        #  情况1， 出现连续五个的情况     (1)找出索引 以及可能正确的数据块
        for i in range(len(Delete_String_Group) - 4):
            if Delete_Group[0][i: i + 5] == "11111":
                temp = temp + 1
                Index_00000 = Delete_Group[0].index("11111")
                for j in range(2):
                    Correct_Group.append(Delete_Group[0][0:Index_00000 + 2 + j] + "0" + Delete_Group[0][
                                                                                        Index_00000 + 2 + j: len(
                                                                                            Delete_String_Group)])
                    Index.append(Index_00000 + 2 + j)
                    Index_String_Pro_List.append([Correct_Group[j], Index[j]])
                # 对数据块进行约束     构建可能争取字符快   索引  和 概率
        if temp > 0:
            for j in range(len(Index_String_Pro_List)):
                for k in range(len(dictionary)):
                    if int(Index_String_Pro_List[j][0]) == int(dictionary[k][2]) and dictionary[k][1] > 0:
                        last_Index_String_Pro_List.append(
                            [Index_String_Pro_List[j][0], Index_String_Pro_List[j][1], dictionary[k][1]])
            #   进行排序
            last_Index_String_Pro_List.sort(key=lambda x: x[2], reverse=True)
            if len(last_Index_String_Pro_List) != 0:
                return last_Index_String_Pro_List[0][1], last_Index_String_Pro_List[0][0]
            if len(last_Index_String_Pro_List) == 0:
                data_part = "1001100101"
                data_MinIndex = 3
                return data_MinIndex, data_part

        # 情况2，出现四个连续相同的0
        for i in range(len(Delete_String_Group) - 3):
            if Delete_Group[0][i: i + 4] == "1111":
                temp = temp + 1
                Index_00000 = Delete_Group[0].index("1111")
                for j in range(3):
                    Correct_Group.append(Delete_Group[0][0:Index_00000 + 1 + j] + "0" + Delete_Group[0][
                                                                                        Index_00000 + j + 1: len(
                                                                                            Delete_String_Group)])
                    Index.append(Index_00000 + 1 + j)
                    Index_String_Pro_List.append([Correct_Group[j], Index[j]])
                # 对数据块进行约束     构建可能争取字符快   索引  和 概率
        if temp > 0:
            for i in range(len(Index_String_Pro_List)):
                for j in range(len(dictionary)):
                    if dictionary[j][1] > 0:
                        last_Index_String_Pro_List.append(
                            [Index_String_Pro_List[i][0], Index_String_Pro_List[i][1], dictionary[j][1]])
            last_Index_String_Pro_List.sort(key=lambda x: x[2], reverse=True)
            if len(last_Index_String_Pro_List) != 0:
                return last_Index_String_Pro_List[0][1], last_Index_String_Pro_List[0][0]
            if len(last_Index_String_Pro_List) == 0:
                data_part = "1001100101"
                data_MinIndex = 3
                return data_MinIndex, data_part

        # 情况三 ：无特殊情况
        Iedex_Case3 = []
        for i in range(len(Delete_String_Group)):
            if i == 0:
                Correct_Group.append("0" + Delete_String_Group)
                Iedex_Case3.append(i)
            elif i == 8:
                Correct_Group.append(Delete_String_Group + "0")
                Iedex_Case3.append(i + 1)
            else:
                Correct_Group.append(Delete_Group[0][0: i + 1] + "0" + Delete_Group[0][i + 1: len(Delete_Group[0])])
                Iedex_Case3.append(i + 1)
        for i in range(len(Correct_Group)):
            Index_String_Pro_List.append([Correct_Group[i], Iedex_Case3[i]])
            # 对数据块进行约束     构建可能争取字符快   索引  和 概率
        for i in range(len(Index_String_Pro_List)):
            for j in range(len(dictionary)):
                if int(Index_String_Pro_List[i][0]) == int(dictionary[j][2]) and dictionary[j][1] > 0:
                    last_Index_String_Pro_List.append([Index_String_Pro_List[i][0], Index_String_Pro_List[i][1],
                                                       dictionary[j][1]])
        #   进行排序
        last_Index_String_Pro_List.sort(key=lambda x: x[2], reverse=True)
        if len(last_Index_String_Pro_List) != 0:
            return last_Index_String_Pro_List[0][1], last_Index_String_Pro_List[0][0]
        if len(last_Index_String_Pro_List) == 0:
            data_part = "1001100101"
            data_MinIndex = 3
            return data_MinIndex, data_part
    else:
        Index_Correct = 3
        Correct_Group = "0011001011"
        return Index_Correct, Correct_Group
"""
替换错误
传递     含有错误的10位二进制数据
返回值    当前索引（int）  纠正后的数据
"""
#   替换错误
def Substition_correct(Substition_String_Group ,dictionary):
    Substition_Group = list()
    Substition_Group.append(Substition_String_Group)
    Correct_Group = []
    temp = 0
    Index = []
    Index_000 = []
    Index_000_Cache = []
    Index_String_Pro_List = list()
    last_Index_String_Pro_List = list()

    if Substition_String_Group.count("0") == 5 and Substition_String_Group.count("1"):
        index = 0
        Substition_String_Group = "0011001011"
        return index,  Substition_String_Group

    if Substition_Group[0].count("0") == 6:
        # case 1 :出现连续六个相等的序列   0000000
        for i in range(len(Substition_Group[0]) - 5):
            if Substition_Group[0][i: i + 6] == "000000":
                temp = temp + 1
                Index_00000 = Substition_Group[0].index("000000")
                for j in range(2):
                    Correct_Group.append(Substition_Group[0][0:Index_00000 + 2 + j] + "1" + Substition_Group[0][
                                                                                            Index_00000 + 3 + j: len(
                                                                                                Substition_String_Group)])
                    Index.append(Index_00000 + 2 + j)
                    Index_String_Pro_List.append([Correct_Group[j], Index[j]])
        # 对数据块进行约束     构建可能争取字符快   索引  和 概率
        if temp > 0:
            for j in range(len(Index_String_Pro_List)):
                for k in range(len(dictionary)):
                    if int(Index_String_Pro_List[j][0]) == int(dictionary[k][2]) and dictionary[k][1] > 0:
                        last_Index_String_Pro_List.append([Index_String_Pro_List[j][0], Index_String_Pro_List[j][1],
                                                           dictionary[k][1]])
            #   进行排序
            last_Index_String_Pro_List.sort(key=lambda x: x[2], reverse=True)
            if len(last_Index_String_Pro_List)  != 0:
                return last_Index_String_Pro_List[0][1], last_Index_String_Pro_List[0][0]
            if len(last_Index_String_Pro_List)  == 0:
                data_part = "1001100101"
                data_MinIndex = 3
                return data_MinIndex, data_part

        # case 2 :出现五个连续的序列
        for i in range(len(Substition_Group[0]) - 4):
            if Substition_Group[0][i: i + 5] == "00000":
                temp = temp + 1
                Index_00000 = Substition_Group[0].index("00000")
                for j in range(3):
                    Correct_Group.append(Substition_Group[0][0:Index_00000 + 1 + j] + "1" + Substition_Group[0][
                                                                                            Index_00000 + 2 + j: len(
                                                                                                Substition_String_Group)])
                    Index.append(Index_00000 + 1 + j)
                    Index_String_Pro_List.append([Correct_Group[j], Index[j]])
        # 对数据块进行约束     构建可能争取字符快   索引  和 概率
        if temp > 0:
            for j in range(len(Index_String_Pro_List)):
                for k in range(len(dictionary)):
                    if int(Index_String_Pro_List[j][0]) == int(dictionary[k][2]) and dictionary[k][1] > 0:
                        last_Index_String_Pro_List.append([Index_String_Pro_List[j][0], Index_String_Pro_List[j][1],
                                                           dictionary[k][1]])
            #   进行排序
            last_Index_String_Pro_List.sort(key=lambda x: x[2], reverse=True)
            if len(last_Index_String_Pro_List) != 0:
                return last_Index_String_Pro_List[0][1], last_Index_String_Pro_List[0][0]
            if len(last_Index_String_Pro_List) == 0:
                data_part = "1001100101"
                data_MinIndex = 3
                return data_MinIndex, data_part

        # case 3 :出现4个连续的序列  0000    会出现三种 情况
        #         （1）索引等于0的时候
        #         （2）索引等于6的时候
        #         （3）索引不在两边的时候

        for i in range(len(Substition_Group[0]) - 3):
            if Substition_Group[0][i: i + 4] == "0000":
                temp = temp + 1
                Index_00000 = Substition_Group[0].index("0000")
                for j in range(4):
                    Correct_Group.append(Substition_Group[0][0: Index_00000 + j] + "1" + Substition_Group[0][
                                                                                         Index_00000 + j + 1: len(
                                                                                             Substition_Group[0])])
                    Index.append(Index_00000 + j)
                for k in range(4):
                    Index_String_Pro_List.append([Correct_Group[k], Index[k]])
            # 对数据块进行约束     构建可能争取字符快   索引  和 概率
        if temp > 0:
            for j in range(len(Index_String_Pro_List)):
                for k in range(len(dictionary)):
                    if int(Index_String_Pro_List[j][0]) == int(dictionary[k][2]) and dictionary[k][1] > 0:
                        last_Index_String_Pro_List.append([Index_String_Pro_List[j][0], Index_String_Pro_List[j][1],
                                                           dictionary[k][1]])
                #   进行排序
            last_Index_String_Pro_List.sort(key=lambda x: x[2], reverse=True)
            if len(last_Index_String_Pro_List) != 0:
                return last_Index_String_Pro_List[0][1], last_Index_String_Pro_List[0][0]
            if len(last_Index_String_Pro_List) == 0:
                data_part = "1001100101"
                data_MinIndex = 3
                return data_MinIndex, data_part

        # case 4: 出现000连续,以及普通情况进行排除
        #         可以判定000左右两个位置一定是 正确的
        #         （1） 找出所有索引的位置
        #         （2） 去除掉000左右的索引位置
        for i in range(len(Substition_String_Group)):
            if Substition_String_Group[i] == "0":
                if i == 0:
                    Correct_Group.append("1" + Substition_String_Group[1: len(Substition_String_Group)])
                    Index.append(i)
                elif i == 9:
                    Correct_Group.append(Substition_String_Group[0: len(Substition_String_Group) - 1] + "1")
                    Index.append(i)
                else:
                    Correct_Group.append(Substition_String_Group[0: i] + "1" + Substition_String_Group[
                                                                               i + 1: len(Substition_String_Group)])
                    Index.append(i)
        for i in range(len(Index)):
            Index_String_Pro_List.append([Correct_Group[i], Index[i]])
        #  对出现000的情况进行排除
        for i in range(len(Substition_String_Group) - 2):
            if Substition_String_Group[i: i + 3]:
                Index_000.append("111")
        #  (1) 当索引在两端是
        if Index_000 == 0:
            Index_000_Cache.append(i + 3)
        elif Index_000 == 7:
            Index_000_Cache.append(i - 1)
        else:
            Index_000_Cache.append(i - 1)
            Index_000_Cache.append(i + 3)
        #        for i in range(len(Index_000)):
        # 对数据块进行约束     构建可能争取字符快   索引  和 概率
        for j in range(len(Index_String_Pro_List)):
            for k in range(len(dictionary)):
                for w in range(len(Index_000_Cache)):
                    if int(Index_String_Pro_List[j][0]) == int(dictionary[k][2]) and dictionary[k][1] > 0 and \
                            Index_String_Pro_List[j][1] != Index_000_Cache[w]:
                        last_Index_String_Pro_List.append([Index_String_Pro_List[j][0], Index_String_Pro_List[j][1],
                                                           dictionary[k][1]])
        #   进行排序
        last_Index_String_Pro_List.sort(key=lambda x: x[2], reverse=True)
        if len(last_Index_String_Pro_List) != 0:
            return last_Index_String_Pro_List[0][1], last_Index_String_Pro_List[0][0]
        if len(last_Index_String_Pro_List) == 0:
            data_part = "1001100101"
            data_MinIndex = 3
            return data_MinIndex, data_part
    elif Substition_Group[0].count("1") == 6:
        # case 1 :出现连续六个相等的序列   111111
        for i in range(len(Substition_Group[0]) - 5):
            if Substition_Group[0][i: i + 6] == "111111":
                temp = temp + 1
                Index_00000 = Substition_Group[0].index("111111")
                for j in range(2):
                    Correct_Group.append(Substition_Group[0][0:Index_00000 + 2 + j] + "0" + Substition_Group[0][
                                                                                            Index_00000 + 3 + j: len(
                                                                                                Substition_String_Group)])
                    Index.append(Index_00000 + 2 + j)
                    Index_String_Pro_List.append([Correct_Group[j], Index[j]])
        # 对数据块进行约束     构建可能争取字符快   索引  和 概率
        if temp > 0:
            for j in range(len(Index_String_Pro_List)):
                for k in range(len(dictionary)):
                    if int(Index_String_Pro_List[j][0]) == int(dictionary[k][2]) and dictionary[k][1] > 0:
                        last_Index_String_Pro_List.append([Index_String_Pro_List[j][0], Index_String_Pro_List[j][1],
                                                           dictionary[k][1]])
            #   进行排序
            last_Index_String_Pro_List.sort(key=lambda x: x[2], reverse=True)
            # return last_Index_String_Pro_List[0][1], last_Index_String_Pro_List[0][0]
            if len(last_Index_String_Pro_List) != 0:
                return last_Index_String_Pro_List[0][1], last_Index_String_Pro_List[0][0]
            if len(last_Index_String_Pro_List) == 0:
                data_part = "1001100101"
                data_MinIndex = 3
                return data_MinIndex, data_part
        # case 2 :出现五个连续的序列
        for i in range(len(Substition_Group[0]) - 4):
            if Substition_Group[0][i: i + 5] == "11111":
                temp = temp + 1
                Index_00000 = Substition_Group[0].index("11111")
                for j in range(5):
                    Correct_Group.append(Substition_Group[0][0:Index_00000 + j] + "0" + Substition_Group[0][
                                                                                        Index_00000 + 1 + j: len(
                                                                                            Substition_String_Group)])
                    Index.append(Index_00000 + 1 + j)
                    Index_String_Pro_List.append([Correct_Group[j], Index[j]])
        # 对数据块进行约束     构建可能争取字符快   索引  和 概率
        if temp > 0:
            for j in range(len(Index_String_Pro_List)):
                for k in range(len(dictionary)):
                    str1 = str(dictionary[k][2]).zfill(10)
                    if int(Index_String_Pro_List[j][0]) == int(str1) and dictionary[k][1] > 0:
                        last_Index_String_Pro_List.append([Index_String_Pro_List[j][0], Index_String_Pro_List[j][1],
                                                           dictionary[k][1]])
            #   进行排序
            last_Index_String_Pro_List.sort(key=lambda x: x[2], reverse=True)
            #for i in range(len(last_Index_String_Pro_List)):
            #    print(last_Index_String_Pro_List[i])
            if len(last_Index_String_Pro_List) != 0:
                return last_Index_String_Pro_List[0][1], last_Index_String_Pro_List[0][0]
            if len(last_Index_String_Pro_List) == 0:
                data_part = "1001100101"
                data_MinIndex = 3
                return data_MinIndex, data_part
        # case 3 :出现4个连续的序列  1111    会出现三种 情况
        #         （1）索引等于0的时候
        #         （2）索引等于6的时候
        #         （3）索引不在两边的时候
        for i in range(len(Substition_Group[0]) - 3):
            if Substition_Group[0][i: i + 4] == "1111":
                temp = temp + 1

                # Index_00000  出现连续四个1111的索引位置
                Index_00000 = Substition_Group[0].index("1111")
                if Index_00000 == 0:
                    for k in range(4):
                        Correct_Group.append(Substition_Group[0][0: k + Index_00000] + "0" + Substition_Group[0][
                                                                                         Index_00000 + 1 + k: len(
                                                                                             Substition_String_Group)])
                        Index.append(Index_00000 + k)


                elif Index_00000 == 6:
                    for k in range(3):
                        Correct_Group.append(Substition_Group[0][0: k + Index_00000] + "0" + Substition_Group[0][
                                                                                             Index_00000 + 1 + k: len(
                                                                                                 Substition_String_Group)])
                        Index.append(Index_00000 + k)
                    Correct_Group.append(Substition_Group[0][0: len(Substition_String_Group) - 1] + "0")
                    Index.append(9)
                else:
                    for k in range(4):
                        Correct_Group.append(Substition_Group[0][0: k + Index_00000] + "0" + Substition_Group[0][
                                                                                             Index_00000 + 1 + k: len(
                                                                                                 Substition_String_Group)])
                        Index.append(Index_00000 + k)

                for k in range(4):
                    Index_String_Pro_List.append([Correct_Group[k], Index[k]])
            # 对数据块进行约束     构建可能争取字符快   索引  和 概率
        if temp > 0:
            for j in range(len(Index_String_Pro_List)):
                for k in range(len(dictionary)):
                    if int(Index_String_Pro_List[j][0]) == int(dictionary[k][2]) and dictionary[k][1] > 0:
                        last_Index_String_Pro_List.append([Index_String_Pro_List[j][0], Index_String_Pro_List[j][1],
                                                           dictionary[k][1]])
                #   进行排序
            last_Index_String_Pro_List.sort(key=lambda x: x[2], reverse=True)
            if len(last_Index_String_Pro_List) != 0:
                return last_Index_String_Pro_List[0][1], last_Index_String_Pro_List[0][0]
            if len(last_Index_String_Pro_List) == 0:
                data_part = "1001100101"
                data_MinIndex = 3
                return data_MinIndex, data_part

        # case 4: 出现000连续,以及普通情况进行排除
        #         可以判定000左右两个位置一定是 正确的
        #         （1） 找出所有索引的位置
        #         （2） 去除掉000左右的索引位置

        for i in range(len(Substition_String_Group)):
            if Substition_String_Group[i] == "1":
                if i == 0:
                    Correct_Group.append("0" + Substition_String_Group[1: len(Substition_String_Group)])
                    Index.append(i)
                elif i == 9:
                    Correct_Group.append(Substition_String_Group[0: len(Substition_String_Group) - 1] + "0")
                    Index.append(i)
                else:
                    Correct_Group.append(Substition_String_Group[0: i] + "0" + Substition_String_Group[
                                                                               i + 1: len(Substition_String_Group)])
                    Index.append(i)
        for i in range(len(Index)):
            Index_String_Pro_List.append([Correct_Group[i], Index[i]])
        #  对出现000的情况进行排除
        for i in range(len(Substition_String_Group) - 2):
            if Substition_String_Group[i: i + 3]:
                Index_000.append("000")
        #  (1) 当索引在两端是
        if Index_000 == 0:
            Index_000_Cache.append(i + 3)
        elif Index_000 == 7:
            Index_000_Cache.append(i - 1)
        else:
            Index_000_Cache.append(i - 1)
            Index_000_Cache.append(i + 3)
        #        for i in range(len(Index_000)):
        # 对数据块进行约束     构建可能争取字符快   索引  和 概率
        for j in range(len(Index_String_Pro_List)):
            for k in range(len(dictionary)):
                for w in range(len(Index_000_Cache)):
                    if int(Index_String_Pro_List[j][0]) == int(dictionary[k][2]) and dictionary[k][1] > 0 and \
                            Index_String_Pro_List[j][1] != Index_000_Cache[w]:
                        last_Index_String_Pro_List.append([Index_String_Pro_List[j][0], Index_String_Pro_List[j][1],
                                                           dictionary[k][1]])
        #   进行排序
        last_Index_String_Pro_List.sort(key=lambda x: x[2], reverse=True)
        '''
        for i in range(len(last_Index_String_Pro_List)):
            print(last_Index_String_Pro_List[i])
        '''
        if len(last_Index_String_Pro_List) != 0:
            return last_Index_String_Pro_List[0][1], last_Index_String_Pro_List[0][0]
        if len(last_Index_String_Pro_List) == 0:
            data_part = "1001100101"
            data_MinIndex = 3
            return data_MinIndex, data_part
    else:
        data_part  = "1001100101"
        data_MinIndex = 3
        return data_MinIndex , data_part

#  插入和删除错误
"""
判断函数
传递参数 ：块的索引（int） 偶数组数据（Odd_MinGroup(list)   list_Group(String)）  奇数组数据（Mate_MinGroup    list   list_Group(String)）    dictionary字典  list里面是（int）
逻辑判断：根据 i的值，向下漂移五位，判断是否出错，并进行讨论
返回参数： 1）判断当前数据时候正确（2）判断索引下时候还有6位  （3）向下漂移五位是否正确
"""
def Judge_Group(index, Odd_MinGroup, Mate_MinGroup, dictionary):
    Odd_MinGroup_String = list()
    Mate_MinGroup_String = list()
    Current_Bool = False
    Traver_Bool = True
    MinGroup_Cache1 = []
    MinGroup_Cache2 = []
    Count = 0
    Count1 = 0
    Count2 = 0

    if index == 15:
        Current_Bool = True
        Traver_Bool = True
        return  Current_Bool , Traver_Bool
    if index == 14 and len(Odd_MinGroup[14]) <= 9:
        Current_Bool = True
        Traver_Bool = True
        return Current_Bool, Traver_Bool
    for i in range(len(Odd_MinGroup)):
        Odd_MinGroup_String.append(str(Odd_MinGroup[i]))

    for i in range(len(Mate_MinGroup)):
        Mate_MinGroup_String.append(str(Mate_MinGroup[i]))

    for i in range(len(dictionary)):
        if int(Odd_MinGroup_String[index]) == int(dictionary[i][2]) and dictionary[i][1] > 0:
            Current_Bool = True

    #  初步判断
    if index + 6 < len(Odd_MinGroup) - 1:
        for i in range(6):
            for j in range(len(dictionary)):
                dictionary1 = dictionary[j][2]
                dictionary2 = Odd_MinGroup_String[index + i]
                dictionary1_type = type(dictionary[j][2])
                dictionary__1 = dictionary[j][1]
                if Odd_MinGroup_String[index + i] == dictionary[j][2] and dictionary[j][1] > 0:
                    Count = Count + 1
        #  left   漂移
        for i in range(5):
            MinGroup_Cache1.append(Odd_MinGroup[index + i + 1][1:10] + Odd_MinGroup[index + i + 2][0:1])
            MinGroup_Cache2.append(Odd_MinGroup[index + i][9:10] + Odd_MinGroup[index + i + 1][0:9])
        for i in range(len(MinGroup_Cache1)):
            for j in range(len(dictionary)):
                if MinGroup_Cache1[i] == dictionary[j][2] and dictionary[j][1] > 0:
                    Count1 = Count1 + 1
        for i in range(len(MinGroup_Cache2)):
            for j in range(len(dictionary)):
                if int(MinGroup_Cache2[i]) == dictionary[j][2] and dictionary[j][1] > 0:
                    Count2 = Count2 + 1
        last_length_insert = len(Odd_MinGroup_String[14])
        last_length_delete = len(Odd_MinGroup_String)
        if (Count < 5  and Count1 == 5 and last_length_insert >= 10)  or (Count < 5  and Count2 == 5 and last_length_delete <= 15) :
            Traver_Bool = False

    # case 2   移位
    elif index == len(Odd_MinGroup) - 1:
        Traver_Bool = True
    else:
        Index_Cache = len(Odd_MinGroup) - index
        for i in range(Index_Cache):
            for j in range(len(dictionary)):
                if int(Odd_MinGroup_String[index + i]) == dictionary[j][2] and dictionary[j][1] > 0:
                    Count = Count + 1
        for i in range(Index_Cache - 2):
            MinGroup_Cache1.append(Odd_MinGroup[index + i + 1][1:10] + Odd_MinGroup[index + i + 2][0:1])
            MinGroup_Cache2.append(Odd_MinGroup[index + i][9:10] + Odd_MinGroup[index + i + 1][0:9])
        for i in range(len(MinGroup_Cache1)):
            for j in range(len(dictionary)):
                if int(MinGroup_Cache1[i]) == dictionary[j][2] and dictionary[j][1] > 0:
                    Count1 = Count1 + 1
        # right  漂移
        for i in range(len(MinGroup_Cache2)):
            for j in range(len(dictionary)):
                if int(MinGroup_Cache2[i]) == dictionary[j][2] and dictionary[j][1] > 0:
                    Count2 = Count2 + 1
        last_length_insert = len(Odd_MinGroup_String[14])
        last_length_delete = len(Odd_MinGroup_String)
        last_data_list = len(Odd_MinGroup_String[-1])
        # if (Count < Index_Cache - 2  and  Count1 >= Index_Cache - 2 and last_length_insert >= 10) or (Count < Index_Cache - 2  and  Count2 >= Index_Cache - 2 and last_length_delete <= 15):
        if (Count <= Index_Cache - 2 and Count1 >= Index_Cache - 2 and last_length_insert >= 10) or (Count <= Index_Cache - 2 and Count2 >= Index_Cache - 2 and last_length_delete <= 15):
            Traver_Bool = False
            if index >= 13 and last_data_list == 10 and last_length_delete == 15:
                Traver_Bool = True
        if index == 14:
            Traver_Bool = True
    return Current_Bool, Traver_Bool

"""
#判断出是插入或者删除的之后的一种操作流程
#传递参数 ：块的索引（int） 偶数组数据（Odd_MinGroup(list)   list_Group(String)）  奇数组数据（Mate_MinGroup    list   list_Group(String)）    dictionary字典  list里面是（int）
#传出数据：奇偶两个列表  以及当前的索引        奇数组   偶数组   当前索引
"""
def Insert_Delete_correct(index, Odd_MinGroup, Mate_MinGroup, dictionary):
    t = 0
    Count = 0
    Index_MinGroup = 0
    Correct_Group = ""
    Odd_MinGroup_String = list()
    Mate_MinGroup_String = list()
    MinGroup_Cache = list()
    MinGroup_Del_Cache = list()
    Mate_MinGroup_String_last = list()
    Mate_MinGroup_Det_Cache = list()
    for i in range(len(Odd_MinGroup)):
        Odd_MinGroup_String.append(str(Odd_MinGroup[i]))
    for i in range(len(Mate_MinGroup)):
        Mate_MinGroup_String.append(str(Mate_MinGroup[i]))
    # 判断他是插入错误,且错误位置不靠近数据末端
    if index + 6 < math.ceil(len(Mate_MinGroup)):
        if t == 0:
            # 插入的判断生成条件       返回值索引和纠正后的数据
            for i in range(5):
                MinGroup_Cache.append(Mate_MinGroup[index + i + 1][1:10] + Mate_MinGroup[index + i + 2][0:1])
            for i in range(len(MinGroup_Cache)):
                for j in range(len(dictionary)):
                    if int(MinGroup_Cache[i]) == int(dictionary[j][2]) and dictionary[j][1] > 0:
                        Count = Count + 1
            last_length = len(Odd_MinGroup[14])
            # 迁移位置
            if Count >= 4 and last_length >= 10 and last_length >= 10:
                Index_MinGroup, Correct_Group = Insert_correct(Mate_MinGroup[index] + Mate_MinGroup[index + 1][0: 1] ,dictionary)
                Mate_MinGroup_String[index] = Correct_Group
                for Count1 in range(len(Mate_MinGroup) - 2 - index):
                    Mate_MinGroup_String[Count1 + int(index) + 1] = Mate_MinGroup[Count1 + int(index) + 1][1: 10] + Mate_MinGroup[Count1 + int(index) + 2][0: 1]
                if len(Mate_MinGroup_String) == 16:
                    Mate_MinGroup_String.pop()
                #  对奇数据组进行插入数据错误进行修改
                if index == 0:
                #if Index_MinGroup == 0:
                    Odd_MinGroup_String[index] = Odd_MinGroup[index][1: 10] + Odd_MinGroup[index + 1][
                                                                                     0: 1]
                elif index == 9:
                #elif Index_MinGroup == 9:
                    Odd_MinGroup_String[index] = Odd_MinGroup[index][0: 9] + Odd_MinGroup[index + 1][0: 1]
                else:
                    Odd_MinGroup_String[index] = Odd_MinGroup[index][0: Index_MinGroup] + Odd_MinGroup[index][
                                                                                                 Index_MinGroup: 10] + \
                                                 Odd_MinGroup[index + 1][0: 1]
                for Count1 in range(math.ceil(len(Odd_MinGroup)) - 1):
                    Odd_MinGroup_String[Count1 ] = Odd_MinGroup[Count1 ][1: 10] + Odd_MinGroup[Count1 + 1][0: 1]
                if len(Odd_MinGroup_String) == 16:
                    Odd_MinGroup_String.pop()
            else:
                t = 1
        if t == 1 :
            t == 1
            # 删除的判断生成条件       返回值索引和纠正后的数据
            Index_MinGroup, Correct_Group = Delete_correct(Mate_MinGroup[index][0: 9] ,dictionary)
            # 假设错误是删除
            MinGroup_Cache = []
            for i in range(5):
                MinGroup_Cache.append(Mate_MinGroup[index + i][9:10] + Mate_MinGroup[index + i + 1][0:9])
            Count2 = 0
            for i in range(len(MinGroup_Cache)):
                for j in range(len(dictionary)):
                    if int(MinGroup_Cache[i]) == int(dictionary[j][2]) and dictionary[j][1] > 0:
                        Count2 = Count2 + 1
            if Count2 >= 5:
                Mate_MinGroup_String[index] = Correct_Group
                Count3 = int(index)
                for Count3 in range(math.ceil(len(Mate_MinGroup)) - 1):
                    Mate_MinGroup_String[Count3 + 1] = Mate_MinGroup[Count3][9: 10] + Mate_MinGroup[Count3 + 1][0: 9]
#                    Mate_MinGroup_String.pop()
                #  对奇数据组进行插入数据错误进行修改
                #   对当前错误进行改正
                if Index_MinGroup == 0:
                    Odd_MinGroup_String[index] = "0" + Odd_MinGroup[index][0: 9]
                elif Index_MinGroup == 9:
                    Odd_MinGroup_String[index] = Odd_MinGroup[index][0: 9] + "0"
                else:
                    Odd_MinGroup_String[index] = Odd_MinGroup[index][0: Index_MinGroup] + "0" + \
                                                 Odd_MinGroup[index][Index_MinGroup: 9]
                # 对后面造成数据飘逸的进行改正
                Count4 = index
                for  Count4  in range(math.ceil(len(Mate_MinGroup) - 1 )):
                    Odd_MinGroup_String[Count4 + 1] = Odd_MinGroup[Count4][9: 10] + Odd_MinGroup[Count4 + 1][0: 9]
    #   遍历不到6个的情况
    if index + 6 >= math.ceil(len(Mate_MinGroup)):
        if t == 0:
            # 插入的判断生成条件       返回值索引和纠正后的数据
            #  Index_MinGroup, Correct_Group = Insert_correct(Mate_MinGroup[index][0: 10] + Mate_MinGroup[index + 1][0: 1])
            # 假设错误是插入
            for i in range(math.ceil(len(Mate_MinGroup)) - index - 2):
                MinGroup_Cache.append(Mate_MinGroup[index + i + 1][1:10] + Mate_MinGroup[index + i + 2][0:1])
            for i in range(len(MinGroup_Cache)):
                for j in range(len(dictionary)):
                    if int(MinGroup_Cache[i]) == int(dictionary[j][2]) and dictionary[j][1] > 0:
                        Count = Count + 1

            last_length = len(Odd_MinGroup[14])
            if Count >= math.ceil(len(Mate_MinGroup)) - index - 2  and last_length >= 10:
                Index_MinGroup, Correct_Group = Insert_correct(Mate_MinGroup[index][0: 10] + Mate_MinGroup[index + 1][0: 1] ,dictionary)
                Mate_MinGroup_String[index] = Correct_Group
                for Count1 in range(math.ceil(len(Mate_MinGroup)) - 2 - index):
                    Mate_MinGroup_String[Count1 + int(index) + 1] = Mate_MinGroup[Count1 + int(index) + 1][
                                                                    1: 10] + Mate_MinGroup[Count1 + int(index) + 2][0: 1]
                if len(Mate_MinGroup_String) == 16:
                    Mate_MinGroup_String.pop()

                #  对奇数据组进行插入数据错误进行修改
                if Index_MinGroup == 0:
                    Odd_MinGroup_String[index] = Odd_MinGroup[index][1: 10] + Odd_MinGroup[index + 1][0: 1]
                elif Index_MinGroup == 9:
                    Odd_MinGroup_String[index] = Odd_MinGroup[index][0: 9] + Odd_MinGroup[index + 1][0: 1]
                else:
                    Odd_MinGroup_String[index] = Odd_MinGroup[index][0: Index_MinGroup] + Odd_MinGroup[index][Index_MinGroup + 1: 10] + Odd_MinGroup[index + 1][0: 1]
                for Count1 in range(math.ceil(len(Odd_MinGroup)) - 2 - index):
                    Odd_MinGroup_String[Count1+ int(index) + 1] = Odd_MinGroup[Count1+ int(index) + 1][1: 10] + Odd_MinGroup[Count1 + index + 2][0: 1]
                if len(Odd_MinGroup_String) == 16:
                        Odd_MinGroup_String.pop()
            else:
                t = 1
        if t == 1:
            # 删除的判断生成条件       返回值索引和纠正后的数据
            Index_MinGroup, Correct_Group = Delete_correct(Mate_MinGroup[index][0: 9] ,dictionary)
            # 假设错误是删除
            for i in range(len(Mate_MinGroup) - 1 - index):
                MinGroup_Cache.append(Mate_MinGroup[index + i][9:10] + Mate_MinGroup[index + i + 1][0:9])
                # MinGroup_Cache = Mate_MinGroup[index + i][9:10] + Mate_MinGroup[index + i + 1][0:9]
            for i in range(len(MinGroup_Cache)):
                for j in range(len(dictionary)):
                    if int(MinGroup_Cache[i]) == int(dictionary[j][2]) and dictionary[j][1] > 0:
                        Count = Count + 1
            if Count >= len(Mate_MinGroup) - 1 - index:
                Mate_MinGroup_String[index] = Correct_Group
                Count1 = int(index)
#                for Count1 in range(math.ceil(len(Mate_MinGroup))):
                for Count1 in range(math.ceil(len(Mate_MinGroup)) - 1):
                    Mate_MinGroup_String[Count1 + 1] = Mate_MinGroup[Count1][9: 10] + Mate_MinGroup[Count1 + 1][0: 9]
#                    Mate_MinGroup_String.pop()
                    #  对奇数据组进行插入数据错误进行修改
                    #   对当前错误进行改正
                if Index_MinGroup == 0:
                    Odd_MinGroup_String[index] = "0" + Odd_MinGroup[index][0: 9]
                elif Index_MinGroup == 9:
                    Odd_MinGroup_String[index] = Odd_MinGroup[index][0: 9] + "0"
                else:
                    Odd_MinGroup_String[index] = Odd_MinGroup[index][0: Index_MinGroup] + "0" + \
                                                 Odd_MinGroup[index][Index_MinGroup: 9]
                for Count1 in range(math.ceil(len(Mate_MinGroup) ) - 1):
                    Odd_MinGroup_String[Count1 + 1] = Odd_MinGroup[Count1][9: 10] + Odd_MinGroup[Count1 + 1][0: 9]
    return Mate_MinGroup_String, Odd_MinGroup_String, index, t


def drift_correct(DnaString_Maxgroup, dictionary):
    Odd_MaxGroup = []
    Mate_MaxGroup = []
    # 将数据分成齐祖打印
    for count_list in range(len(DnaString_Maxgroup)):
        Odd = ""
        Mate = ""
        #  step 1 将数据   转为二进制数据，并分为奇偶两部分，只针对与150个碱基       目前只针对于一组
        DnaString_BitGroup = convert_dna_to_data_sequence(DnaString_Maxgroup[count_list])
        # step 2   (1) 将数据分成奇偶两部分并进行分成小组
        for i in range(len(DnaString_BitGroup)):
            if i % 2 == 0:
                Odd = Odd + DnaString_BitGroup[i]
            else:
                Mate = Mate + DnaString_BitGroup[i]
        # step 3   (2) 将数据分割成10个一小组，Mate_MinGroup     Odd_MinGroup(list   里面是String)
        Mate_MinGroup = list()
        for j in range(0, len(Mate), 10):
            Mate1_MinGroup = Mate[j:j + 10]
            Mate_MinGroup.append(Mate1_MinGroup)
        Odd_MinGroup = []
        for j in range(0, len(Odd), 10):
            Odd1_MinGroup = Odd[j:j + 10]
            Odd_MinGroup.append(Odd1_MinGroup)

        # step 4   case 1 完全正确      （1） 比对数据正确     （2）判断索引块的位置，距离边界大于四，正常判断，小于四，按照最大长度判断  （3） 正确 返回值String类型  整列  返回Mate 和 Odd
        #          case 2 数据正确      （1）同上 （2）判断索引长度，距离边界大于四，正常判断，小于四，按照最大长度判断    （3）出现错误，进行插入或者删除纠错，返回Mate 和 Odd
        #          case 3 出现错误      （1）出现错误         （2）判断索引长度，距离边界大于四，正常判断，小于四，按照最大长度判断    （3） 正确    对对当前数据进行纠错
        #          case 4 出现错误      （1）出现错误         （2）判断索引长度，距离边界大于四，正常判断，小于四，按照最大长度判断    （3） 错误    进行插入或者删除纠错
        #          Judge_Group(index , Odd_MinGroup , Mate_MinGroup , dictionary):
        #          Insert_Delete_correct(index , Odd_MinGroup , Mate_MinGroup , dictionary):    return Odd_MinGroup_String , Mate_MinGroup_String , index
        #          传递参数时是先偶数  后奇数   接受数据时是   先奇数后 偶数
        Insert_Count = 0
        i = 0
        if count_list <= len(DnaString_Maxgroup) - 1 and len(DnaString_Maxgroup[count_list]) > 140:
            while i < 15:
                Insert_Count = 0
                # 调用判断函数做基本判断
                Current_Bool, Traver_Bool = Judge_Group(i, Odd_MinGroup, Mate_MinGroup, dictionary)
                if Current_Bool == True:
                    #  出现插入和删除错误    分出是插入还是删除错误
                    if Traver_Bool == False:
                        #  插入错误
                        Odd_MinGroup1, Mate1_MinGroup, index, t = Insert_Delete_correct(i, Mate_MinGroup, Odd_MinGroup,
                                                                                        dictionary)
                        if t == 0:
                            j = i
                            for j in range(len(Mate1_MinGroup)):
                                Odd_MinGroup[j] = Odd_MinGroup1[j]
                                Mate_MinGroup[j] = Mate1_MinGroup[j]
                            if len(Odd_MinGroup) == 16:
                                Mate_MinGroup.pop()
                                Odd_MinGroup.pop()
                            Insert_Count = Insert_Count + 1
                        else:
                            for j in range(len(Mate1_MinGroup)):
                                Odd_MinGroup[j] = Odd_MinGroup1[j]
                                Mate_MinGroup[j] = Mate1_MinGroup[j]
                if Current_Bool == False:
                    # 替换错误
                    if Traver_Bool == True:
                        index, Odd_MinGroup[i] = Substition_correct(Odd_MinGroup[i] ,dictionary)
                    # 插入删除错误
                    else:
                        Odd1_MinGroup, Mate1_MinGroup, index, t = Insert_Delete_correct(i, Mate_MinGroup, Odd_MinGroup,
                                                                                        dictionary)
                        if t == 0:
                            for j in range(len(Mate1_MinGroup)):
                                Odd_MinGroup[j] = Odd1_MinGroup[j]
                                Mate_MinGroup[j] = Mate1_MinGroup[j]
                                # 改动的地方
                            if len(Odd_MinGroup) == 16:
                                Mate_MinGroup.pop()
                                Odd_MinGroup.pop()
                        else:
                            for j in range(len(Mate1_MinGroup)):
                                Odd_MinGroup[j] = Odd1_MinGroup[j]
                                Mate_MinGroup[j] = Mate1_MinGroup[j]
                i = i + 1
            last_length = 0
            if len(Odd_MinGroup[-1]) < 10:
                last_length = last_length + 1
            Odd_MaxGroup.append(Odd_MinGroup)
            Mate_MaxGroup.append(Mate_MinGroup)
    MaxGroup = []
    for i in range(len(Odd_MaxGroup)):
        Odd_cache = ""
        Mate_cache = ""
        for j in range(len(Odd_MaxGroup[i])):
            test = str(Odd_MaxGroup[i][j])
            Odd_cache = Odd_cache + str(Odd_MaxGroup[i][j])
            Mate_cache = Mate_cache + str(Mate_MaxGroup[i][j])
        MaxGroup.append(convert_dna_to_data_sequence1(Odd_cache, Mate_cache))
    if len(DnaString_Maxgroup[-1]) < 140:
        MaxGroup.append(DnaString_Maxgroup[-1])
    return MaxGroup




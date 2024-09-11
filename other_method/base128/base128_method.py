from PIL import Image
import random
import numpy as np
from PIL import Image
import re
from collections import Counter
import math
import os
from scipy import signal
from scipy import ndimage
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import cv2
import base128_decode
import correct_error_add
import self_correct
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error as MSE

def fspecial_gauss(size, sigma):
    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g/g.sum()
 
 
def ssim_1(img1, img2, cs_map=False):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    size = 11
    sigma = 1.5
    window = fspecial_gauss(size, sigma)
    K1 = 0.01
    K2 = 0.03
    L = 255 #bitdepth of image
    C1 = (K1*L)**2
    C2 = (K2*L)**2
    mu1 = signal.fftconvolve(window, img1, mode='valid')
    mu2 = signal.fftconvolve(window, img2, mode='valid')
    mu1_sq = mu1*mu1
    mu2_sq = mu2*mu2
    mu1_mu2 = mu1*mu2
    sigma1_sq = signal.fftconvolve(window, img1*img1, mode='valid') - mu1_sq
    sigma2_sq = signal.fftconvolve(window, img2*img2, mode='valid') - mu2_sq
    sigma12 = signal.fftconvolve(window, img1*img2, mode='valid') - mu1_mu2
    if cs_map:
        return (((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
                    (sigma1_sq + sigma2_sq + C2)), 
                (2.0*sigma12 + C2)/(sigma1_sq + sigma2_sq + C2))
    else:
        return ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
                    (sigma1_sq + sigma2_sq + C2))
 
def mssim(img1, img2):
    """
    refer to https://github.com/mubeta06/python/tree/master/signal_processing/sp
    """
    level = 5
    weight = np.array([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
    downsample_filter = np.ones((2, 2))/4.0
    im1 = img1.astype(np.float64)
    im2 = img2.astype(np.float64)
    mssim = np.array([])
    mcs = np.array([])
    for l in range(level):
        ssim_map, cs_map = ssim_1(im1, im2, cs_map=True)
        mssim = np.append(mssim, ssim_map.mean())
        mcs = np.append(mcs, cs_map.mean())
        filtered_im1 = ndimage.filters.convolve(im1, downsample_filter, 
                                                mode='reflect')
        filtered_im2 = ndimage.filters.convolve(im2, downsample_filter, 
                                                mode='reflect')
        im1 = filtered_im1[::2, ::2]
        im2 = filtered_im2[::2, ::2]
    return (np.prod(mcs[0:level-1]**weight[0:level-1])*
                    (mssim[level-1]**weight[level-1]))
                    
def write_list_to_txt(data, filename):
    with open(filename, "w") as file:
        for item in data:
            file.write(str(item) + "\n")


def confirm_dna_seq_value(input_data):
    if input_data == '00':
        return 'A'
    elif input_data == '10':
        return 'C'
    elif input_data == '01':
        return 'T'
    elif input_data == '11':
        return 'G'
    else:
        ValueError('please input correct value!')

# 整理结果数据
def collating_final_code(f_list):
    final_arr = []
    for idx in range(len(f_list)):
        final_code = f_list[idx]
        f_c_arr = []
        for idy in range(len(final_code)):
            f_c = str(final_code[idy])
            f_c_arr.append(f_c)
            code_s = "".join(f_c_arr)
        final_arr.append(code_s)
    return final_arr

# 字典构建
def confirm_Seven_Pro_Ten_list(input_data):
    #  文件传过来数据   字符串类型
    data  = input_data
    i = 0
    sumSep = 0

    # 将十进制转为7位二进制
    seq_Sev = []
    for x in range(0, 128):
        a = x
        a = bin(x)  # 将数据转化为二进制的函数
        a = a[2:]  # 从第三位开始取数据
        seq_Sev.append(a.zfill(7))
    #    print(seq_Sev[x])
    a = [0] * 128
    while i <= len(data):
        for t in range(128):
            if seq_Sev[t] == str(data[i + 10: i + 17]):
                a[t] = int(a[t]) + 1
        i = i + 17
    #  上面数据 生成七位的二进制数据和出现在数据集中的概率
    # 将生成的两位列表，按照概率的高的进行排序
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
            data1.append(data[i][:10])
    for i in range(len(dictionary)):
        dictionary[i].append(data1[i])
    return dictionary

#   七进制映射成十进制函数
def functions_to_ten(num, dictionary):
    string1 = ""
    for i in range(len(dictionary)):
        if int(num) == dictionary[i][0]:
            string1 = str(dictionary[i][2]).zfill(10)
            return string1
'''
step 2
（1）将数据分割成17个一组,%17≠0，
（2）后7个通过映射函数映射为十进制均衡码
（3）均衡码与前10个数据异或，获得离散数据
（4）均衡码与离散数据合成最终数据  交叉
'''
def core_method(text, dictionary):
    # 读取txt数据，切分17为1组
    data = text
    sub_group = []
    # 17组打印
    for i in range(0, len(data), 17):
        split_sub_group = data[i:i + 17]
        arr = list(split_sub_group)
        sub_group.append(arr)
    end_arr = []
    if len(sub_group[-1]) % 17 > 0:
        end_sec_arr_list = sub_group[-2]
        end_sec_arr_data = end_sec_arr_list[len(sub_group[-1]):17]
        for i in range(len(sub_group[-1])):
            index = i
            end_sec_arr_data.append(sub_group[-1][index])
        end_arr = end_sec_arr_data
    sub_group[-1] = end_arr
    # 每个组，后7个数据通过映射函数映射为十进制均衡码
    equilibrium_arr = []
    for idx in range(len(sub_group)):
        sub_data = sub_group[idx]
        back_seven_data = sub_data[-7:]
        str_back_seven_data = "".join(back_seven_data)
        #  调用函数将七位数据映射成十进制数据
        equilibrium_code = functions_to_ten(str_back_seven_data, dictionary)
        equilibrium_arr.append(equilibrium_code)
    # 每个组，前10个数据与后7个生成的均衡码进行异或操作
    discrete_arr = []
    for idx in range(len(sub_group)):
        # 均衡码
        eq_code = equilibrium_arr[idx]
        # 获取每组前10数据
        sub_data = sub_group[idx]
        sub_for_ten_data = sub_data[:10]
        discrete_arr.append(sub_for_ten_data)
        str_for_ten_data = "".join(sub_for_ten_data)
    # 均衡码与离散数据合成最终数据
    final_list = []
    for idx in range(len(equilibrium_arr)):
        d_code = equilibrium_arr[idx]
        e_code = discrete_arr[idx]
        final_code = []
        for idy in range(len(e_code)):
            e = d_code[idy]
            final_code.append(int(e))
            d = e_code[idy]
            final_code.append(d)
        final_list.append(final_code)
    # print(final_list)
    return final_list
'''
step 3
00-A  10-C  01-T  11-G
'''
def convert_data_to_dna_sequence(code):  # 11 01 11 01 11 10 00 01 00 10
    dna_seq = []
    for i in range(len(code) // 2):
        a = code[i * 2:i * 2 + 2]
        value = confirm_dna_seq_value(a)
        dna_seq.append(value)
    dna_seq_str = "".join(dna_seq)
    return dna_seq_str

def split_string_by_length(string, length):
    result = []
    for i in range(0, len(string), length):
        result.append(string[i:i+length])
    return result    
def mutate(string,rate):
    ins=0;dele=0;subi=0
    dna = list(string)
    random.seed(None)
    for index, char in enumerate(dna):
        if (random.random() <= rate):
            h=random.random()
            if(h<=0.8):
                subi+=1
                dna[index]=sub(dna[index])
            elif(h<=0.9):
                dele+=1
                dna[index]=""
            else:
                ins+=1
                dna[index]+=insert()
    return "".join(dna),subi,dele,ins



def sub(base):
    random.seed(None)
    suiji=random.randint(0,8)
    if(base=="A"):    
        if(suiji<3): return 'T'
        if(suiji<6): return 'C'
        if(suiji<9): return 'G'
    if(base=="G"):
        if(suiji<3): return 'T'
        if(suiji<6): return 'C'
        if(suiji<9): return 'A'
    if(base=="C"):
        if(suiji<3): return 'T'
        if(suiji<6): return 'G'
        if(suiji<9): return 'A'
    if(base=="T"):
        if(suiji<3): return 'G'
        if(suiji<6): return 'C'
        if(suiji<9): return 'A'

    
def insert():
    suiji=random.randint(0,3)
    if(suiji==1): return 'T'
    if(suiji==2): return 'C'
    if(suiji==3): return 'G'
    if(suiji==0): return 'A'
def correct_length(group_sequence):
    correct_length_sequence=[]
    for l in group_sequence:
        length=len(l)
        if(length>150):
            while(length>150):
                l = l[:75] + l[76:]
                length-=1
        else:
            while(length<150):
                l = l[:75] +"C" +l[75:]
                length+=1
        correct_length_sequence.append(l)
    return correct_length_sequence
def sequence_error(sequence,rate):
    h=[];s=0;d=0;i=0
    for each in sequence:
        temp,subi,dele,ins=mutate(each, rate)
        s+=subi;d+=dele;i+=ins
        h.append(temp)
    return h  
if __name__ == '__main__':
    image='lena.bmp'

    data_value=[]
    # step 1 : 图像转为二进制数据
    img = Image.open(image).convert('L')
    # 转换为Numpy数组
    img_np = np.array(img)
    # 转换为一维数组
    quantized_coef_1d = img_np.flatten()
    # 一维数组转换为二进制数据
    compressed_data = ''.join([format(int(x), '08b') for x in quantized_coef_1d])
    # step 2 :字典构建     将或得的数据compressed_data   加入到函数中
    dictionary = confirm_Seven_Pro_Ten_list(compressed_data)
    # step 3 ：核心编码
    final_list = core_method(compressed_data, dictionary)
    # 整理结果
    result = collating_final_code(final_list)
    # 还原为最终DNA序列
    get_dna_seq = ""
    for idx in range(len(result)):
        res = result[idx]
        get_dna_seq = str(get_dna_seq) + str(convert_data_to_dna_sequence(res))
    #data_1 = correct_error_add.error_add(get_dna_seq, error_rate)

                    
                        
    data_1=sequence_error(split_string_by_length(get_dna_seq, 150), 0.01)
    data = self_correct.drift_correct(data_1, dictionary)
    data_2=correct_length(data)
    sequence=""
    for dna in data_2:
        sequence+=dna
        
    base_error=0
    
    for i in range(min(len(get_dna_seq),len(sequence))):
        if(get_dna_seq[i]!=sequence[i]):
            base_error+=1
        
    shuju = base128_decode.decode_core_method(data_2, dictionary)
    shuju1 = ""
    for i in range(len(shuju)):
        shuju1 = shuju1 + "".join(shuju[i])
    # 图像重建部分
    binary_bit_data = shuju1[:len(compressed_data)]
    last_bit_data = binary_bit_data
    bit_wrong=0

    for i in range(len(compressed_data)):
        if(compressed_data[i]!=last_bit_data[i]):
            bit_wrong+=1
    quantized_coef_1d = np.array([int(last_bit_data[i:i + 8], 2) for i in range(0, len(last_bit_data), 8)])
    quantized_coef = np.reshape(quantized_coef_1d, img_np.shape)
    quantized_coef = quantized_coef.astype(np.uint8)
    # 显示图像
    
    
    
    
    cv2.imwrite('Reconstructed.bmp', quantized_coef)
    image=cv2.imread('Reconstructed.bmp')
    median = cv2.medianBlur(image, 3)
    cv2.imwrite("denoise.bmp", median)
    
    
    img1 = cv2.imread("lena.bmp",cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread('Reconstructed.bmp',cv2.IMREAD_GRAYSCALE)
    img3 = cv2.imread('denoise.bmp',cv2.IMREAD_GRAYSCALE)
    
    
            
    
    
    print(str(ssim(img1, img2))+"\t"+str(ssim(img1, img3))+"\n")
                
    print(str(mssim(img1, img2))+"\t"+str(mssim(img1, img3))+"\n")
    print(str(psnr(img1, img2))+"\t"+str(psnr(img1, img3))+"\n")
    print(str(MSE(img1, img2))+"\t"+str(MSE(img1, img3))+"\n")
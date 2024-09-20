import numpy as np
from PIL import Image
import re

import random

from scipy import signal
from scipy import ndimage
import cv2
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error as MSE


def fspecial_gauss(size, sigma):
    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g/g.sum()
 

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

def sequence_error(sequence,rate):
    h=[];s=0;d=0;i=0
    for each in sequence:
        temp,subi,dele,ins=mutate(each, rate)
        s+=subi;d+=dele;i+=ins
        h.append(temp)
    return h



def split_string_by_length(string, length):
    result = []
    for i in range(0, len(string), length):
        result.append(string[i:i+length])
    return result

def dna_decode(sequence,x,r):
    string_local=""
    for aa in sequence:
        x = logistic_map(x, r)
        
        q=int((x*24000))%24   
        trans=method(q)
        reverse_dict1 = dict([(value,key) for (key,value) in trans.items()])
        string_local+=reverse_dict1[aa]
    return string_local


def direct(a):
    mat = np.array(a)
    
    b = mat.transpose()
    
    return b

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

def getMaxDupChar(s, startIndex, curMaxLen, maxLen):
    if startIndex == len(s) - 1:
        return max(curMaxLen, maxLen)
    if list(s)[startIndex] == list(s)[startIndex + 1]:
        return getMaxDupChar(s, startIndex + 1, curMaxLen + 1, maxLen)
    else:
        return getMaxDupChar(s, startIndex + 1, 1, max(curMaxLen, maxLen))

def method(num):
    if(num==0):
        return {"00": "A","11": "T","10": "C","01": "G"}
    if(num==1):
        return {"00": "A","11": "T","10": "G","01": "C"}
    if(num==2):
        return {"00": "A","11": "G","10": "T","01": "C"}
    if(num==3):
        return {"00": "A","11": "G","10": "C","01": "T"}
    if(num==4):
        return {"00": "A","11": "C","10": "T","01": "G"}
    if(num==5):
        return {"00": "A","11": "C","10": "G","01": "T"}
    if(num==6):
        return {"00": "G","11": "C","10": "T","01": "A"}
    if(num==7):
        return {"00": "G","11": "C","10": "A","01": "T"}
    if(num==8):
        return {"00": "G","11": "T","10": "C","01": "A"}
    if(num==9):
        return {"00": "G","11": "T","10": "A","01": "C"}
    if(num==10):
        return {"00": "G","11": "A","10": "C","01": "T"}
    if(num==11):
        return {"00": "G","11": "A","10": "T","01": "C"}
    if(num==12):
        return {"00": "C","11": "G","10": "T","01": "A"}
    if(num==13):
        return {"00": "C","11": "G","10": "A","01": "T"}
    if(num==14):
        return {"00": "C","11": "T","10": "G","01": "A"}
    if(num==15):
        return {"00": "C","11": "T","10": "A","01": "G"}
    if(num==16):
        return {"00": "C","11": "A","10": "T","01": "G"}
    if(num==17):
        return {"00": "C","11": "A","10": "G","01": "T"}
    if(num==18):
        return {"00": "T","11": "A","10": "C","01": "G"}
    if(num==19):
        return {"00": "T","11": "A","10": "G","01": "C"}
    if(num==20):
        return {"00": "T","11": "G","10": "C","01": "A"}
    if(num==21):
        return {"00": "T","11": "G","10": "A","01": "C"}
    if(num==22):
        return {"00": "T","11": "C","10": "G","01": "A"}
    if(num==23):
        return {"00": "T","11": "C","10": "A","01": "G"}
    
def dna_encode(binary,DNA_encoding):

    return DNA_encoding[binary]
def image_to_bitstream(image_path):


    img = Image.open(image_path)
    img_arr = np.array(img)
    
    
    
    bitstream = ''.join([f"{bin(pixel)[2:].zfill(8)}" for pixel in img_arr.flatten()])
    
    return bitstream


def bitstream_to_image(bitstream, image_size):
    array_length = image_size[0] * image_size[1]
    
    image_array = np.array([int(bitstream[i:i+8], 2) for i in range(0, array_length*8, 8)], dtype=np.uint8)
    image_array = image_array.reshape(image_size[1], image_size[0])
    image = Image.fromarray(image_array)
    return image
def logistic_map(x, r):
    return r * x*(1-x)

if __name__ == '__main__':    
    #pixel bitstream
    bitstream=image_to_bitstream("lena.bmp") 
    
    #encrypted encode
    string=""
    binary=re.findall(r'\w{2}', bitstream)
    x= 0.4 
    r = 3.9 
    
    for m in binary:
        x = logistic_map(x, r) 
        q=int((x*24000))%24    
        string+=dna_encode(m, method(q))
    
    stl=split_string_by_length(string, 150)
    
    
    #transposition interleaving, group with 150 sequences
    dna=[]
    for i in range(len(stl)//150):
        a=[]
        for j in range(150):
            lst = list(stl[i*150+j])
            a.append(lst)
        data = np.array(a)        
        new_data=direct(data)
        for j in range(len(new_data)): 
            name_str = ''.join(x if x else "" for x in new_data[j])
            dna.append(name_str)
    tmp=len(stl)-len(stl)//150*150
    others=""
    for i in range(tmp):
        if(len(stl[len(stl)//150*150+i])==150):
            dna.append(stl[len(stl)//150*150+i])
        else:
            others=stl[len(stl)//150*150+i]
    
    #add error        
    muta_dna=sequence_error(dna,0.01)
    
    #correct length Maximum-probability insertion and deletion
    correct_dna=correct_length(muta_dna)
        
    
    
    #deinterleaving    
    dna_sequence=""
    for i in range(len(correct_dna)//150):
        a=[]
        for j in range(150):
            lst = list(correct_dna[i*150+j])
            a.append(lst)
        data = np.array(a)        
        new_data=direct(data) 
        for j in range(len(new_data)):
            name_str = ''.join(x if x else "" for x in new_data[j])
            dna_sequence+=name_str
    tmp=len(correct_dna)-len(correct_dna)//150*150
    a=[]
    for i in range(tmp):
        dna_sequence+=correct_dna[len(correct_dna)//150*150+i]
    dna_sequence+=others
    
    #decode   
    bitstream_1=dna_decode(dna_sequence,0.4,3.9)
    
    #image
    image_size = (256, 256)
    image = bitstream_to_image(bitstream_1, image_size)
    image.save("trans_lena.bmp")
    
    img = cv2.imread("trans_lena.bmp",cv2.IMREAD_GRAYSCALE)
    #median filter
    median = cv2.medianBlur(img, 3)
    cv2.imwrite("trans_denosie_lena.bmp", median)
    img1 = cv2.imread("lena.bmp",cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread("trans_lena.bmp",cv2.IMREAD_GRAYSCALE)
    img3= cv2.imread("trans_denosie_lena.bmp",cv2.IMREAD_GRAYSCALE)
    
    
    #the quality of reconstructed image
    print(str(ssim(img1, img2))+"\t"+str(ssim(img1, img3))+"\n")                    
    print(str(mssim(img1, img2))+"\t"+str(mssim(img1, img3))+"\n")
    print(str(psnr(img1, img2))+"\t"+str(psnr(img1, img3))+"\n")
    print(str(MSE(img1, img2))+"\t"+str(MSE(img1, img3))+"\n")




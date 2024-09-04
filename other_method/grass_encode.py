import numpy as np
from fuzzywuzzy import fuzz
import random
from PIL import Image
from scipy import signal
from scipy import ndimage
import cv2
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error as MSE
def replace_string(text, new_str, start, end):
    return text[:start] + new_str + text[end:]

def encode(bit_segment):
    dna_sequence = []

    
    if len(bit_segment) % 16 != 0:
        raise ValueError("The length of inputted binary segment must be divided by 16!")

    for position in range(0, len(bit_segment), 16):
        decimal_number = int("".join(list(map(str, bit_segment[position: position + 16]))), 2)

        rule_indices = []
        for index in range(3):
            rule_indices.append(decimal_number % 47)
            decimal_number -= rule_indices[-1]
            decimal_number /= 47

        rule_indices = rule_indices[::-1]
        for rule_index in rule_indices:
            dna_sequence += mapping_rules[0][mapping_rules[1].index(rule_index)]




    return "".join(dna_sequence)

def decode(dna_sequence):


    bit_segment = []
    for position in range(0, len(dna_sequence), 9):
        decimal_number, carbon_piece = 0, dna_sequence[position: position + 9]
        for index in [0, 3, 6]:
            try:
                position = mapping_rules[0].index("".join(carbon_piece[index: index + 3]))
            except:
                tmp="";maxx=0
                for k in mapping_rules[0]:
                    similarity = fuzz.ratio(carbon_piece[index: index + 3], k)
                    if(similarity>maxx):
                        maxx=similarity;tmp=k
                
                
                position = mapping_rules[0].index(tmp)
            value = mapping_rules[1][position]
            decimal_number = decimal_number * 47 + value
        bit_segment += list(map(str, list(str(bin(decimal_number))[2:].zfill(16))))[:16]
        




    return "".join(bit_segment)


 

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

      

def image_to_bitstream(image_path):

    img = Image.open(image_path)
    img_arr = np.array(img)    
    bitstream = ''.join([f"{bin(pixel)[2:].zfill(8)}" for pixel in img_arr.flatten()])
    
    return bitstream,img.size



      
def bitstream_to_image(bitstream, image_size):
    array_length = image_size[0] * image_size[1]
    
    image_array = np.array([int(bitstream[i:i+8], 2) for i in range(0, array_length*8, 8)], dtype=np.uint8)
    image_array = image_array.reshape(image_size[1], image_size[0])
    image = Image.fromarray(image_array)
    return image

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

if __name__ == "__main__":
    
    base_values = [index for index in range(48)]
    mapping_rules = [[], []]
    gc_codes = ['AAC', 'AAG', 'AAT', 'ACA', 'ACG', 'ACT', 'AGA', 'AGC', 'AGT', 'ATA', 'ATC', 'ATG',
                'CAC', 'CAG', 'CAT', 'CCA', 'CCG', 'CCT', 'CGA', 'CGC', 'CGT', 'CTA', 'CTC', 'CTG',
                'GAC', 'GAG', 'GAT', 'GCA', 'GCG', 'GCT', 'GGA', 'GGC', 'GGT', 'GTA', 'GTC', 'GTG',
                'TAC', 'TAG', 'TAT', 'TCA', 'TCG', 'TCT', 'TGA', 'TGC', 'TGT', 'TTA', 'TTC', 'TTG']
    for index, value in enumerate(base_values):
        if 0 <= base_values[index] < 47:
            mapping_rules[0].append(gc_codes[index])
            mapping_rules[1].append(value)    
        
    
    
    #h="fly_grey.jpg"
    #h="baboon_grey.jpg"
    h="lena.bmp"
    
    bitstream,image_size =image_to_bitstream(h) 
    #分割
    
    #16->9 base
    data_base=encode(bitstream)
    length=len(data_base)
    split_sequence=split_string_by_length(data_base, 150)
    

    muta_dna=sequence_error(split_sequence,0.01)
    
    
    correct_dna=correct_length(muta_dna)
    sequence=""
    for i in correct_dna:
        sequence+=i
    
    
    bitstream_2=decode(sequence[:294912])
    
    image=bitstream_to_image(bitstream_2, image_size)
    
    image.save("grass.bmp")
    image=cv2.imread("grass.bmp")
    median = cv2.medianBlur(image, 3)
    cv2.imwrite("denoise.bmp", median)
    img1 = cv2.imread(h,cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread('grass.bmp',cv2.IMREAD_GRAYSCALE)

    img3 = cv2.imread('denoise.bmp',cv2.IMREAD_GRAYSCALE)
    print(str(ssim(img1, img2))+"\t"+str(ssim(img1, img3))+"\n")
                    
    print(str(mssim(img1, img2))+"\t"+str(mssim(img1, img3))+"\n")
    print(str(psnr(img1, img2))+"\t"+str(psnr(img1, img3))+"\n")
    print(str(MSE(img1, img2))+"\t"+str(MSE(img1, img3))+"\n")
    

            
            
            
            

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
####################
from numpy import fromfile, uint8
import numpy as np
from PIL import Image
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


######Chunking the input binary and randomizing the first position
def segment(ascii_binary):
    ascii_length = len(ascii_binary)
    pad = (5 - ascii_length % 5) % 5
    temp = ascii_binary + '0' * pad
    n = (ascii_length + pad) // 5
    temp = np.array(list(temp)).reshape((n, 5))
    temp_first_bit = (temp[:, 0])
    temp_first_bit_bool = [bool(int(x)) for x in temp_first_bit]
    c = temp_first_bit_bool
    ratio = 0.0
    numl = 0
    ###### Balancing the ratio of first position sequences 0 and 1 so that the final coding yields a balanced base sequence GC
    while not 0.40 <= ratio <= 0.60:
        numl += 1
        length1 = len(temp_first_bit_bool)
        # X_1 iteration
        x_1 = 0.1 + numl * 5 * 10 ** -5
        u = 3.9
        y = np.zeros(length1)
        x_n_1 = x_1
        for k in range(length1):
            if x_n_1 > 0.5:
                y[k] = 1
            else:
                y[k] = 0
            x_n_2 = u * (1 - x_n_1) * x_n_1
            x_n_1 = x_n_2
        y = y.astype(bool)
        c = np.logical_xor(temp_first_bit_bool, y)
        c = (c).astype(int)
        n = len(c)
        ratio = np.count_nonzero(c == 0) / n

    #
    #print(''.join(y.astype(int).astype(str)))
    #########Output the initial value of the chaotic sequence for subsequent decoding
    print("Initial value of chaotic sequence: ")
    print(x_1)
    #print("The binary sequence after the dissimilarity is: ")
    #print(c)
    print("The weight of 0 in the binary sequence after the dissimilarity is: " + str(ratio))
    c = ''.join(c.astype(int).astype(str))
    c = np.reshape(list(c), (n, 1))
    temp[:, 0] = c[:, 0]
    preprocessed_binary_data = np.reshape(temp, n * 5)
    print("The length of the binary sequence to be encoded is: ")
    print(len(preprocessed_binary_data))
    #print("The binary order to be encoded is: ")
    #print(preprocessed_binary_data)
    return preprocessed_binary_data,x_1,numl

def encode_sequence(sequence):
    rule_0 = {
        '0000': 'ACT',
        '0001': 'AGT',
        '0010': 'ATC',
        '0011': 'ATG',
        '0100': 'TCA',
        '0101': 'TGA',
        '0110': 'TAC',
        '0111': 'TAG',
        '1100': 'CAT',
        '1101': 'CTA',
        '1110': 'CAA',
        '1111': 'CTT',
        '1000': 'GTA',
        '1001': 'GAT',
        '1010': 'GTT',
        '1011': 'GAA'
    }
    rule_1 = {
        '0000': 'ACG',
        '0001': 'AGC',
        '0010': 'AGG',
        '0011': 'ACC',
        '0100': 'TGC',
        '0101': 'TCG',
        '0110': 'TCC',
        '0111': 'TGG',
        '1100': 'CAC',
        '1101': 'CAG',
        '1110': 'CTC',
        '1111': 'CTG',
        '1000': 'GAG',
        '1001': 'GAC',
        '1010': 'GTG',
        '1011': 'GTC'
    }

    first_bit = int(sequence[0])
    key = sequence[1:]

    if first_bit == 0:
        if key in rule_0:
            return rule_0[key]
    elif first_bit == 1:
        if key in rule_1:
            return rule_1[key]

    return "Invalid sequence"


def decode_sequence(seq):
    rule_0 = {
        'ACT': '00000',
        'AGT': '00001',
        'ATC': '00010',
        'ATG': '00011',
        'TCA': '00100',
        'TGA': '00101',
        'TAC': '00110',
        'TAG': '00111',
        'CAT': '01100',
        'CTA': '01101',
        'CAA': '01110',
        'CTT': '01111',
        'GTA': '01000',
        'GAT': '01001',
        'GTT': '01010',
        'GAA': '01011'
    }
    rule_1 = {
        'ACG': '10000',
        'AGC': '10001',
        'AGG': '10010',
        'ACC': '10011',
        'TGC': '10100',
        'TCG': '10101',
        'TCC': '10110',
        'TGG': '10111',
        'CAC': '11100',
        'CAG': '11101',
        'CTC': '11110',
        'CTG': '11111',
        'GAG': '11000',
        'GAC': '11001',
        'GTG': '11010',
        'GTC': '11011'
    }
    #########Counting the number of GCs within a base slice for decoding purposes
    count_gc = seq.count('G') + seq.count('C')
    if count_gc == 1:
        if seq in rule_0:
            return rule_0[seq]
    elif count_gc == 2:
        if seq in rule_1:
            return rule_1[seq]
    return "11111"

#############Converting files to binary
def read_bits_from_file(path, segment_length=150, need_logs=True):

    img = Image.open(path).convert('L')
    img_arr = np.array(img)
    
    
    
    bitstream = ''.join([f"{bin(pixel)[2:].zfill(8)}" for pixel in img_arr.flatten()])
    
    bitstream+= -((len(bitstream)) % segment_length) *"0"
    
    return list(bitstream)

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
##########Convert binary to file

def calculate_GC_content(input_str):
    total_length = len(input_str)
    count_GC = input_str.count('G') + input_str.count('C')
    GC_content = count_GC / total_length * 100
    return GC_content

def bitstream_to_image(bitstream, image_size):
    array_length = image_size[0] * image_size[1]
    
    image_array = np.array([int(bitstream[i:i+8], 2) for i in range(0, array_length*8, 8)], dtype=np.uint8)
    image_array = image_array.reshape(image_size[1], image_size[0])
    image = Image.fromarray(image_array)
    return image
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

    
    
######Path to the input file
binary=read_bits_from_file('lena.bmp', need_logs=True)

binary_sequence = ''.join(map(str, binary))
temp=segment(binary_sequence)
print(temp[2])
temp = temp[0]
temp = ''.join(temp)
#Turn temp into an array of n rows and 5 columns.
n = len(temp) // 5
# Slicing a string and assembling it into a list of n rows and 5 columns
substrings = [temp[i*5:(i+1)*5].zfill(5) for i in range(n)]
a=len(substrings)
output = ""  # Creating an Empty String
for i in range(len(substrings)):
    DNA_sequence= encode_sequence(substrings[i])
    output += DNA_sequence
#print(output)
DNAsequence=output
print("The logical storage density is",len(binary)/len(DNAsequence))
GC_content = calculate_GC_content(DNAsequence)
print("GC content: {:.2f}%".format(GC_content))
print("Coding Completed")

string=split_string_by_length(DNAsequence,150)







mutation=sequence_error(string, 0.01)
mutation_length=correct_length(mutation)
DNAsequence_2=""

for k in mutation_length:
    DNAsequence_2+=k


################Decoding of DNA Sequence.
n = len(DNAsequence_2) // 3
# Slicing a string and assembling it into a list of n rows and 5 columns
DNAstrings = [DNAsequence_2[i*3:(i+1)*3].zfill(3) for i in range(n)]
Binary_output = ""
for i in range(len(DNAstrings)):
    Binary_sequence= decode_sequence(DNAstrings[i])
    Binary_output += Binary_sequence
#print(Binary_output)

source_first_bit = ''
for i in range(0, len(Binary_output), 5):
    source_first_bit += Binary_output[i]

#print(source_first_bit)
source_first_bit = np.fromstring(source_first_bit, dtype=np.uint8) - ord('0')
source_first_bit = np.logical_xor(source_first_bit.astype(bool), False)
print(source_first_bit.shape)
x_1 = 0.10005
u = 3.9
y = np.zeros(len(source_first_bit))
x_n_1 = x_1

for k in range(len(source_first_bit)):
    if x_n_1 > 0.5:
        y[k] = 1
    else:
        y[k] = 0

    x_n_2 = u * (1 - x_n_1) * x_n_1
    x_n_1 = x_n_2
y = y.astype(bool)
temp_first_bit = np.logical_xor(source_first_bit, y)
temp_first_bit = (temp_first_bit).astype(int)
temp_first_bit = ''.join(temp_first_bit.astype(int).astype(str))
temp_first_bit = np.reshape(list(temp_first_bit), (len(temp_first_bit), 1))
Binary_output = np.array(list(Binary_output)).reshape((len(Binary_output)//5), 5)
Binary_output[:, 0] = temp_first_bit[:, 0]
primary_binary_data = np.reshape(Binary_output, len(Binary_output) * 5)
primary_binary_list = [str(bit) for bit in primary_binary_data]

############Recover the base sequence to the original file and save it in the corresponding path, here 3840480bit is the size of the file to be recovered.

image=bitstream_to_image("".join(primary_binary_list)[:524288], (256,256))
image.save("target.bmp")
print("Decoding completed" )
image=cv2.imread("target.bmp")
median = cv2.medianBlur(image, 3)
cv2.imwrite("denoise.bmp", median)

img1 = cv2.imread("lena.bmp",cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('target.bmp',cv2.IMREAD_GRAYSCALE)
img3 = cv2.imread('denoise.bmp',cv2.IMREAD_GRAYSCALE)

print(str(ssim(img1, img2))+"\t"+str(ssim(img1, img3))+"\n")
                
print(str(mssim(img1, img2))+"\t"+str(mssim(img1, img3))+"\n")
print(str(psnr(img1, img2))+"\t"+str(psnr(img1, img3))+"\n")
print(str(MSE(img1, img2))+"\t"+str(MSE(img1, img3))+"\n")

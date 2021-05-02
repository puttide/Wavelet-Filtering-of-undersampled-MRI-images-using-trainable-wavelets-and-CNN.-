
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 13:56:16 2019

@author: putti
"""

import os
import numpy as np
from pathlib import Path
import nibabel as nib
from sklearn.model_selection import train_test_split
#from torch.utils.data import DataLoader
#from PyTorchModelSummary import ModelSummary  
import lowlevel
from skimage.measure import compare_ssim as ssim

#####################note:low level.py should be imported
folder_path='D:\PROJECT\Oasis Brains Dataset'#########fully sampled images but in 3d format
folder_path_fully='D:\PROJECT\Oasis Brains Dataset'#########fully sampled  images but in 3d format
folder_path_under='D:\PROJECT\Artifacts'#########undersampled images but in 3d format
extension_under='nii'


def Nifti3Dto2D(Nifti3D):
    #Nifti3DWOChannel = Nifti3D[:,:,:,0] #Considering there is only one chnnel info
    Nifti2D = Nifti3D.reshape(np.shape(Nifti3D)[0], np.shape(Nifti3D)[1] * np.shape(Nifti3D)[2])
    #Nifti1D = Nifti2D.reshape(np.shape(Nifti2D)[0] * np.shape(Nifti2D)[1])
    return Nifti2D
def Nifti4Dto2D(Nifti4D):
    Nifti3DWOChannel = Nifti4D[:,:,:,0] #Considering there is only one chnnel info
    Nifti2D = Nifti3DWOChannel.reshape(np.shape(Nifti3DWOChannel)[0], np.shape(Nifti3DWOChannel)[1] * np.shape(Nifti3DWOChannel)[2])
    #Nifti1D = Nifti2D.reshape(np.shape(Nifti2D)[0] * np.shape(Nifti2D)[1])
    return Nifti2D

def Nifti4Dto3D(Nifti4D):
    Nifti3D = Nifti4D[:,:,:,0] #Considering there is only one chnnel info
    return Nifti3D
    
    
    
#def Nifti2Dto3D(Nifti2D, height):
def Nifti2Dto3D(Nifti2D):
  #  Nifti2D = Nifti1D.reshape(height,int(np.shape(Nifti1D)[0]/height))
   Nifti3DWOChannel = Nifti2D.reshape(np.shape(Nifti2D)[0],np.shape(Nifti2D)[0],int((np.shape(Nifti2D)[1])/(np.shape(Nifti2D)[0])))
   #Nifti3DWOChannel = Nifti2D.reshape(np.shape(Nifti2D)[0],np.shape(Nifti2D)[1],np.shape(Nifti2D)[1]/height) 
   return Nifti3DWOChannel
   # return Nifti2D


def FileRead(file_path):
    nii = nib.load(file_path)
    data = nii.get_data()
    #if (np.shape(np.shape(data))[0] == 3): #If channel data not present
    #    data = np.expand_dims(data, 3)
    return data

def FileSave(data, file_path):
    nii = nib.Nifti1Image(data, np.eye(4))
    nib.save(nii, file_path)
    
def normalize(x):
    """
        argument
            - x: input image data in numpy array [32, 32, 3]
        return
            - normalized x 
    """
    min_val = np.min(x)
    max_val = np.max(x)
    x = (x-min_val) / (max_val-min_val)
    return x



def ReadNIFTIVolsWithUnder(folder_path_fully, folder_path_under, extension_under):
    subfolders_fully=[]
    Volumes = []
    VolumesUnder = []
    subjectNames = []
    subjectOfVol = []
    fileNames = []
   

    
    for f in os.listdir(folder_path_fully):
        subfolders_fully.append(f)
        subfolders_fully.sort()
#print(subfolders_fully)
    for folder in subfolders_fully:
        fullpath_subfolder_fully = os.path.join(folder_path_fully, folder)
        fullpath_subfolder_under = os.path.join(folder_path_under, folder)
        subject = folder
    #print(subject)
    #print(fullpath_subfolder_fully)
        for f in os.listdir(fullpath_subfolder_fully):
                files=[]
                files.append(f)
                files.sort()
                #print(files)
                if (len(os.listdir(fullpath_subfolder_fully)) > 0):
                    for file in files:
                        
                        if (file.endswith('.hdr')):
                            fullpath_file_fully = os.path.join(fullpath_subfolder_fully, file)
                            #print(fullpath_file_fully)    
                            imagenameNoExt = file.split('.')[0]
                            #print(imagenameNoExt)
                            undersampledImageName = imagenameNoExt+'.'+extension_under
                            #print(undersampledImageName)
                            fullpath_file_under = os.path.join(fullpath_subfolder_under, undersampledImageName)
                            #print(fullpath_file_under)
                            V = FileRead(fullpath_file_fully)
                            v=Nifti4Dto2D(V)
                            Vu = FileRead(fullpath_file_under)
                            vu=Nifti3Dto2D(Vu)
                            Volumes.append(v)
                            VolumesUnder.append(vu)
                            if subject not in subjectNames:
                                subjectNames.append(subject)
                                subjectOfVol.append(subjectNames.index(subject))
                            fileNames.append(imagenameNoExt)
    Volumes = np.asarray(Volumes,dtype=np.float32)
    VolumesUnder = np.asarray(VolumesUnder,dtype=np.float32)
    return Volumes, VolumesUnder, subjectOfVol, subjectNames, fileNames
#print(fileNames)
                            
#j=ReadNIFTIVols(folder_path) 
#j=ReadNIFTIVols(folder_path) 
[Volumes, VolumesUnder, subjectOfVol, subjectNames, fileNames] =ReadNIFTIVolsWithUnder(folder_path_fully, folder_path_under, extension_under)



#for value1,value2 in zip(z[0],z[1]):
    
artifact1=VolumesUnder-Volumes

#for index,img in enumerate(artifact):
 #   k=Nifti2Dto3D(img)
  #  index+=1
   # FileSave(k,'articmpwave_brain_image'+str(index)+'.img')
'''    
def normalize1(x):
    """
    Normalize a list of sample image data in the range of 0 to 1
    : x: List of image data.  
    : return: Numpy array of normalize data
    """
    # TODO: Implement Function
    x=np.array(x)
    x=x.astype(float)
    x=np.divide(x,np.max(x))
    return x
'''
def normalize1(x):
    """
        argument
            - x: input image data in numpy array [32, 32, 3]
        return
            - normalized x 
    """
    min_val = np.min(x)
    max_val = np.max(x)
    x = (x-min_val) / (max_val-min_val)
    return x

import pywt
artifact=[]
cAf=[]
cDf=[]

    
undersampled=VolumesUnder

Volumes_val=Volumes.reshape(8,256,32768) 
Volumes_val=Volumes.reshape(8,256,256,128)
Volumes_val=Volumes_val[0:8,0:256,0:256,50:70]
Volumes_val=Volumes_val.reshape(8,1,256,5120)
'''
cAvf=[]
cDvf=[]
for dat3 in Volumes_val :
    
    (cAv, cDv) = pywt.dwt(dat3, 'haar')
    cAvf.append(cAv)

    cDvf.append(cDv)
    
#(cA2, cD2) = pywt.dwt(artifact1[0], 'haar')
cAvvf=np.asarray(cAvf)
cDvvf=np.asarray(cDvf)
vol_val=np.stack((cAvvf,cDvvf),axis=0)
print(np.amin(vol_val))
print(np.amax(vol_val))
'''
vol_val=Volumes_val
vol_val=normalize(vol_val)
#undersampledf=normalize(undersampledf)
#artifact=np.stack((cAf1,cDf1))
#artifact=artifact.reshape(8,2,256,16384)############(B;C;L;W)
#vol_val=vol_val.reshape(8,2,256,256,64)
#vol_val=vol_val[0:2,0:2,0:256,0:256,30:40]
#vol_val=vol_val.reshape(8,2,256,2560)





undersampled=undersampled.reshape(8,256,256,128)
undersampled=undersampled[0:8,0:256,0:256,50:70]
undersampled=undersampled.reshape(8,1,256,5120)



undersampledf=undersampled
undersampledf=normalize(undersampledf)

#artifact=np.stack((cAf1,cDf1))
#vol_val=vol_val.reshape(8,2,256,2560)############(B;C;L;W)
#undersampledf=undersampledf.reshape(8,2,256,2560)#######(B;C;L;W)
#vol_val=normalize(vol_val)
#undersampled=normalize(undersampled)
#artifact=normalize(artifact)

#from sklearn.model_selection import train_test_split
undersampled_train,undersampled_test,vol_val_train,vol_val_test = train_test_split(undersampledf,vol_val, test_size=0.2, random_state=0)
    

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.autograd import Variable
    
import torch.utils.data as utils




#TRAINING SET 
#undersampled_train=undersampled_train.reshape[2,1,256,15360] 
#undersampled_train = np.transpose(undersampled_train, (0, 1, 3, 2))
#undersampled_train=undersampled_train[0:6,0:256,0:15360,0:1]

undersampled=VolumesUnder
undersampled_train=torch.tensor(undersampled_train)
print(undersampled_train.size())

#undersampled_train=undersampled_train[-1,0:256,0:15360]
print(undersampled_train.size()) 


#artifact_train = np.transpose(artifact_train, (0, 1, 3, 2))
#artifact_train=artifact_train[0:6,0:256,0:15360,0:1]

vol_val_train = torch.tensor(vol_val_train)
print(vol_val_train.size()) 

#artifact_train=artifact_train[-1,0:256,0:15360]
trainset=utils.TensorDataset(undersampled_train,vol_val_train)
#train_dataset=utils.DataLoader(trainset)
print(undersampled_train.size())

#TEST SET
undersampled_test=torch.tensor(undersampled_test)
vol_val_test = torch.tensor(vol_val_test)
testset=utils.TensorDataset(undersampled_test,vol_val_test)
#test_dataset=utils.DataLoader(testset)


def reflect(x, minx, maxx):
    """Reflect the values in matrix *x* about the scalar values *minx* and
    *maxx*.  Hence a vector *x* containing a long linearly increasing series is
    converted into a waveform which ramps linearly up and down between *minx*
    and *maxx*.  If *x* contains integers and *minx* and *maxx* are (integers +
    0.5), the ramps will have repeated max and min samples.
    .. codeauthor:: Rich Wareham <rjw57@cantab.net>, Aug 2013
    .. codeauthor:: Nick Kingsbury, Cambridge University, January 1999.
    """
    x = np.asanyarray(x)
    rng = maxx - minx
    rng_by_2 = 2 * rng
    mod = np.fmod(x - minx, rng_by_2)
    normed_mod = np.where(mod < 0, mod + rng_by_2, mod)
    out = np.where(normed_mod >= rng, rng_by_2 - normed_mod, normed_mod) + minx
    return np.array(out, dtype=x.dtype)




'''
STEP 2: MAKING DATASET ITERABLE
'''

batch_size = 1
#n_iters = 30
#num_epochs = n_iters / (len(trainset) / batch_size)
#num_epochs = int(num_epochs)
num_epochs = 100

from torch.utils.data import DataLoader
from PyTorchModelSummary import ModelSummary 

train_loader = DataLoader(dataset=trainset, 
                        batch_size=batch_size, 
                        shuffle=True)

test_loader = DataLoader(dataset=testset, 
                        batch_size=batch_size, 
                        shuffle=False)

def act(b):

    c=torch.pow(b,2)+b+0.1

    d=torch.clamp(c, -1,1)
    return d


'''
STEP 3: CREATE MODEL CLASS
'''


class DWTForward(nn.Module):
    """ Performs a 2d DWT Forward decomposition of an image
    Args:
        J (int): Number of levels of decomposition
        wave (str or pywt.Wavelet): Which wavelet to use. Can be a string to
            pass to pywt.Wavelet constructor, can also be a pywt.Wavelet class,
            or can be a two tuple of array-like objects for the analysis low and
            high pass filters.
        mode (str): 'zero', 'symmetric', 'reflect' or 'periodization'. The
            padding scheme
        separable (bool): whether to do the filtering separably or not (the
            naive implementation can be faster on a gpu).
        """
    def __init__(self, J=1, wave='db1', mode='zero'):
        super().__init__()
        if isinstance(wave, str):
            wave = pywt.Wavelet(wave)
        if isinstance(wave, pywt.Wavelet):
            h0_col, h1_col = wave.dec_lo, wave.dec_hi
            h0_row, h1_row = h0_col, h1_col
        else:
            if len(wave) == 2:
                h0_col, h1_col = wave[0], wave[1]
                h0_row, h1_row = h0_col, h1_col
            elif len(wave) == 4:
                h0_col, h1_col = wave[0], wave[1]
                h0_row, h1_row = wave[2], wave[3]

        # Prepare the filters
        filts = lowlevel.prep_filt_afb2d(h0_col, h1_col, h0_row, h1_row)
        self.h0_col = nn.Parameter(filts[0], requires_grad=False)
        self.h1_col = nn.Parameter(filts[1], requires_grad=False)
        self.h0_row = nn.Parameter(filts[2], requires_grad=False)
        self.h1_row = nn.Parameter(filts[3], requires_grad=False)
        self.J = J
        self.mode = mode

    def forward(self, x):
        """ Forward pass of the DWT.
        Args:
            x (tensor): Input of shape :math:`(N, C_{in}, H_{in}, W_{in})`
        Returns:
            (yl, yh)
                tuple of lowpass (yl) and bandpass (yh)
                coefficients. yh is a list of length J with the first entry
                being the finest scale coefficients. yl has shape
                :math:`(N, C_{in}, H_{in}', W_{in}')` and yh has shape
                :math:`list(N, C_{in}, 3, H_{in}'', W_{in}'')`. The new
                dimension in yh iterates over the LH, HL and HH coefficients.
        Note:
            :math:`H_{in}', W_{in}', H_{in}'', W_{in}''` denote the correctly
            downsampled shapes of the DWT pyramid.
        """
        yh = []
        ll = x
        mode = lowlevel.mode_to_int(self.mode)

        # Do a multilevel transform
        for j in range(self.J):
            # Do 1 level of the transform
            ll, high = lowlevel.AFB2D.apply(
                ll, self.h0_col, self.h1_col, self.h0_row, self.h1_row, mode)
            #yh.append(high)
        #yh=np.asarray(yh,dtype=np.int32)

        return ll, high


class DWTInverse(nn.Module):
    """ Performs a 2d DWT Inverse reconstruction of an image
    Args:
        wave (str or pywt.Wavelet): Which wavelet to use
        C: deprecated, will be removed in future
    """
    def __init__(self, wave='db1', mode='zero'):
        super().__init__()
        if isinstance(wave, str):
            wave = pywt.Wavelet(wave)
        if isinstance(wave, pywt.Wavelet):
            g0_col, g1_col = wave.rec_lo, wave.rec_hi
            g0_row, g1_row = g0_col, g1_col
        else:
            if len(wave) == 2:
                g0_col, g1_col = wave[0], wave[1]
                g0_row, g1_row = g0_col, g1_col
            elif len(wave) == 4:
                g0_col, g1_col = wave[0], wave[1]
                g0_row, g1_row = wave[2], wave[3]
        # Prepare the filters
        filts = lowlevel.prep_filt_sfb2d(g0_col, g1_col, g0_row, g1_row)
        self.g0_col = nn.Parameter(filts[0], requires_grad=False)
        self.g1_col = nn.Parameter(filts[1], requires_grad=False)
        self.g0_row = nn.Parameter(filts[2], requires_grad=False)
        self.g1_row = nn.Parameter(filts[3], requires_grad=False)
        self.mode = mode

    def forward(self, coeffs):
        """
        Args:
            coeffs (yl, yh): tuple of lowpass and bandpass coefficients, where:
              yl is a lowpass tensor of shape :math:`(N, C_{in}, H_{in}',
              W_{in}')` and yh is a list of bandpass tensors of shape
              :math:`list(N, C_{in}, 3, H_{in}'', W_{in}'')`. I.e. should match
              the format returned by DWTForward
        Returns:
            Reconstructed input of shape :math:`(N, C_{in}, H_{in}, W_{in})`
        Note:
            :math:`H_{in}', W_{in}', H_{in}'', W_{in}''` denote the correctly
            downsampled shapes of the DWT pyramid.
        Note:
            Can have None for any of the highpass scales and will treat the
            values as zeros (not in an efficient way though).
        """
        yl, yh = coeffs
        ll = yl
        mode = lowlevel.mode_to_int(self.mode)

        # Do a multilevel inverse transform
        #yh=list(yh)
        for h in yh[::-1]:
        #for h in yh.permute(4,3,2,1,0):
            if h is None:
                h = torch.zeros(ll.shape[0], ll.shape[1], 3, ll.shape[-2],
                                ll.shape[-1], device=ll.device)

            # 'Unpad' added dimensions
            if ll.shape[-2] > h.shape[-2]:
                ll = ll[...,:-1,:]
            if ll.shape[-1] > h.shape[-1]:
                ll = ll[...,:-1]
            ll = lowlevel.SFB2D.apply(
                ll, h, self.g0_col, self.g1_col, self.g0_row, self.g1_row, mode)
        return ll


class SWTForward(nn.Module):
    """ Performs a 2d Stationary wavelet transform (or undecimated wavelet
    transform) of an image
    Args:
        J (int): Number of levels of decomposition
        wave (str or pywt.Wavelet): Which wavelet to use. Can be a string to
            pass to pywt.Wavelet constructor, can also be a pywt.Wavelet class,
            or can be a two tuple of array-like objects for the analysis low and
            high pass filters.
        mode (str): 'zero', 'symmetric', 'reflect' or 'periodization'. The
            padding scheme. PyWavelets uses only periodization so we use this
            as our default scheme.
        """
    def __init__(self, J=1, wave='db1', mode='periodization'):
        super().__init__()
        if isinstance(wave, str):
            wave = pywt.Wavelet(wave)
        if isinstance(wave, pywt.Wavelet):
            h0_col, h1_col = wave.dec_lo, wave.dec_hi
            h0_row, h1_row = h0_col, h1_col
        else:
            if len(wave) == 2:
                h0_col, h1_col = wave[0], wave[1]
                h0_row, h1_row = h0_col, h1_col
            elif len(wave) == 4:
                h0_col, h1_col = wave[0], wave[1]
                h0_row, h1_row = wave[2], wave[3]

        # Prepare the filters
        filts = lowlevel.prep_filt_afb2d(h0_col, h1_col, h0_row, h1_row)
        self.h0_col = nn.Parameter(filts[0], requires_grad=False)
        self.h1_col = nn.Parameter(filts[1], requires_grad=False)
        self.h0_row = nn.Parameter(filts[2], requires_grad=False)
        self.h1_row = nn.Parameter(filts[3], requires_grad=False)

        self.J = J
        self.mode = mode

    def forward(self, x):
        """ Forward pass of the SWT.
        Args:
            x (tensor): Input of shape :math:`(N, C_{in}, H_{in}, W_{in})`
        Returns:
            List of coefficients for each scale. Each coefficient has
            shape :math:`(N, C_{in}, 4, H_{in}, W_{in})` where the extra
            dimension stores the 4 subbands for each scale. The ordering in
            these 4 coefficients is: (A, H, V, D) or (ll, lh, hl, hh).
        """
        ll = x
        coeffs = []
        # Do a multilevel transform
        filts = (self.h0_col, self.h1_col, self.h0_row, self.h1_row)
        for j in range(self.J):
            # Do 1 level of the transform
            y = lowlevel.afb2d_atrous(ll, filts, self.mode, 2**j)
            coeffs.append(y)
            ll = y[:,:,0]

        return coeffs
 
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        
        # Convolution 1
        #b=0
        #a=torch.pow(b,2)+b+0.1
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        #self.act = torch.clamp(a,-1,1)
        self.tanh1= nn.Tanh()
        
        self.cnn2 = nn.Conv2d(in_channels=32, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.tanh2 = nn.Tanh()
        
        # Max pool 1
        #self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        #self.cnn3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        #self.tanh3 = nn.Tanh()
          
        self.cnn4 = nn.Conv2d(in_channels=128, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.tanh4 = nn.Tanh()
        
       
        #self.cnn5 = nn.Conv2d(in_channels=256, out_channels=32, kernel_size=3, stride=1, padding=1)
        #self.tanh5 = nn.Tanh()
       # self.cnn6 = nn.Conv2d(in_channels=64, out_channels=16, kernel_size=3, stride=1, padding=1)
        #self.tanh6 = nn.Tanh()
        # Convolution 2
        self.cnnout = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.tanhout = nn.Tanh()
        
        # Max pool 2
       # self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        
        # Fully connected 1 (readout)
        #self.fc1 = nn.Linear(1 * 256 * 32768, 1 * 256 * 32768) 
    
    def forward(self, x):
        # Convolution 1
        out = self.cnn1(x)
        #out =torch.clamp((torch.pow(out,2)+out+0.1),-1,1)
        ###############################################
        # Max pool 1
        #out = self.maxpool1(out)
        out=self.tanh1(out)
        
        out = self.cnn2(out)
        out = self.tanh2(out)
        
        #out = self.cnn3(out)
        #out = self.tanh3(out)
        #out = self.relu3(out)
        #out = self.cnn4(out)
        #out = self.relu4(out)
        out = self.cnn4(out)
        out = self.tanh4(out)
        #out = self.cnn5(out)
        #out = self.tanh5(out)
        #out = self.cnn6(out)
        #out = self.tanh6(out)
       
        
       
        # Convolution 2 
        out = self.cnnout(out)
        out = self.tanhout(out)
        
        # Max pool 2 
        #out = self.maxpool2(out)
        
        # Resize
        # Original size: (100, 32, 7, 7)
        # out.size(0): 100
        # New out size: (100, 32*7*7)
        #out = out.view(out.size(0), -1)

        # Linear function (readout)
        #out = self.fc1(out)
        
        return out
    
    


'''
STEP 4: INSTANTIATE MODEL CLASS
'''

model = CNNModel()
summary = ModelSummary(model, input_size=(1,256,5120), bytes=4, batch_size=1, device="cuda")
summary.getSummary()
#print(model)

#######################
#  USE GPU FOR MODEL  #
#######################

if torch.cuda.is_available():
    model=model.cuda()

'''
STEP 5: INSTANTIATE LOSS CLASS
'''
criterion = nn.MSELoss()


'''
STEP 6: INSTANTIATE OPTIMIZER CLASS
'''
learning_rate = 0.01

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

'''
STEP 7: TRAIN THE MODEL
'''
iter = 0

for epoch in range(num_epochs):
    #for  i,(images, labels) in enumerate(train_loader):
    for i,(images, labels)  in enumerate(train_loader, 0):
        
        #######################
        #  USE GPU FOR MODEL  #
        #######################
        '''
        if torch.cuda.is_available():
           
            images = Variable(images.cuda()).permute(0,1,2,3)
            labels = Variable(labels.cuda()).permute(0,1,2,3)
            #print(images.size())
        else:
            images = Variable(images).permute(0,1,2,3)
            labels = Variable(labels).permute(0,1,2,3)
        '''
            
        images = Variable(images).permute(0,1,2,3)
        labels = Variable(labels).permute(0,1,2,3)
        #print(images.size())
        # Clear gradients w.r.t. parameters
        
        optimizer.zero_grad()
        
        # Forward pass to get output/logits
        #model = CNNModel()
        
        dwt=DWTForward()
        images1,images2=dwt(images)
        
        #images1=np.array(images1, dtype=np.float32)
        #images=np.asarray(images,dtype=np.float32)
        #images= torch.tensor(images1)
        images1=torch.squeeze(images1)
        images1= images1.unsqueeze(0)
        #print(images1.size())
        #images2=np.asarray(images2,dtype=np.float32)
        #images2= torch.tensor(images2)
        
        images2=torch.squeeze(images2)
        #print(images2.size())
        images1= images1.unsqueeze(0)
        images2= images2.unsqueeze(0)
        images2= images2.unsqueeze(0)
        #print(images1.size())
        #print(images2.size())
        
        idwt=DWTInverse()
        images=idwt((images1,[images2]))
        #images = torch.cat((images1, images2), 0)
        #images= images.unsqueeze(0)
        images=images.cuda()
        #print(images.size())
        #print(images.device)
        #images=images
        model=model.cuda()
        outputs = model(images)
        #outputs=DWTInverse(outputs)
        
        # Calculate Loss: softmax --> cross entropy loss
            
        labels1,labels2=dwt(labels)
        labels1=torch.squeeze(labels1)
        labels1= labels1.unsqueeze(0)
        labels2=torch.squeeze(labels2)
        labels1= labels1.unsqueeze(0)
        labels2= labels2.unsqueeze(0)
        labels2= labels2.unsqueeze(0)
        
        
        #labels = torch.cat((labels1, labels2), 0)
        #labels= labels.unsqueeze(0)
        images=idwt((labels1,[labels2]))
        labels=labels.cuda()
        
        
        loss = criterion(outputs, labels)
        
        #loss=
        
        # Getting gradients w.r.t. parameters
        loss.backward()
        
        # Updating parameters
        optimizer.step()
        
        print('[%d/%d][%d/%d] Loss: %.4f' % ((epoch+1), num_epochs, i, len(train_loader), loss.data.item())) # We print les losses of the convolutional translator.


#torch.save(model.state_dict(), 'cnnitermodel.ckpt')
    

''' 

print(outputs.size())
outputs.view(-1, 256,2560).size() 
for index,img in enumerate(outputs):
    print(img.size())
    #img=img.cpu
    imagef=img.cpu().data.numpy()
    #imagef.reshape(256,2560,2)
    ca=imagef[0:1,0:256,0:2560]
    cd=imagef[1:2,0:256,0:2560]
    #ca,cd=np.dsplit(imagef, 2)
    imageff=pywt.idwt(ca,cd, 'haar')
    print(np.shape(imageff))
    imageff=imageff.reshape(256,256,20)
    
   # k=Nifti2Dto3D(imageff)
    index+=1
    FileSave(imageff,'iterftrailtorch_brain_imagesfinal1'+str(index)+'.img')   
'''
    
model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
with torch.no_grad():
    correct = 0
    total = 0
    total_val_loss=0
    fullysampled_val=[]
    for images, labels in test_loader:
         if torch.cuda.is_available():
            images = Variable(images.cuda()).permute(0,1,2,3)
            labels = Variable(labels.cuda()).permute(0,1,2,3)
            vole = model(images)
            img=vole
            lab=labels
            #print(vole.size())
            #print(labels.size())
            val_loss= criterion(img, lab)
            print('val loss')
            print(val_loss.data.item())
         
                #val_loss= criterion(images, labels)
            img=img.cpu().data.numpy()
            lab=lab.cpu().data.numpy()
                 #imagef=imagef.reshape((256,5120))
                 
                 
            
           
            #imagevol=imagevol.reshape(1,256,5120)
            #imgfull=imgfull.reshape(1,256,5120)
            for im1,im2 in zip(img,lab):
                im1=im1.reshape(256,256,20)
                im2=im2.reshape(256,256,20)
    
   # k=Nifti2Dto3D(imageff)
                #index+=1
                FileSave(im1,'iter1_test_brain_imagesfinal1'+str(123)+'.img') 
                FileSave(im2,'iter2_test_brain_imagesfinal1'+str(456)+'.img') 
            
               # im1=im1.reshape(256,256,20)
                #im2=im2.reshape(256,256,20)
            
               
                #mse_const = mse(img, img_const)
                
                ssim_const = ssim(im1,im2)
                  
                print('ssim')
                print(ssim_const)   

         
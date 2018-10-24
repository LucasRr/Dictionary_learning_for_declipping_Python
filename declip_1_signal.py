"""
 This code compares 4 different approaches for signal declipping:

 - solvers.IHT_inpainting discards the clipped samples and performs sparse decomposition 
     on the unclipped samples, using IHT and a fixed DCT dictionary
 - solvers.DictionaryLearning_inpainting discards the clipped samples and performs a 
     gradient descent-based dictionary learning on the unclipped samples
 - solvers.consistentIHT performs consistent IHT for declipping, using a fixed DCT
 dictionary [1]
 - solvers.consistentDictionaryLearning performs consistent dictionary learning for
 signal declipping, as proposed in [2]


 References:
 [1]: Consistent iterative hard thresholding for signal declipping, 
     S. Kitic, L. Jacques, N. Madhu, M. P. Hopwood, A. Spriet, C. De Vleeschouwer, ICASSP, 2013
 
 [2]: Consistent dictionary learning for signal declipping, 
     L. Rencker, F. Bach, W. Wang, M. D. Plumbley,
     Latent Variable Analysis and Signal Separation (LVA/ICA), Guildford, UK, 2018
 
 --------------------- 

 Author: Lucas Rencker
         Centre for Vision, Speech and Signal Processing (CVSSP), University of Surrey

 Contact: lucas.rencker@surrey.ac.uk
                    
 Last update: 01/05/18
 
 This code is distributed under the terms of the GNU Public License version 3 
 (http://www.gnu.org/licenses/gpl.txt).
"""                

import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile
import solvers
import utils as u

#import importlib
#importlib.reload(solvers)
#importlib.reload(u)

plt.close('all')

#%% Parameters

param = {}
param["N"] = 256  # size of frames
param["hop"] = 0.25*param["N"]  # hop size
param["redundancyFactor"] = 2  # redundancy of dictionary
param["M"] = param["N"] * param["redundancyFactor"]  # number of atoms
param["wa"] = u.wHamm  # analysis window
param["ws"] = param["wa"]  # synthesis window

M = param["M"]

#%% Generate DCT dictionary:

D_DCT = u.DCT_dictionary(param)

#%% Read signal

filename = 'glockenspiel.wav'

fs, x = scipy.io.wavfile.read(filename)

x = x/np.max(abs(x))  # normalize signal

# plt.plot(x)

#%% Clip signal:

SNRInput = 5  # desired input SNR
y, ClippingLevel = u.clip_signal(x, SNRInput)

#plt.plot(x)
#plt.plot(y)

SNRin = u.SNR(x,y)
print('Input SNR: %.3f dB' % SNRin)
 
#%% Decompose signal into overlapping time-frames:

Y = u.signal2frames(y,param)
Nframes = Y.shape[1]

# crop signals:
L = int((Nframes-1)*param["hop"]+param["N"])
y = y[:L]
x = x[:L]

#%% Detect reliable samples:

# Detect clipping level:
ClippingLevel = max(abs(y))

reliable_samples = np.logical_and(y<ClippingLevel,y>-ClippingLevel)
reliable_samples_mat = u.binary_vec2mat(reliable_samples,param)

clipped_samples = np.logical_not(reliable_samples)
clipped_samples_mat = np.logical_not(reliable_samples_mat)

SNRin_clipped = u.SNR(x[clipped_samples],y[clipped_samples])

print('%.1f percent of clipped samples' % (sum(1*clipped_samples)/x.size*100))

#%% Reconstruct signal using IHT for inpainting:

print('IHT for inpainting:')

alg_param={}
alg_param["K"] = 32 # number of non-zero coefficients
alg_param["Nit"] = 50 # max number of iterations
alg_param["loud"] = 0 # 1 to print the results
alg_param["A_init"] = np.zeros((M,Nframes)) # initialize sparse matrix

A, cost = solvers.IHT_inpainting(Y,clipped_samples_mat,D_DCT,alg_param)

X_est_IHT = D_DCT@A
x_est_IHT = u.frames2signal(X_est_IHT,param)

#plt.plot(np.log(cost))
#plt.title('Objective')

SNRout_IHT = u.SNR(x,x_est_IHT)
SNRout_clipped = u.SNR(x[clipped_samples],x_est_IHT[clipped_samples])

print('SNRout: %.3f dB' % SNRout_IHT)
print('SNR clipped improvement: %.3f dB' % (SNRout_clipped-SNRin_clipped))

plt.figure()
plt.plot(x,color="blue", linewidth=1.0, linestyle="-", label="clean")
plt.plot(x_est_IHT,color="red", linewidth=1.0, linestyle="-", label="estimate")
plt.plot(y,color="green", linewidth=1.0, linestyle="--", label="clipped")
plt.xlim(0, L)
plt.legend()
plt.title("IHT for inpainting: SNR = %.2f dB" % SNRout_IHT)
plt.tight_layout()

#%% Reconstruct signal using dictionary learning for inpainting:

print('Dictionary learning for inpainting:')

# DL parameters:
paramDL={}
paramDL["K"] = 32 # number of non-zero coefficients 
paramDL["Nit"] = 50 # number of iterations
paramDL["Nit_sparse_coding"] = 20 # number of iterations sparse coding step
paramDL["Nit_dict_update"] = 20 # number of iterations dictionary update step
paramDL["warm_start"] = 1 # 1 to perform warm start at each iteration
paramDL["A_init"] = np.zeros((M,Nframes)) # initialize sparse coefficient matrix
paramDL["D_init"] = D_DCT # initialize dictionary
paramDL["loud"] = 1 # print results

D_DL, A, cost = solvers.DictionaryLearning_inpainting(Y,clipped_samples_mat,paramDL)

X_est_DL = D_DL@A
x_est_DL = u.frames2signal(X_est_DL,param)

#plt.figure()
#plt.plot(np.log(cost))
#plt.title('Objective')

SNRout_DL = u.SNR(x,x_est_DL)
SNRout_clipped = u.SNR(x[clipped_samples],x_est_DL[clipped_samples])

print('SNRout: %.3f dB' % SNRout_DL)
print('SNR clipped improvement: %.3f dB' % (SNRout_clipped-SNRin_clipped))

plt.figure()
plt.plot(x,color="blue", linewidth=1.0, linestyle="-", label="clean")
plt.plot(x_est_DL,color="red", linewidth=1.0, linestyle="-", label="estimate")
plt.plot(y,color="green", linewidth=1.0, linestyle="--", label="clipped")
plt.xlim(0, L)
plt.legend()
plt.title("DL for inpainting: SNR = %.2f dB" % SNRout_DL)
plt.tight_layout()


#%% Reconstruct signal using consistent IHT:

print('consistent IHT for declipping:')

alg_param={}
alg_param["K"] = 32 # number of non-zero coefficients
alg_param["Nit"] = 50 # max number of iterations
alg_param["loud"] = 1 # 1 to print the results
alg_param["A_init"] = np.zeros((M,Nframes)) # initialize sparse matrix

A, cost = solvers.consistentIHT(Y,clipped_samples_mat,D_DCT,alg_param)

X_est_consIHT = D_DCT@A
x_est_consIHT = u.frames2signal(X_est_consIHT,param)

#plt.figure()
#plt.plot(np.log(cost))
#plt.title('Objective')

SNRout_consIHT = u.SNR(x,x_est_consIHT)
SNRout_clipped = u.SNR(x[clipped_samples],x_est_consIHT[clipped_samples])

print('SNRout: %.3f dB' % SNRout_consIHT)
print('SNR clipped improvement: %.3f dB' % (SNRout_clipped-SNRin_clipped))

plt.figure()
plt.plot(x,color="blue", linewidth=1.0, linestyle="-", label="clean")
plt.plot(x_est_consIHT,color="red", linewidth=1.0, linestyle="-", label="estimate")
plt.plot(y,color="green", linewidth=1.0, linestyle="--", label="clipped")
plt.xlim(0, L)
plt.legend()
plt.title("Consistent IHT: SNR = %.2f dB" % SNRout_consIHT)
plt.tight_layout()

#%% Reconstruct signal using consistent dictionary learning:

print('Proposed consistent dictionary learning:')

# DL parameters:
paramDL={}
paramDL["K"] = 32 
paramDL["Nit"] = 50 # number of iterations
paramDL["Nit_sparse_coding"] = 20 # number of iterations sparse coding step
paramDL["Nit_dict_update"] = 20 # number of iterations dictionary update step
paramDL["warm_start"] = 1 # 1 to perform warm start at each iteration
paramDL["A_init"] = np.zeros((M,Nframes)) # initialize sparse coefficient matrix
paramDL["D_init"] = D_DCT # initialize dictionary
paramDL["loud"] = 1 # print results

D_consDL, A, cost = solvers.consistentDictionaryLearning(Y,clipped_samples_mat,paramDL)

X_est_consDL = D_consDL@A
x_est_consDL = u.frames2signal(X_est_consDL,param)

#plt.figure()
#plt.plot(np.log(cost))
#plt.title('Objective')

SNRout_consDL = u.SNR(x,x_est_consDL)
SNRout_clipped = u.SNR(x[clipped_samples],x_est_consDL[clipped_samples])

print('SNRout: %.3f dB' % SNRout_consDL)
print('SNR clipped improvement: %.3f dB' % (SNRout_clipped-SNRin_clipped))

plt.figure()
plt.plot(x,color="blue", linewidth=1.0, linestyle="-", label="clean")
plt.plot(x_est_consDL,color="red", linewidth=1.0, linestyle="-", label="estimate")
plt.plot(y,color="green", linewidth=1.0, linestyle="--", label="clipped")
plt.xlim(0, L)
plt.legend()
plt.title("Consistent DL: SNR = %.2f dB" % SNRout_consDL)
plt.tight_layout()

#%% Plots

# "zoom-in" to particular area:
samples = np.arange(46800,46900) 

# percentage of missing samples:
# sum(~reliable_samples(samples))/length(samples)*100

f, axarr = plt.subplots(2, 2)
axarr[0, 0].plot(samples, x[samples], label="clean")
axarr[0, 0].plot(samples, x_est_IHT[samples], label="estimate")
axarr[0, 0].plot(samples, y[samples], label="clipped")
axarr[0, 0].set_title(('IHT for inpainting: SNR = %.2f dB' % SNRout_IHT))
axarr[0, 0].legend()
axarr[0, 1].plot(samples, x[samples], label="clean")
axarr[0, 1].plot(samples, x_est_DL[samples], label="estimate")
axarr[0, 1].plot(samples, y[samples], label="clipped")
axarr[0, 1].set_title(('DL for inpainting: SNR = %.2f dB' % SNRout_DL))
axarr[0, 1].legend()
axarr[1, 0].plot(samples, x[samples], label="clean")
axarr[1, 0].plot(samples, x_est_consIHT[samples], label="estimate")
axarr[1, 0].plot(samples, y[samples], label="clipped")
axarr[1, 0].set_title(('Consistent IHT: SNR = %.2f dB' % SNRout_consIHT))
axarr[1, 0].legend()
axarr[1, 1].plot(samples, x[samples], label="clean")
axarr[1, 1].plot(samples, x_est_consDL[samples], label="estimate")
axarr[1, 1].plot(samples, y[samples], label="clipped")
axarr[1, 1].set_title(('Consistent dictionary learning: SNR = %.2f dB' % SNRout_consDL))
axarr[1, 1].legend()
plt.tight_layout()

plt.show()

#%% Save results
 
# We can re-project on the unclipped samples to avoid extra distortion due
# to the sparse approximation:

#x_est_IHT[reliable_samples] = y[reliable_samples]
#x_est_DL[reliable_samples] = y[reliable_samples]
#x_est_consIHT[reliable_samples] = y[reliable_samples]
#x_est_consDL[reliable_samples] = y[reliable_samples]
#
#scipy.io.wavfile.write('clean.wav',fs,x)
#scipy.io.wavfile.write('clipped.wav',fs,y)
#scipy.io.wavfile.write('declipped_IHT.wav',fs,x_est_IHT)
#scipy.io.wavfile.write('declipped_DL.wav',fs,x_est_DL)
#scipy.io.wavfile.write('declipped_consIHT.wav',fs,x_est_consIHT)
#scipy.io.wavfile.write('declipped_consDL.wav',fs,x_est_consDL)

import numpy as np



def wHamm(N):
    """
    Hamming window of size N
    """
    return np.hamming(N)

def SNR(x, y):
    """
    compute SNR of y, using x as reference clean signal
    """
    snr = 10*np.log10(np.sum(x**2)/np.sum((x-y)**2))
    snr = min(snr, 80)  # define 80dB as "perfect" signal (to avoid inf)
    return snr

def hard_threshold(A,K):
    """
    Hard thresholding function. Keeps the K largest coefficients in each column of matrix A, 
    and sets the rest to zero
    """
    return A * (abs(A) >= np.sort(np.abs(A),0)[A.shape[0]-K,:])

def signal2frames(y, param):
    """
    divide signal into overlapping time frames
    """

    hop = param['hop']
    N = param['N']
    wa = param['wa'](param["N"])
    L = y.size  # size of signal

    Nframes = int((L-N)/hop)+1  # total number of frames

    iframes = [np.arange(int(hop*nframe), int(hop*nframe+N)) for nframe in range(Nframes)]
    iframes = np.vstack(iframes).T # index map

    return np.diag(wa).dot(y[iframes])


def frames2signal(Y, param):
    """
    Reconstruct signal from frames using overlap and add
    """
    N = param["N"]
    hop = param["hop"]
    wa = param["wa"](param["N"])
    ws = param["ws"](param["N"])
    
    Nframes = Y.shape[1]

    L = int((Nframes-1)*hop+N)  # size of reconstructed signal

    y_reconst = np.zeros(L)
    wNorm = np.zeros(L)
    
    iframes = [np.arange(int(hop*nframe), int(hop*nframe+N)) for nframe in range(Nframes)]
    iframes = np.vstack(iframes).T # index map

    Nframes = iframes.shape[1] # number of frames

    for nframe in range(Nframes):
        y_reconst[iframes[:,nframe]] = y_reconst[iframes[:,nframe]] + ws*Y[:,nframe]
        wNorm[iframes[:,nframe]] = wNorm[iframes[:,nframe]] + ws*wa

    return y_reconst/wNorm
 
def binary_vec2mat(binary_mask_vec, param):
    """
    Divide binary vector mask into matrix mask, in the same way as signal2frames
    """

    hop = param['hop']
    N = param['N']
    L = binary_mask_vec.size  # size of signal

    Nframes = int((L-N)/hop)+1  # total number of frames

    iframes = [np.arange(int(hop*nframe), int(hop*nframe+N)) for nframe in range(Nframes)]
    iframes = np.vstack(iframes).T # index map

    return binary_mask_vec[iframes]
 
def DCT_dictionary(param):
    """
    DCT dictionary of size NxM
    """
	
    N = param['N']
    M = param['M']
    wa = param['wa'](param["N"])

    t = np.arange(N)  # time 
    k = np.arange(M)  # frequency

    D = np.cos(np.pi/M * np.dot((t[np.newaxis, :].T+1/2), k[np.newaxis, :]+1/2))

    D = np.diag(wa).dot(D) # Analysis window 

    return	normalize_dictionary(D)
    
    
    
def normalize_dictionary(D):
    """
    normalize dictionary
    """
    norm = np.sqrt(np.sum(D**2,0))
    return D/norm
    
    
    
def clip_signal(x, SNR_target):
    """
    %  Clip a signal at a given SNR level, using bisection method.
    %  This method should be precise up to +/- 0.001 dB
    """
    
    ClippingLevel1 = 0;
    ClippingLevel2 = max(abs(x));
    SNRtmp = float("inf");
    it = 0;
    
    # Search between ClippingLevel1 and ClippingLevel2:
    
    while abs(SNRtmp-SNR_target) > 0.001 and it < 20:
        
        it = it + 1
        
        ClippingLevel = (ClippingLevel1+ClippingLevel2)/2
        y = np.maximum(np.minimum(x,ClippingLevel),-ClippingLevel)  # clip signal
        SNRtmp = SNR(x,y)  # check SNR

        # update search interval
        if SNRtmp < SNR_target:
            ClippingLevel1 = ClippingLevel
        else:
            ClippingLevel2 = ClippingLevel
                
    return y, ClippingLevel

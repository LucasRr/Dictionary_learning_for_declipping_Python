# Consistent dictionary learning for signal declipping

<!--- This is a markdown README generated for the github page. For a human readable readme, see README.txt --->


This is an implementation of the algorithm proposed in:

*Consistent dictionary learning for signal declipping*, L. Rencker, F. Bach, W. Wang, M. D. Plumbley, 
Latent Variable Analysis and Signal Separation (LVA/ICA), Guildford, UK, 2018

The paper can be found at http://epubs.surrey.ac.uk/846156/1/Consistent_DL_for_signal_declipping.pdf.

## Author:
Lucas Rencker,  
*Centre for Vision, Speech and Signal Processing (CVSSP)*, University of Surrey, UK  
Contact: lucas \[dot\] rencker \[at\] surrey.ac.uk  

## Quick demo:

Clipping, or saturation, is a common nonlinear distortion in signal processing. Clipping occurs when the signal reaches a maximum threshold  and the waveform is truncated.

Declipping aims at recovering the clipped samples using the surrounding unclipped samples. 

This code performs declipping using 4 different approaches:
* **Iterative Hard Thresholding (IHT) for inpainting:** discards the clipped sample and performs sparse coding on the unclipped samples using IHT and a fixed DCT dictionary
* **Dictionary learning for inpainting:** discards the clipped samples and performs dictionary learning on the unclipped samples
* **Consistent IHT:** performs consistent IHT using a fixed DCT dictionary \[1\]
* **Consistent dictionary learning:** performs consistent dictionary learning using the algorithm proposed in \[2\]

Run `declip_1_signal.py` for an example.

## References:
\[1\]: Consistent iterative hard thresholding for signal declipping, 
   S. Kitic, L. Jacques, N. Madhu, M. P. Hopwood, A. Spriet, C. De Vleeschouwer, ICASSP, 2013

\[2\]: Consistent dictionary learning for signal declipping, 
    L. Rencker, F. Bach, W. Wang, M. D. Plumbley,
    Latent Variable Analysis and Signal Separation (LVA/ICA), Guildford, UK, 2018



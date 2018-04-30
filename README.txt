
This code provides an implementation of the consistent dictionary learning algorithm proposed in:

Consistent dictionary learning for signal declipping, 
L. Rencker, F. Bach, W. Wang, M. D. Plumbley,
Latent Variable Analysis and Signal Separation (LVA/ICA), Guildford, UK, 2018


============ Author: =================
Lucas Rencker
Centre for Vision, Speech and Signal Processing (CVSSP), University of Surrey

Contact: lucas.rencker@surrey.ac.uk


============ Usage: ==================

You can run Declip_1_signal.m in the Experiment/ folder for a demo.

This demo compares 4 different approaches for declipping:

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


This code has been implemented and tested using Python version 3.6.4.

============ License: ================

Consistent dictionary learning for signal declipping,
Copyright (C) 2018  Lucas Rencker

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.





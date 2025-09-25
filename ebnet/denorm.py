"""
Utility functions for denormalizing labels/predictions. 

MIT License
Copyright (c) 2025 hutchresearch

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import torch
import numpy as np

def denormalize_labels(
    labels: torch.Tensor,
) -> torch.Tensor:

    denorm_labels = torch.zeros_like(labels)
    denorm_labels[:, 0] = 10 ** ((labels[:, 0] * 0.4) + 3.85)
    denorm_labels[:, 1] = 10 ** ((labels[:, 1] * 0.4) + 3.85)
    denorm_labels[:, 2] = 10 ** (labels[:, 2])
    denorm_labels[:, 3] = 10 ** (labels[:, 3])
    denorm_labels[:, 4] = labels[:, 4]
    denorm_labels[:, 5] = labels[:, 5]
    denorm_labels[:, 6] = 10 ** (labels[:, 6] * 1.5)
    denorm_labels[:, 7] = 10 ** (labels[:, 7] * 1.5)
    denorm_labels[:, 8] = 10 ** (labels[:, 8] * 2)
    denorm_labels[:, 9] = 10 ** (labels[:, 9] * 2)
    denorm_labels[:, 10] = 10 ** ((labels[:, 10] * 2) + 1.5)
    denorm_labels[:, 11] = labels[:, 11]
    denorm_labels[:, 12] = 90.1 - (10 ** (labels[:, 12] + 0.6))
    denorm_labels[:, 13] = labels[:, 13] * 100
    denorm_labels[:, 14] = (10 ** (labels[:, 14] - 0.7)) - 0.02
    denorm_labels[:, 15] = labels[:, 15]
    denorm_labels[:, 16] = labels[:, 16]
    denorm_labels[:, 17] = 10 ** (labels[:, 17] + 1.5)
    denorm_labels[:, 18] = 10 ** (labels[:, 18] + 1.5)
    denorm_labels[:, 19] = labels[:, 19]
    denorm_labels[:, 20] = labels[:, 20]

    return denorm_labels

def denormalize_std(
    std_devs: torch.Tensor, 
    labels: torch.Tensor,      
) -> torch.Tensor:
    denorm_labels = denormalize_labels(labels)
    denorm_std = torch.zeros_like(labels)

    denorm_std[:,0]=(std_devs[:,0]*0.4)*denorm_labels[:,0]*np.log(10)
    denorm_std[:,1]=(std_devs[:,1]*0.4)*denorm_labels[:,1]*np.log(10)
    denorm_std[:,2]=std_devs[:,2]*denorm_labels[:,2]*np.log(10)
    denorm_std[:,3]=std_devs[:,3]*denorm_labels[:,3]*np.log(10)
    denorm_std[:,4]=std_devs[:,4]
    denorm_std[:,5]=std_devs[:,5]
    denorm_std[:,6]=(std_devs[:,6]*1.5)*denorm_labels[:,6]*np.log(10)
    denorm_std[:,7]=(std_devs[:,7]*1.5)*denorm_labels[:,7]*np.log(10)
    denorm_std[:,8]=(std_devs[:,8]*2)*denorm_labels[:,8]*np.log(10)
    denorm_std[:,9]=(std_devs[:,9]*2)*denorm_labels[:,9]*np.log(10)
    denorm_std[:,10]=(std_devs[:,10]*2)*denorm_labels[:,10]*np.log(10)
    denorm_std[:,11]=std_devs[:,11]
    denorm_std[:,12]=std_devs[:,12]*10**(labels[:,12]+0.6)*np.log(10)
    denorm_std[:,13]=std_devs[:,13]*100
    denorm_std[:,14]=std_devs[:,14]*10**(labels[:,14]-0.7)*np.log(10)
    denorm_std[:,15]=std_devs[:,15]
    denorm_std[:,16]=std_devs[:,16]
    denorm_std[:,17]=(std_devs[:,17])*denorm_labels[:,17]*np.log(10)
    denorm_std[:,18]=(std_devs[:,18])*denorm_labels[:,18]*np.log(10)
    denorm_std[:,19]=std_devs[:,19]
    denorm_std[:,20]=std_devs[:,20]
    
    return denorm_std
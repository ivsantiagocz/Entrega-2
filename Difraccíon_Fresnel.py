import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from mpl_toolkits.mplot3d import Axes3D
from numpy.fft import fft2, fftshift, ifft2 

def cmask(a,b,radius,N):
  t, s = np.ogrid[-a:N-a,-b:N-b]
  mask = (t)**2 + (s)**2 <= radius*radius
  array = np.zeros((N,N))
  array[mask] = 1.

  return array


wl= 633e-6
L=0.5 #medida en milimetros del objeto
N= 1024  #numero de particiones del objeto
dx= L/N #tamaño de particiones del objeto
nf=3


radio = 0.05

z=(radio**2)/(wl*nf)

Ut= np.zeros((N,N))*np.exp(-1j*0)
ut= cmask(int(N/2),int(N/2),radio/dx,N)*np.exp(-1j*0)
hh= np.zeros((N,N))*np.exp(-1j*0)

k=(2*np.pi)/wl

#if (z>=(N*(dx)**2)/wl):
print("z está en el dominio de aplicación de DD de FF")
for t in range(int(-N/2),int(N/2)):
            for s in range(int(-N/2),int(N/2)):
    
                h= np.exp(1j*k*((dx*t)**2 + (dx*s)**2)/(2*z))
                hh[t,s]=h 

h1=np.fft.fftshift(hh)
for t in range(int(-N/2),int(N/2)):
        for s in range(int(-N/2),int(N/2)):
                            
            Ut[t,s] = ut[t,s]*h1[t,s]
            
U1=np.fft.fft2(Ut)
Uf=np.fft.fftshift(U1)
U=abs(Uf*Uf)

#I = Image.fromarray(abs(U), 'P')
plt.figure(figsize=(7,4))
im1 = plt.imshow(np.angle(h1),cmap='gray',extent=(-L/2,L/2,-L/2,L/2))
im2 = plt.imshow(U,cmap='gray')
plt.ylabel('mm')
plt.xlabel('mm')
plt.colorbar(im1,shrink=0.5)
plt.show()


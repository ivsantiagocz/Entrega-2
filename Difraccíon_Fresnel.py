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
nf=4


radio = 0.05

z=(radio**2)/(wl*nf)

Ut= np.zeros((N,N))*np.exp(-1j*0)
ut= cmask(int(N/2),int(N/2),radio/dx,N)*np.exp(-1j*0)
hh= np.zeros((N,N))*np.exp(-1j*0)

k=(2*np.pi)/wl


for t in range(int(-N/2),int(N/2)):
            for s in range(int(-N/2),int(N/2)):
    
                h= np.exp(1j*k*((dx*t)**2 + (dx*s)**2)/(2*z))
                hh[t,s]=h 

h1=np.fft.fft2(hh)

uf=np.fft.fft2(ut)
Ut=uf*h1            
Uf=np.fft.ifft2(Ut)
#Uf=np.fft.fftshift(U1)
U=abs(Uf*Uf)

#I = Image.fromarray(abs(U), 'P')
plt.figure(figsize=(7,4))
#im1 = plt.imshow(np.angle(h1),cmap='gray',extent=(-L/2,L/2,-L/2,L/2))
plt.imshow(U,cmap='gray')

plt.show()


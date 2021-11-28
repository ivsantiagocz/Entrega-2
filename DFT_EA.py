#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from mpl_toolkits.mplot3d import Axes3D
from numpy.fft import fft2, fftshift, ifft2 



def cmask(radius,N):
  t, s = np.ogrid[0:N,0:N]
  mask = (t-int(N/2))**2 + (s-int(N/2))**2 <= radius*radius
  array = np.zeros((N,N))
  array[mask] = 1.

  return array

def icmask(radius,N):
  t, s = np.ogrid[0:N,0:N]
  mask = (t-int(N/2))**2 + (s-int(N/2))**2 <= radius*radius
  array = np.ones((N,N))
  array[mask] = 0.

  return array

def rmask(N,c,d):
  t, s = np.ogrid[0:N,0:N]
  mask1 = (t+0*s)<= int(N/2)+c
  mask2 = (t+ 0*s)>=int(N/2)-c
  mask3 = (0*t+s)<= int(N/2)+d
  mask4 = (0*t+ s)>=int(N/2)-d
  mask=mask1*mask2*mask3*mask4
  array = np.zeros((N,N))
  array[mask] = 1.

  return array


wl= 633e-6
L=5#medida en milimetros del objeto
N= 100  #numero de particiones del objeto
dx= L/N #tamaño de particiones del objeto
nf=4


#radio = 0.5
#radio_externo=1

#z=(radio**2)/(wl*nf)
z=L*dx/wl

Ut= np.zeros((N,N))*np.exp(-1j*0)
Utt= np.zeros((N,N))*np.exp(-1j*0)
ut=rmask(N,5,5)
#ut= u*np.exp(-1j*0)
hh= np.zeros((N,N))*np.exp(-1j*0)
print ("z=", z)


k=(2*np.pi)/wl
df=1/(N*dx) #cambiar
c=0

print("z está en el dominio de aplicación de DD de FF")

for t in range(0,N):
    for s in range(0,N):
        h2=0
        for n in range(-int(N/2),int(N/2)):
            for m in range(-int(N/2),int(N/2)):                
                h2=h2+(ut[n+int(N/2),m+int(N/2)]*(np.exp((-1j*2*np.pi/N)*(n*t+m*s))))
        h=np.exp(1j*k*z*np.sqrt(1-((df*wl)**2)*((t-int(N/2))**2 + (s-int(N/2))**2)))
        hh[t,s]=h
        Ut[t,s]=h2*h


for t in range(0,N):
    for s in range(0,N):
        h2=0
        for n in range(-int(N/2),int(N/2)):
            for m in range(-int(N/2),int(N/2)):                
                h2=h2+(Ut[n+int(N/2),m+int(N/2)]*(np.exp((1j*2*np.pi/N)*(n*t+m*s))))
        Utt[t,s]=h2
        
#hh=np.fft.fftshift(hh)                         
Uttf=np.fft.fftshift(Utt) 
U=abs(Utt*Utt)

plt.figure(figsize=(7,4))
im1 = plt.imshow(np.angle(hh),cmap='gray')
plt.colorbar(im1,shrink=0.5)
plt.show()

plt.figure(figsize=(7,4))
im2 = plt.imshow((ut),cmap='gray')
plt.show()


plt.figure(figsize=(7,4))
im3 = plt.imshow(U,cmap='gray')#,extent=(-L/2,L/2,-L/2,L/2))
plt.show()

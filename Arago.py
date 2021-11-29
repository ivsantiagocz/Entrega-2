#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from mpl_toolkits.mplot3d import Axes3D
from numpy.fft import fft2, fftshift, ifft2 

#%%
#se definen la máscaras
def cmask(a,b,radius,N):
  t, s = np.ogrid[-a:N-a,-b:N-b]
  mask = (t)**2 + (s)**2 <= radius*radius
  array = np.zeros((N,N))
  array[mask] = 1.

  return array

def icmask(a,b,radius,N):
  t, s = np.ogrid[-a:N-a,-b:N-b]
  mask = (t)**2 + (s)**2 <= radius*radius
  array = np.ones((N,N))
  array[mask] = 0.

  return array

#%%

wl= 633e-6 #longitud de onda 
L=5#medida en milimetros del objeto
N= 1024 #numero de particiones del objeto
dx= L/N #tamaño de particiones del objeto
nf=5 #número de Fresnel

radio = 0.5
radio_externo=2

z=(radio**2)/(wl*nf) #z en terminos del número de Fresnel

#se crea la mascara ut
ut=cmask(int(N/2),int(N/2),radio_externo/dx,N)*icmask(int(N/2),int(N/2),radio/dx,N)

hh= np.zeros((N,N))*np.exp(-1j*0)
print ("z=", z)

k=(2*np.pi)/wl

# se decide que método usar para la propagación según z
if (z>=(N*(dx)**2)/wl):
    print("z está en el dominio de aplicación de Transformada de Fresnel")
    for t in range(int(-N/2),int(N/2)):
                for s in range(int(-N/2),int(N/2)):
        
                    h= np.exp(1j*k*((dx*t)**2 + (dx*s)**2)/(2*z))  #fase esférica de Fresnel
                    hh[t,s]=h 
                             
    hh=np.fft.fftshift(hh)  #se centra la máscara y la función hh
    #ut=np.fft.fftshift(ut)
    Ut = ut*hh
                
    U1=np.fft.fft2(Ut)
    Uf=np.fft.fftshift(U1)   #por si necesita centrarse el resultado
    U=abs(Uf*Uf)
    
    plt.figure(figsize=(7,4))
    plt.imshow(np.angle(hh),cmap='gray',extent=(-L/2,L/2,-L/2,L/2))
    plt.ylabel('mm')
    plt.xlabel('mm')
    plt.show()

    plt.figure(figsize=(7,4))
    plt.imshow((ut),cmap='gray',extent=(-L/2,L/2,-L/2,L/2))
    plt.ylabel('mm')
    plt.xlabel('mm')
    plt.show()

    plt.figure(figsize=(7,4))
    plt.imshow(U,cmap='gray',extent=(-L/2,L/2,-L/2,L/2))
    plt.ylabel('mm')
    plt.xlabel('mm')
    plt.show()
                
else:
    print("z está en el domino de aplicación de Espectro Angular")
    for t in range(int(-N/2),int(N/2)):
                for s in range(int(-N/2),int(N/2)):
        
                    h= np.exp(1j*k*z*np.sqrt(1-(wl**2)*((t/(N*dx))**2 + (s/(N*dx))**2)))  #función de transferencia
                    hh[t,s]=h 

    h1=np.fft.fftshift(hh)
    gt=np.fft.fftshift(np.fft.fft2(ut))

                             
    Ut = gt*h1
    
    
    U1=np.fft.ifft2(Ut)
    #Uf=np.fft.fftshift(U1)
    UU=abs(U1*U1)


    plt.figure(figsize=(7,4))
    im1 = plt.imshow(np.angle(h1),cmap='gray',extent=(-L/2,L/2,-L/2,L/2))
    plt.ylabel('mm')
    plt.xlabel('mm')
    plt.colorbar(im1,shrink=0.5)
    plt.show()

    plt.figure(figsize=(7,4))
    im3 = plt.imshow((ut),cmap='gray',extent=(-L/2,L/2,-L/2,L/2))
    plt.ylabel('mm')
    plt.xlabel('mm')
    plt.colorbar(im1,shrink=0.5)
    plt.show()

    plt.figure(figsize=(7,4))
    im2 = plt.imshow(UU,cmap='gray',extent=(-L/2,L/2,-L/2,L/2))
    plt.ylabel('mm')
    plt.xlabel('mm')
    plt.show()

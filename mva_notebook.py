# -*- coding: utf-8 -*-


# Commented out IPython magic to ensure Python compatibility.
#load required libraries, set default colour map
# %matplotlib qt
import hyperspy.api as hs
import numpy as np
import matplotlib.pyplot as plt
plt.set_cmap("magma")

#load the SIs
s1 = hs.load("Control1.emi")
s2 = hs.load("Control2.emi")
s3 = hs.load("Aerosol1.emi")
s4 = hs.load("Aerosol2.emi")

#cut energy axis at 13 keV
s1.crop(axis=2, end=2600)
s2.crop(axis=2, end=2600)
s3.crop(axis=2, end=2600)
s4.crop(axis=2, end=2600)

#crop navigation axes
s1.crop(axis=0, start=13, end=88)
s2.crop(axis=0, start=15, end=90)
s3.crop(axis=0, start=25, end=115)
s4.crop(axis=0, start=17, end=107)

#rebin energy axis by 4
s1 = s1.rebin([75,500,650])
s2 = s2.rebin([75,500,650])
s3 = s3.rebin([90,500,650])
s4 = s4.rebin([90,500,650])

#standardise EDX signal intensity
s1 = s1*1.2
s2 = s2*1.1

#stack SIs
s = hs.stack([s1, s2, s3, s4], axis = 0)

#set title
s.metadata.set_item("General.title", 'MVA')

s.change_dtype('float32')

#PCA
s.decomposition(normalize_poissonian_noise=True)

s.plot_explained_variance_ratio(n=30)

s.plot_decomposition_results()

#put PCA factors into variable of type signal, named PCA_factors
PCA_factors=s.get_decomposition_factors()
#do the same for loadings
PCA_loadings=s.get_decomposition_loadings()

#save PCA factors as .tif
for i in [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]:
    PCA_factors.inav[i].plot()
    fig = plt.gcf()
    fig.savefig('PCA_factor_%i.tif'%i)

#save PCA loadings as .tif
for i in [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]:
    PCA_loadings.inav[i].plot()
    fig = plt.gcf()
    fig.savefig('PCA_loading_%i.tif'%i)

#save PCA factors for plotting if needed
PCA_factors.inav[0].data.tofile(file="PCA_factor_0", sep=",", format="%f")
PCA_factors.inav[1].data.tofile(file="PCA_factor_1", sep=",", format="%f")
PCA_factors.inav[2].data.tofile(file="PCA_factor_2", sep=",", format="%f")
PCA_factors.inav[3].data.tofile(file="PCA_factor_3", sep=",", format="%f")
PCA_factors.inav[4].data.tofile(file="PCA_factor_4", sep=",", format="%f")
PCA_factors.inav[5].data.tofile(file="PCA_factor_5", sep=",", format="%f")
PCA_factors.inav[6].data.tofile(file="PCA_factor_6", sep=",", format="%f")
PCA_factors.inav[7].data.tofile(file="PCA_factor_7", sep=",", format="%f")
PCA_factors.inav[8].data.tofile(file="PCA_factor_8", sep=",", format="%f")
PCA_factors.inav[9].data.tofile(file="PCA_factor_9", sep=",", format="%f")
PCA_factors.inav[10].data.tofile(file="PCA_factor_10", sep=",", format="%f")
PCA_factors.inav[11].data.tofile(file="PCA_factor_11", sep=",", format="%f")
PCA_factors.inav[12].data.tofile(file="PCA_factor_12", sep=",", format="%f")
PCA_factors.inav[13].data.tofile(file="PCA_factor_13", sep=",", format="%f")
PCA_factors.inav[14].data.tofile(file="PCA_factor_14", sep=",", format="%f")
PCA_factors.inav[15].data.tofile(file="PCA_factor_15", sep=",", format="%f")

#NMF
s.decomposition(True, algorithm="nmf", tol=1e-10, max_iter=1000, output_dimension=4)

s.plot_decomposition_results()

#put NMF-ed factors into variable of type signal, named NMF_factors
NMF_factors=s.get_decomposition_factors()
#do the same for loadings
NMF_loadings=s.get_decomposition_loadings()

#save NMF factors for plotting if needed
NMF_factors.inav[0].data.tofile(file="NMF_factor_0", sep=",", format="%f")
NMF_factors.inav[1].data.tofile(file="NMF_factor_1", sep=",", format="%f")
NMF_factors.inav[2].data.tofile(file="NMF_factor_2", sep=",", format="%f")
NMF_factors.inav[3].data.tofile(file="NMF_factor_3", sep=",", format="%f")

for i in [0,1,2,3]:
    NMF_factors.inav[i].plot()
    fig = plt.gcf()
    fig.savefig('NMF_factor_%i.tif'%i)

for i in [0,1,2,3]:
    NMF_loadings.inav[i].plot()
    fig = plt.gcf()
    fig.savefig('NMF_loading_%i.tif'%i)

#Elemental mapping

#build denoised model
sm=s.get_decomposition_model([0,1,2,3])

sm.set_signal_type("EDS_TEM")

sm.set_elements(['C','I','In','N','Pb','Sn'])

sm.set_lines(["C_Ka","I_La","In_La","N_Ka","Pb_La","Sn_La"])

sm.plot()

#construct background windows
bw = sm.estimate_background_windows(line_width=[5.0, 2.0])

#manually set position and width of BG windows

#C_Ka
bw[0,0] = 145
bw[0,1] = 201
bw[0,2] = 640
bw[0,3] = 696

#I_La
bw[1,0] = 2815.5
bw[1,1] = 2925
bw[1,2] = 5272
bw[1,3] = 5381.5

#In_La
bw[2,0] = 2815.5
bw[2,1] = 2925
bw[2,2] = 5272
bw[2,3] = 5381.5

#N_Ka
bw[3,0] = 145.0
bw[3,1] = 201
bw[3,2] = 640
bw[3,3] = 696.0

#Pb_La
bw[4,0] = 10045.2
bw[4,1] = 10224
bw[4,2] = 10820
bw[4,3] = 10988.9

#Sn_La
bw[5,0] = 2815.5
bw[5,1] = 2925
bw[5,2] = 5272
bw[5,3] = 5381.5

bw

#check that BG windows do not overlap with any peaks
sm.plot(background_windows=bw)

#extract line intensities
intensities=sm.get_lines_intensity(background_windows=bw, plot_result=False)

#remove unphysical negative signal intensity
intensities[0].data[intensities[0].data < 0] = 0
intensities[1].data[intensities[1].data < 0] = 0
intensities[2].data[intensities[2].data < 0] = 0
intensities[3].data[intensities[3].data < 0] = 0
intensities[4].data[intensities[4].data < 0] = 0
intensities[5].data[intensities[5].data < 0] = 0

kfactors = [2.452, #C_Ka
            3.413, #I_La
            2.620, #In_La
            3.856, #N_Ka
            4.374, #Pb_La
            2.653] #Sn_La

#quantification
maps=sm.quantification(intensities, method='CL', factors=kfactors, composition_units='atomic', plot_result=False)

#save quant maps as .tif
maps_img0 = maps[0].as_signal2D((0,1))
maps_img0.plot(vmin=0, vmax=100, cmap="magma")
fig = plt.gcf()
fig.savefig('C_Ka_map.tif')

maps_img1 = maps[1].as_signal2D((0,1))
maps_img1.plot(vmin=0, vmax=70, cmap="magma")
fig = plt.gcf()
fig.savefig('I_La_map.tif')

maps_img2 = maps[2].as_signal2D((0,1))
maps_img2.plot(vmin=0, vmax=30)
fig = plt.gcf()
fig.savefig('In_La_map.tif')

maps_img3 = maps[3].as_signal2D((0,1))
maps_img3.plot(vmin=0, vmax=30, cmap="magma")
fig = plt.gcf()
fig.savefig('N_Ka_map.tif')

maps_img4 = maps[4].as_signal2D((0,1))
maps_img4.plot(vmin=0, vmax=25, cmap="magma")
fig = plt.gcf()
fig.savefig('Pb_La_map.tif')

maps_img5 = maps[5].as_signal2D((0,1))
maps_img5.plot(vmin=0, vmax=20)
fig = plt.gcf()
fig.savefig('Sn_La_map.tif')

#extract quantified values for each element
C = maps[0]
I = maps[1]
In = maps[2]
N = maps[3]
Pb = maps[4]
Sn = maps[5]

#ratio maps construction
#only show pixels where percentages of I and Pb are non-zero
mask_I  = I > 0.01
mask_Pb = Pb > 0.01
maskI=I*mask_I
maskPb=Pb*mask_Pb
I_Pb=maskI/maskPb

#remove negative and infinite values
I_Pb.data = np.nan_to_num(I_Pb.data)
I_Pb.data[I_Pb.data < 0] = 0 
I_Pb.data[I_Pb.data > 1000000] = 1000000

I_Pb.metadata.set_item("General.title", 'I/Pb at%')
I_Pbimage = I_Pb.as_signal2D((0,1))
I_Pbimage = I_Pbimage.rebin([165,250])
I_Pbimage = I_Pbimage/4
I_Pbimage.plot(vmin=2.5, vmax=3.5, cmap="magma")
fig = plt.gcf()
fig.savefig('I_Pbratio.tif')

#ratio maps construction
#only show pixels where percentages of N and Pb are non-zero
mask_N  = N > 0.01
mask_Pb = Pb > 0.01
maskN=N*mask_N
maskPb=Pb*mask_Pb
N_Pb=maskN/maskPb

#remove negative and infinite values
N_Pb.data = np.nan_to_num(N_Pb.data)
N_Pb.data[N_Pb.data < 0] = 0 
N_Pb.data[N_Pb.data > 1000000] = 1000000

N_Pb.metadata.set_item("General.title", 'N/Pb at%')
N_Pbimage = N_Pb.as_signal2D((0,1))
N_Pbimage = N_Pbimage.rebin([165,250])
N_Pbimage = N_Pbimage/4
N_Pbimage.plot(vmin=0, vmax=2, cmap="magma")
fig = plt.gcf()
fig.savefig('N_Pbratio.tif')

histo_C = np.histogram(C, range=[0,100], bins=100, density=False)
histo_I = np.histogram(I, range=[0,100], bins=100, density=False)
histo_N = np.histogram(N, range=[0,100], bins=100, density=False)
histo_Pb = np.histogram(Pb, range=[0,100], bins=100, density=False)
histo_I_Pb = np.histogram(I_Pb, range=[0,5], bins=100, density=False)
histo_N_Pb = np.histogram(N_Pb, range=[0,5], bins=100, density=False)

histo_C[0].tofile(file="histo_C", sep=",", format="%f")
histo_I[0].tofile(file="histo_I", sep=",", format="%f")
histo_N[0].tofile(file="histo_N", sep=",", format="%f")
histo_Pb[0].tofile(file="histo_Pb", sep=",", format="%f")
histo_I_Pb[0].tofile(file="histo_I_Pb", sep=",", format="%f")
histo_N_Pb[0].tofile(file="histo_N_Pb", sep=",", format="%f")


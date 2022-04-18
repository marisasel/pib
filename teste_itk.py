#!/usr/bin/env python

import itk
#import argparse
import matplotlib.pyplot as plt
import cv2 as cv

#parser = argparse.ArgumentParser(
#    description="Adaptive Histogram Equalization Image Filter."
#)
#parser.add_argument("input_image")
#parser.add_argument("output_image")
#parser.add_argument("alpha", type=float)
#parser.add_argument("beta", type=float)
#parser.add_argument("radius", type=int)
#args = parser.parse_args()

Dimension = 2

PixelType = itk.ctype("unsigned char")
#PixelType = itk.UC
ImageType = itk.Image[PixelType, Dimension]

reader = itk.ImageFileReader[ImageType].New()
reader.SetFileName('D405923.dcm')


# FILTRO MEDIANA
meanFilter = itk.MeanImageFilter[ImageType, ImageType].New()
meanFilter.SetInput(reader.GetOutput())
meanFilter.SetRadius(5)

writer = itk.ImageFileWriter[ImageType].New()
writer.SetFileName('image_itk_mediana_3.dcm')
writer.SetInput(meanFilter.GetOutput())
writer.Update()

# FILTRO COM KERNEL GAUSSIANO
smoothFilter = itk.SmoothingRecursiveGaussianImageFilter[ImageType, ImageType].New()
smoothFilter.SetInput(reader.GetOutput())
smoothFilter.SetSigma(3) #ENTENDER EXATAMENTE O QUE É O SIGMA

writer = itk.ImageFileWriter[ImageType].New()
writer.SetFileName('image_itk_gaussiano_3.dcm')
writer.SetInput(smoothFilter.GetOutput())

writer.Update()

# THRESHOLD OTSU
thresholdFilter = itk.OtsuMultipleThresholdsImageFilter[ImageType, ImageType].New()
thresholdFilter.SetInput(reader.GetOutput())

thresholdFilter.SetNumberOfHistogramBins(2)
thresholdFilter.SetNumberOfThresholds(1)
thresholdFilter.SetLabelOffset(2)

rescaler = itk.RescaleIntensityImageFilter[ImageType, ImageType].New()
rescaler.SetInput(thresholdFilter.GetOutput())
rescaler.SetOutputMinimum(0)
rescaler.SetOutputMaximum(255)

writer = itk.ImageFileWriter[ImageType].New()
writer.SetFileName('image_itk_otsu_1.dcm')
writer.SetInput(rescaler.GetOutput())
writer.Update()

# THRESHOLD EM CIMA DAS IMAGENS COM FILTRO
#mediana3
mediana3 = itk.ImageFileReader[ImageType].New()
mediana3.SetFileName('image_itk_mediana_3.dcm')
thresholdFilter.SetInput(mediana3.GetOutput())

thresholdFilter.SetNumberOfHistogramBins(2)
thresholdFilter.SetNumberOfThresholds(1)
thresholdFilter.SetLabelOffset(2)

rescaler = itk.RescaleIntensityImageFilter[ImageType, ImageType].New()
rescaler.SetInput(thresholdFilter.GetOutput())
rescaler.SetOutputMinimum(0)
rescaler.SetOutputMaximum(255)

writer = itk.ImageFileWriter[ImageType].New()
writer.SetFileName('image_itk_otsu_1_mediana3.dcm')
writer.SetInput(rescaler.GetOutput())
writer.Update()

#mediana5
mediana5 = itk.ImageFileReader[ImageType].New()
mediana5.SetFileName('image_itk_mediana_5.dcm')
thresholdFilter.SetInput(mediana5.GetOutput())

thresholdFilter.SetNumberOfHistogramBins(2)
thresholdFilter.SetNumberOfThresholds(1)
thresholdFilter.SetLabelOffset(2)

rescaler = itk.RescaleIntensityImageFilter[ImageType, ImageType].New()
rescaler.SetInput(thresholdFilter.GetOutput())
rescaler.SetOutputMinimum(0)
rescaler.SetOutputMaximum(255)

writer = itk.ImageFileWriter[ImageType].New()
writer.SetFileName('image_itk_otsu_1_mediana5.dcm')
writer.SetInput(rescaler.GetOutput())
writer.Update()

#gaussiano1
gaussiano1 = itk.ImageFileReader[ImageType].New()
gaussiano1.SetFileName('image_itk_gaussiano_1.dcm')
thresholdFilter.SetInput(gaussiano1.GetOutput())

thresholdFilter.SetNumberOfHistogramBins(2)
thresholdFilter.SetNumberOfThresholds(1)
thresholdFilter.SetLabelOffset(2)

rescaler = itk.RescaleIntensityImageFilter[ImageType, ImageType].New()
rescaler.SetInput(thresholdFilter.GetOutput())
rescaler.SetOutputMinimum(0)
rescaler.SetOutputMaximum(255)

writer = itk.ImageFileWriter[ImageType].New()
writer.SetFileName('image_itk_otsu_1_gaussiano1.dcm')
writer.SetInput(rescaler.GetOutput())
writer.Update()

#gaussiano2
gaussiano2 = itk.ImageFileReader[ImageType].New()
gaussiano2.SetFileName('image_itk_gaussiano_2.dcm')
thresholdFilter.SetInput(gaussiano2.GetOutput())

thresholdFilter.SetNumberOfHistogramBins(2)
thresholdFilter.SetNumberOfThresholds(1)
thresholdFilter.SetLabelOffset(2)

rescaler = itk.RescaleIntensityImageFilter[ImageType, ImageType].New()
rescaler.SetInput(thresholdFilter.GetOutput())
rescaler.SetOutputMinimum(0)
rescaler.SetOutputMaximum(255)

writer = itk.ImageFileWriter[ImageType].New()
writer.SetFileName('image_itk_otsu_1_gaussiano2.dcm')
writer.SetInput(rescaler.GetOutput())
writer.Update()

#gaussiano3
gaussiano3 = itk.ImageFileReader[ImageType].New()
gaussiano3.SetFileName('image_itk_gaussiano_3.dcm')
thresholdFilter.SetInput(gaussiano3.GetOutput())

thresholdFilter.SetNumberOfHistogramBins(2)
thresholdFilter.SetNumberOfThresholds(1)
thresholdFilter.SetLabelOffset(2)

rescaler = itk.RescaleIntensityImageFilter[ImageType, ImageType].New()
rescaler.SetInput(thresholdFilter.GetOutput())
rescaler.SetOutputMinimum(0)
rescaler.SetOutputMaximum(255)

writer = itk.ImageFileWriter[ImageType].New()
writer.SetFileName('image_itk_otsu_1_gaussiano3.dcm')
writer.SetInput(rescaler.GetOutput())
writer.Update()

# EROSAO IMAGEM BINARIZADA OTSU - MUITO PEQUENO PRA USAR, SOME TUDO
#ReaderType = itk.ImageFileReader[ImageType]
#eroded = ReaderType.New()
#eroded.SetFileName('image_itk_otsu.dcm')

#StructuringElementType = itk.FlatStructuringElement[Dimension]
#structuringElement = StructuringElementType.Ball(1)

#ErodeFilterType = itk.BinaryErodeImageFilter[
#    ImageType, ImageType, StructuringElementType
#]
#erodeFilter = ErodeFilterType.New()
#erodeFilter.SetInput(eroded.GetOutput())
#erodeFilter.SetKernel(structuringElement)
#erodeFilter.SetForegroundValue(255)  # Intensity value to erode
#erodeFilter.SetBackgroundValue(0)  # Replacement value for eroded voxels

#WriterType = itk.ImageFileWriter[ImageType]
#writer = WriterType.New()
#writer.SetFileName('image_itk_otsu_com_erosao.dcm')
#writer.SetInput(erodeFilter.GetOutput())

#writer.Update()

# EQUALIZAÇÃO DE HISTOGRAMA - FICOU UM LIXO
#histogramEqualization = itk.AdaptiveHistogramEqualizationImageFilter.New(reader)
#histogramEqualization.SetAlpha(0)
#histogramEqualization.SetBeta(0)

#radius = itk.Size[Dimension]()
#radius.Fill(5)
#histogramEqualization.SetRadius(radius)

#itk.imwrite(histogramEqualization, 'image_itk.dcm')

#plt.imshow(histogramEqualization, cmap = plt.cm.gray)
#plt.show()
#plt.hist(histogramEqualization)
#plt.show()
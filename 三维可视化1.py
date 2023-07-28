#!/usr/bin/env python
# -*- coding:utf-8 -*-
import vtk
# 有三种方法可以实现
# Method 1:
'''
# 生成图像序列的文件名数组
fileArray = vtk.vtkStringArray()
for i in range(100):
    print(r'E:/xujia\project\learn_vtk\Head\head%03d.jpg'%(i+1))
    fileStr = str(r'E:/xujia/project/learn_vtk/Head/head%03d.jpg'%(i+1))
    fileArray.InsertNextValue(fileStr)
# 读取JPG序列图像
reader = vtk.vtkJPEGReader()
reader.SetFileNames(fileArray)

style = vtk.vtkInteractorStyle()
# 显示读取的JPG图像
imageViewer = vtk.vtkImageViewer2()
imageViewer.SetInputConnection(reader.GetOutputPort())

renderWindowInteractor = vtk.vtkRenderWindowInteractor()
renderWindowInteractor.SetInteractorStyle(style)

imageViewer.SetSlice(50)  # 默认显示第50个切片，即第5层
imageViewer.SetSliceOrientationToXY()
imageViewer.SetupInteractor(renderWindowInteractor)
imageViewer.Render()

imageViewer.GetRenderer().SetBackground(1.0, 1.0, 1.0)
imageViewer.SetSize(640, 480)
imageViewer.GetRenderWindow().SetWindowName('ReadSeriesImages')
renderWindowInteractor.Start()
'''
# Method 2:
'''
reader = vtk.vtkJPEGReader()
reader.SetFilePrefix(r'E:/xujia/project/learn_vtk/Head/head')   # 设置文件名相同的部分
reader.SetFilePattern('%s%03d.jpg')   # 设置图像文件名中的序号变化的部分
reader.SetDataExtent(0, 255, 0, 255, 1, 100)
reader.Update()

style = vtk.vtkInteractorStyle()

imageViewer = vtk.vtkImageViewer2()
imageViewer.SetInputConnection(reader.GetOutputPort())

readerWindowInteractor = vtk.vtkRenderWindowInteractor()
readerWindowInteractor.SetInteractorStyle(style)

imageViewer.SetSlice(50)
imageViewer.SetSliceOrientationToXY()
# imageViewer.SetSliceOrientationToXZ()
# imageViewer.SetSliceOrientationToYZ()
imageViewer.SetupInteractor(readerWindowInteractor)
imageViewer.Render()

imageViewer.SetSize(640, 480)
imageViewer.GetRenderWindow().SetWindowName('ReadSeriesImages')

readerWindowInteractor.Start()
'''
# Method 3:
append = vtk.vtkImageAppend()   # 做数据的合并操作
append.SetAppendAxis(1)   # 指定Z轴为读入的每层图像数据的堆叠方向

reader = vtk.vtkJPEGReader()
for i in range(23):
    reader.SetFileName(r'C:/Users/admin/Desktop/20220401/%2d.jpg'%(i+1))
    append.AddInputConnection(reader.GetOutputPort())

style = vtk.vtkInteractorStyle()

imageViewer = vtk.vtkImageViewer2()
imageViewer.SetInputConnection(reader.GetOutputPort())

renderWindowInteractor = vtk.vtkRenderWindowInteractor()
renderWindowInteractor.SetInteractorStyle(style)
imageViewer.SetSlice(50)
imageViewer.SetSliceOrientationToXY()
# imageViewer.SetSliceOrientationToXZ()
# imageViewer.SetSliceOrientationToYZ()
imageViewer.SetupInteractor(renderWindowInteractor)
imageViewer.Render()

imageViewer.SetSize(640, 480)
imageViewer.GetRenderWindow().SetWindowName('ReadSeriesImages')

renderWindowInteractor.Start()

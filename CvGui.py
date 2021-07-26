##@2020/7/18
##zh
# -*- coding: UTF-8 -*-
"""
    Ubuntu
"""

"""
现在两个类中的"self.XXX"表示的是同样的内容

从窗口传递参数的两种方式 ：
1、通过“self.对象”传入需要的地方
2、通过新建函数件接受参数。
注意：如果通过出发函数传入，两个 信号连接一个函数，可能会引发冲突

插入图片执行：
pyrcc5 -o img_rc.py img.qrc
"""

import sys
from PyQt5 import QtWidgets
import cv2 as cv
import numpy as np
##导入窗口构件
from mainwindows import Ui_MainWindow

# TODO many things
class MyPyQT_Form(QtWidgets.QMainWindow,Ui_MainWindow):

    def __init__(self):
        super(MyPyQT_Form, self).__init__()
        self.setupUi(self)
        self.Ration="2:1"
        self.k=None
        self.adap_thresh_type="GAUSSIAN_C"
        self.adap_binary_type="BINARY_INV"
        self.blur_type="GaussianBlur"
        self.retrieval_mode="EXTERNAL"
        self.approx_mode="SIMPLE"
        self.srcImg=None
        self.shape_fitting="Rect"
        self.k= self.lineEdit_8.text()


    def ReadSrcImg(self):
        path = self.lineEdit.text()
        self.img=cv.imread(str(path))
        self.srcImg=self.img
        #cv.imshow("",self.img)

    def GetSrcImg(self):
        try:
            self.ReadSrcImg()
        except:
            print("读取图片出错")
        cv.imshow("src",self.img)
        #self.textEdit_2.clear()

    def GetGrayImg(self):
        self.gray=cv.cvtColor(self.img,cv.COLOR_BGR2GRAY)
        self.img=self.gray
        cv.imshow("gray",self.gray)

    def GetResizeImg(self):
        #self.ReadSrcImg()
        if(self.Ration=="500*500"):
            self.Resize=cv.resize(self.img,(500,500),fx=0,fy=0)
        elif(self.Ration=="2:1"):
            self.Resize=cv.resize(self.img,(0,0),fx=0.5,fy=0.5)
        elif(self.Ration=="1:2"):
            self.Resize=cv.resize(self.img,(0,0),fx=2,fy=2)
        self.img= self.Resize
        self.srcImg=self.img
        cv.imshow("Resize",self.img)

    def GetBlurImg(self):
        #self.ReadSrcImg()
        kernel=self.lineEdit_2.text()
        #kernel=11
        if(self.blur_type=="GaussianBlur"):
            self.img = cv.GaussianBlur(self.img,(int(kernel),int(kernel)),0)
        elif(self.blur_type=="medianBlur"):
            self.img = cv.medianBlur(self.img,int(kernel))
        elif(self.blur_type=="双边滤波"):
            self.img = cv.bilateralFilter(src=self.img, d=0, sigmaColor=100, sigmaSpace=15)
        cv.imshow("Blur",self.img)

    def FindThresh(self):
        cv.namedWindow('image', flags=cv.WINDOW_NORMAL)
        cv.createTrackbar('num', 'image', 150, 255, lambda x: None)
        while True:
            num = cv.getTrackbarPos('num', 'image')
            _, threshold1 = cv.threshold(self.img, num, 255, cv.THRESH_BINARY)

            cv.imshow('image',threshold1)
            k=cv.waitKey(1)& 0xFF
            if(k==27):
                break
        ##按esc退出：k==27
            #waitKey(1)将显示一帧

    def GetAdaptThreImg(self):
        #self.img = cv.adaptiveThreshold(self.img, maxval, thresh_type, type, Block Size, C)
        #self.ReadSrcImg()
        maxval=self.lineEdit_9.text()
        Block_Size=self.lineEdit_10.text()
        maxval=int( maxval)
        Block_Size=int( Block_Size)
        if(self.adap_thresh_type=="MEAN_C"):
            if(self.adap_binary_type=="BINARY"):
                self.img= cv.adaptiveThreshold(self.img,maxval,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY,  Block_Size, 2,)
            elif(self.adap_binary_type=="BINARY_INV"):
                self.img= cv.adaptiveThreshold(self.img,maxval,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY_INV,  Block_Size, 2,)
        elif(self.adap_thresh_type=="GAUSSIAN_C"):
            if(self.adap_binary_type=="BINARY"):
                self.img= cv.adaptiveThreshold(self.img,maxval,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,  Block_Size, 2,)
            elif(self.adap_binary_type=="BINARY_INV"):
                self.img= cv.adaptiveThreshold(self.img,maxval,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY_INV,  Block_Size, 2,)
        cv.imshow("AdapImg",self.img)
        ##最后一个参数C表示与算法有关的参数，它是一个从均值或加权均值提取的常数，可以是负数。（具体见下面的解释）。

    def GetThreshImg(self):
        ##https://blog.csdn.net/qq_42505705/article/details/88380616
        #ret,img1 = cv2.threshold(img,30,120,cv2.THRESH_BINARY)
        #self.ReadSrcImg()
        thresh=self.lineEdit_3.text()
        maxval=self.lineEdit_4.text()
        thresh=int(thresh)
        maxval=int(maxval)
        # self.ReadSrcImg()
        # self.img=cv.cvtColor(self.img,cv.COLOR_BGR2GRAY)
        if(self.thresh_type=="BINARY"):
            ret,self.img=cv.threshold(self.img,thresh,maxval,cv.THRESH_BINARY)
        elif(self.thresh_type=="BINARY_INV"):
            ret,self.img=cv.threshold(self.img,thresh,maxval,cv.THRESH_BINARY_INV)
        elif(self.thresh_type=="TRUNC"):
            ret,self.img=cv.threshold(self.img,thresh,maxval,cv.THRESH_TRUNC)
        elif(self.thresh_type=="TOZERO"):
            ret,self.img=cv.threshold(self.img,thresh,maxval,cv.THRESH_TOZERO)
        elif(self.thresh_type=="TOZERO_INV"):
            ret,self.img=cv.threshold(self.img,thresh,maxval,cv.THRESH_TOZERO_INV)
        elif(self.thresh_type=="OTSU"):
            ret,self.img=cv.threshold(self.img,0,255,cv.THRESH_OTSU)
        elif(self.thresh_type=="TRIANGLE"):
            ret,self.img=cv.threshold(self.img,thresh,maxval,cv.THRESH_TRIANGLE)
        elif(self.thresh_type=="MASK"):
            ret,self.img=cv.threshold(self.img,thresh,maxval,cv.THRESH_MASK)
        cv.imshow("Thresh",self.img)

    def GetCannyImg(self):
        thresh1=self.lineEdit_5.text()
        thresh2=self.lineEdit_6.text()
        kernel=self.lineEdit_7.text()
        self.img = cv.Canny(self.img,int(thresh1),int(thresh2), int(kernel))
        cv.imshow("Canny",self.img)

    def GetErodedImg(self):
        self.img = cv.erode(self.img,cv.getStructuringElement(cv.MORPH_RECT, (int( self.k), int( self.k))))
        cv.imshow("Eroded",self.img)
    def GetDilateImg(self):
        self.img = cv.dilate(self.img,cv.getStructuringElement(cv.MORPH_RECT, (int( self.k), int( self.k))))
        cv.imshow("Dilate",self.img)
    def GetOpenImg(self):
        self.img = cv.erode(self.img,cv.getStructuringElement(cv.MORPH_RECT, (int( self.k), int( self.k))))
        self.img = cv.dilate(self.img,cv.getStructuringElement(cv.MORPH_RECT, (int( self.k), int( self.k))))
        cv.imshow("Open",self.img)
    def GetCloseImg(self):
        self.img = cv.dilate(self.img,cv.getStructuringElement(cv.MORPH_RECT, (int( self.k), int( self.k))))
        self.img = cv.erode(self.img,cv.getStructuringElement(cv.MORPH_RECT, (int( self.k), int( self.k))))
        cv.imshow("Close",self.img)

    def GetCLAHEImg(self):
        clipLimit=self.lineEdit_11.text()
        tileGridSize=self.lineEdit_12.text()
        clipLimit=int(clipLimit)
        tileGridSize=int(tileGridSize)
        clahe = cv.createCLAHE(clipLimit=clipLimit, tileGridSize=(tileGridSize, tileGridSize))
        self.img = clahe.apply(self.img)
        cv.imshow("Clahe",self.img)

    def GetContoursImg(self):
        #self.ReadSrcImg()
        #self.GetCannyImg()
        line_size=self.lineEdit_13.text()
        contours_number=self.lineEdit_14.text()
        line_size=int(line_size)
        contours_number=int(contours_number)
        if(self.retrieval_mode=="EXTERNAL"):
            if(self.approx_mode=="NONE"):
                img,contours,hierarchy = cv.findContours(self.img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)  # 轮廓检测
                self.srcImg= cv.drawContours(self.srcImg,contours, contours_number, (0,255,0), line_size)
            elif(self.approx_mode=="SIMPLE"):
                img,contours,hierarchy = cv.findContours(self.img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)  # 轮廓检测
                self.srcImg= cv.drawContours(self.srcImg,contours,contours_number, (0,0,255), line_size)
            elif(self.approx_mode=="TC89_L1"):
                img,contours,hierarchy = cv.findContours(self.img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_TC89_L1)  # 轮廓检测
                self.srcImg= cv.drawContours(self.srcImg,contours,contours_number, (0,255,0), line_size)
        elif(self.retrieval_mode=="LIST"):
            if(self.approx_mode=="NONE"):
                img,contours,hierarchy = cv.findContours(self.img, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)  # 轮廓检测
                self.srcImg= cv.drawContours(self.srcImg,contours,contours_number, (0,255,0), line_size)
            elif(self.approx_mode=="SIMPLE"):
                img,contours,hierarchy = cv.findContours(self.img, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)  # 轮廓检测
                self.srcImg= cv.drawContours(self.srcImg,contours,contours_number, (0,255,0), line_size)
            elif(self.approx_mode=="TC89_L1"):
                img,contours,hierarchy = cv.findContours(self.img, cv.RETR_LIST, cv.CHAIN_APPROX_TC89_L1)  # 轮廓检测
                self.srcImg= cv.drawContours(self.srcImg,contours,contours_number, (0,255,0), line_size)
        elif(self.retrieval_mode=="CCOMP"):
            if(self.approx_mode=="NONE"):
                img,contours,hierarchy = cv.findContours(self.img, cv.RETR_CCOMP, cv.CHAIN_APPROX_NONE)  # 轮廓检测
                self.srcImg= cv.drawContours(self.srcImg,contours,contours_number, (0,255,0), line_size)
            elif(self.approx_mode=="SIMPLE"):
                img,contours,hierarchy = cv.findContours(self.img, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)  # 轮廓检测
                self.srcImg= cv.drawContours(self.srcImg,contours,contours_number, (0,255,0), line_size)
            elif(self.approx_mode=="TC89_L1"):
                img,contours,hierarchy = cv.findContours(self.img, cv.RETR_CCOMP, cv.CHAIN_APPROX_TC89_L1)  # 轮廓检测
                self.srcImg= cv.drawContours(self.srcImg,contours,contours_number, (0,255,0), line_size)
        elif(self.retrieval_mode=="TREE"):
            if(self.approx_mode=="NONE"):
                img,contours,hierarchy = cv.findContours(self.img, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)  # 轮廓检测
                self.srcImg= cv.drawContours(self.srcImg,contours,contours_number, (0,255,0), line_size)
            elif(self.approx_mode=="SIMPLE"):
                img,contours,hierarchy = cv.findContours(self.img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)  # 轮廓检测
                self.srcImg= cv.drawContours(self.srcImg,contours,contours_number, (0,255,0), line_size)
            elif(self.approx_mode=="TC89_L1"):
                img,contours,hierarchy = cv.findContours(self.img, cv.RETR_TREE, cv.CHAIN_APPROX_TC89_L1)  # 轮廓检测
                self.srcImg= cv.drawContours(self.srcImg,contours,contours_number, (0,255,0), line_size)
        cv.imshow("DrawContour",self.srcImg)
        ##有三个，第一个是输入图像，第二个是轮廓检索模式，第三个是轮廓近似方法。
        ##第三个参数设置为cv2.CHAIN_APPROX_NONE，所有的边界点 都会被存储；
        ##设置为cv2.CHAIN_APPROX_SIMPLE 将轮廓冗余的点去掉，压缩轮廓，节省内存开销。
        #self.img= cv.drawContours(self.img,contours,-1, (0,255,0), 3)
        #函数 cv2.drawContours() 可以被用来绘制轮廓。它可以根据你提供 的边界点绘制任何形状。它的第一个参数是原始图像，
        # 第二个参数是轮廓，一 个 Python 列表。
        # 第三个参数是轮廓的索引（在绘制独立轮廓是很有用，当设 置为 -1 时绘制所有轮廓）。接下来的参数是轮廓的颜色和厚度等。

    def GetFittingImg(self):
        # self.ReadSrcImg()
        # self.GetGrayImg()
        # self.GetCannyImg()
        contours,hierarchy = cv.findContours(self.img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        if(self.shape_fitting=="Rect"):
            for i in range(len(contours)):
                x,y,w,h = cv.boundingRect(contours[i])
                #画矩形的语句：
                cv.rectangle(self.srcImg, (x, y), (x + w, y + h), (0, 255, 0), 1)  # 画出矩'
                #area = cv2.contourArea(cnt)
                # aspect_ratio = float(w)/h  # 长宽比
                # rect_area = w*h
                # extent = float(area)/rect_area  # 轮廓面积与边界矩形面积的比。
                # hull = cv2.convexHull(cnt)
                # hull_area = cv2.contourArea(hull)
                # solidity = float(area)/hull_area  # 轮廓面积与凸包面积的比。
                # cv2.rectangle(img1, (x, y), (x+w, y+h), (0, 255, 0), 2)
        elif(self.shape_fitting=="MinRect"):
            for i in range(len(contours)):
                rect = cv.minAreaRect(contours[i])
                # minAreaRect：rect是一个由三个元素组成的元组，
                # 依次代表旋转矩形的中心坐标、尺寸和旋转角度，可以确定一个旋转矩形）
                box = cv.boxPoints(rect)
                #boxPoints：根据旋转矩形的中心的坐标、尺寸和旋转角度，计算出旋转矩形的四个顶点
                box = np.int0(box)  # 获得矩形顶点
                # area = cv.contourArea(box)
                # width = rect[1][0]
                # height = rect[1][1]
                cv.polylines(self.srcImg, [box], True, (0, 0, 255), 1)
                ##也可以用：cv.drawContours 区别？？
        elif(self.shape_fitting=="Circle"):
            for i in range(len(contours)):
                (x, y), radius = cv.minEnclosingCircle(contours[i])
                center = (int(x), int(y))
                radius = int(radius)
                cv.circle(self.srcImg, center, radius, (255,0,0), 1)
                # area = cv.contourArea(contours)
                # equi_diameter = np.sqrt(4*area/np.pi) #平均直径？
        elif(self.shape_fitting=="Ellipse"):
            for i in range(len(contours)):
                ellipse = cv.fitEllipse(contours[i])
                #或者(x, y), (a, b), angle = cv.fitEllipse(contours[i])
                cv.ellipse(self.srcImg, ellipse, (100, 255, 100), 1)
        elif(self.shape_fitting=="Line"):
            for i in range(len(contours)):
                rows, cols = self.img.shape[:2]
                [vx, vy, x, y] = cv.fitLine(contours[i], cv.DIST_L2, 0, 0.01, 0.01)
                slope = -float(vy)/float(vx)  # 直线斜率
                lefty = int((x*slope) + y)
                righty = int(((x-cols)*slope)+y)
                cv.line(self.srcImg, (cols-1, righty), (0, lefty), (50, 100, 255), 1)
        elif(self.shape_fitting=="approxPolyDP "):
            pass

        cv.imshow("FittingImg",self.srcImg)

    def GetSaveImg(self):
        cv.imwrite("Img_saved.jpg",self.img)
        cv.imwrite("SrcImg_saved.jpg",self.srcImg)

    def ChooseRation(self,Ration):
        self.Ration=Ration
    def ChooseBlurType(self,blur_type):
        self.blur_type=blur_type
    def ChooseThreshType(self,thresh_type):
        self.thresh_type=thresh_type
    def ChooseAdapThreshType(self,AdapThreshType):
        self.adap_thresh_type=AdapThreshType
    def ChooseAdapBinaryType(self,AdapBinaryType):
        self.adap_binary_type=AdapBinaryType
    def ChooseDrawContourMode1(self,retrieval_mode):
        self.retrieval_mode=retrieval_mode
    def ChooseDrawContourMode2(self,approx_mode):
        self.approx_mode=approx_mode
    def ChooseFittingShape(self,shape_fitting):
        self.shape_fitting=shape_fitting

if __name__ == '__main__':

    app = QtWidgets.QApplication(sys.argv)
    my_pyqt_form = MyPyQT_Form()
    #my_pyqt_form.GetFittingImg()
    my_pyqt_form.show()
    sys.exit(app.exec_())

# The demo program of ALOCC model    

# discription  
 Implementation of ALOCC using tensorflow.   If any bug, please send me e-mail.   
 You have to  download MNIST.npz from official site.   
 Now, modifing computation of AUC value.
 https://github.com/houssamzenati/Efficient-GAN-Anomaly-Detection/tree/master/data  
 
# official implementation  
official implementation is here
https://github.com/khalooei/ALOCC-CVPR2018

# literature  
 [Adversarially Learned One-Class Classifier for Novelty Detection](https://arxiv.org/abs/1802.09088)  

# dependency  
I confirmed operation only with..   
1)python 3.6.3  
2)tensorflow 1.7.0  
3)numpy 1.14.2    
4)Pillow 4.3.0  

# result Image  
After learning about 200epochs using MNIST  digit "1", I input "1" and other digits to R-Network.  
![resultimage_18110905_235](https://user-images.githubusercontent.com/15444879/48254515-9fd5be00-e44d-11e8-86f6-8ea0976b0682.png)
Right side is prediction for "1", and left side is prediction for other digits.  From left colomn, input images, reconstructioned images, differences.  

# email  
t.ohmasa@w-farmer.com  

# Dynamic System Identificastion using RNN
The following dynamic system is run in the Simulink file:  
<a href="https://www.codecogs.com/eqnedit.php?latex=x_1(k&plus;1)&space;=&space;0.5&space;*&space;(\frac{x_1}{(1&plus;x_2^2)}&space;&plus;&space;u_1)" target="_blank"><img src="https://latex.codecogs.com/svg.latex?x_1(k&plus;1)&space;=&space;0.5&space;*&space;(\frac{x_1}{(1&plus;x_2^2)}&space;&plus;&space;u_1)" title="x_1(k+1) = 0.5 * (\frac{x_1}{(1+x_2^2)} + u_1)" /></a>  
<a href="https://www.codecogs.com/eqnedit.php?latex=x_2(k&plus;1)&space;=&space;0.5&space;*&space;(\frac{x_1*x_2}{(1&plus;x_2^2)}&space;&plus;&space;u_2)" target="_blank"><img src="https://latex.codecogs.com/svg.latex?x_2(k&plus;1)&space;=&space;0.5&space;*&space;(\frac{x_1*x_2}{(1&plus;x_2^2)}&space;&plus;&space;u_2)" title="x_2(k+1) = 0.5 * (\frac{x_1*x_2}{(1+x_2^2)} + u_2)" /></a>  
<a href="https://www.codecogs.com/eqnedit.php?latex=y&space;=&space;x_1&space;&plus;&space;x_2" target="_blank"><img src="https://latex.codecogs.com/svg.latex?y&space;=&space;x_1&space;&plus;&space;x_2" title="y = x_1 + x_2" /></a>  
The data is generated by Matlab and saved as csv file.  
The identification is done by three layer of LSTM. After the training, the performance on the training set itself is as shown below:  
![alt text](https://github.com/marryabd/Dynamis-system-identification/blob/master/Images/Figure_1.png)
Moreover, on the providing test set, the performance is also good. Note that, the test set data frequency is higher than the training and the network can estimate the output wiht high accuracy. However, the network does not perform very well on generalizing for larger input domain. To improve the performance, larger training set is required.  
![alt text](https://github.com/marryabd/Dynamis-system-identification/blob/master/Images/Figure_2.png)

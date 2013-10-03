import Image
import numpy as np
test = Image.open('/home/matthew/Dropbox/Work/C/Cuda/cs344/HW2/cinque_terre.gold')
output = Image.open('/home/matthew/Dropbox/Work/C/Cuda/blur/cinque_terre_small.jpg')
tp = test.load()
op = ouput.load()

compare = np.ones(o.size)==1
for i in range(0,o.size[0]):
	for j in range(0,o.size[1]):
		r = abs(op[i,j][0]-tp[i,j][0])
		g = abs(op[i,j][1]-tp[i,j][1])
		b = abs(op[i,j][2]-tp[i,j][2])
		if r>5 or g>5 or b>5:
			compare[i,j] = False

r=1
x= range(0,o.size[1])
fig = plt.figure()
ax0 = fig.add_subplot(311)
ax1 = fig.add_subplot(312)
ax2 = fig.add_subplot(313)
for i in x:
	ax0.plot(i,op[r,i][0],'ro' , label='mine')	
	ax0.plot(i,tp[r,i][0],'bo', label='test')
	ax1.plot(i,op[r,i][1],'ro' , label='mine')	
	ax1.plot(i,tp[r,i][1],'bo', label='test')
	ax2.plot(i,op[r,i][2],'ro' , label='mine')	
	ax2.plot(i,tp[r,i][2],'bo', label='test') 	



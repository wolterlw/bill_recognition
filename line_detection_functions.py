import numpy as np
import cv2
import matplotlib.pyplot as plt

def binarize(img, mask=None, plot=False):
	inv = 255 - img
	inv[img==0] = 0
	eroded = cv2.erode(inv,np.ones((5,5)),iterations=5)
	# eroded = cv2.GaussianBlur(eroded,(25,5),5)
	tuned = eroded+img

	# tuned[eroded==0] = 255

	tuned = 255-tuned
	tuned[eroded==0] = 255

	tmp = cv2.medianBlur(
		cv2.erode(tuned,np.ones((1,5)))
		,5)

	res_img = (tuned.astype('int16') - tmp)

	if mask is None:
		res_img[(res_img<10)|(eroded==0)] = 0
	else:
		res_img[(res_img<10)|(mask==0)] = 0

	if plot:
		f,ax = plt.subplots(2,2,figsize=(15,10))
		ax[0,0].imshow(inv[100:300,:], cmap='Greys_r')
		ax[0,0].set_title('inverted')
		ax[0,1].imshow(eroded[100:300,:], cmap='Greys_r')
		ax[0,1].set_title('eroded')
		ax[1,0].imshow(tmp[100:300,:], cmap='Greys_r')
		ax[1,0].set_title('median blur')
		ax[1,1].imshow(res_img[100:300,:], cmap='Greys_r')
		ax[1,1].set_title('resulting')
		plt.show()
	return res_img

def get_lines(bin_img, return_coord = False):
	thresh = (bin_img>30).astype('uint8')
	dil = cv2.dilate(thresh,np.ones((1,5)),iterations=5)
	erod = cv2.erode(dil,np.ones((3,1)),iterations=2)
	
	histo = erod.sum(axis=1)*0.5 + bin_img.sum(axis=1)*0.5
	y = lambda x: histo[x]

	population = np.random.randint(0,len(histo),len(histo))

	pop_size = len(population)-1
	for delta in range(7,0,-1):
		for i in range(100):
			population = set([min([max(0,p-delta),p,min(p+delta, pop_size)],key=y) for p in population])

	population = sorted(list(population))

	for i,j in zip(population[:-1],population[1:]):
		if (j-i) < 10:
			if i in population and j in population:
				population.remove(max([i,j],key=y))
	
	if return_coord:
		return population
	else:
		beg_end = zip(population[:-1],population[1:])
		lines = [bin_img[b:e,:] for b,e in beg_end]
		
		return lines

def shorten_line(line,window_size=8):
	pad_left = window_size // 2
	pad_right = window_size // 2 + window_size % 2
	mf = line[5:15,:].sum(axis=0)
	eroded = np.pad(
		[mf[i:i+8].max() for i in range(mf.shape[0]-8)],
		(pad_left,pad_right),'constant'
	)
	return line[:,np.argwhere(eroded>0).reshape(-1)]

def preprocess_text(img):
	assert len(img.shape) == 2, "convert to grayscale first"
	assert img.shape[1] == 300, "images should be size (-1,300)"
	lines = [shorten_line(l) for l in get_lines(binarize(img))]
	return [l for l in lines if (l.shape[0]*l.shape[1])>50]
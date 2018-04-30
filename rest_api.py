# USAGE
# Start the server:
# 	python run_keras_server.py
# Submit a request via cURL:
# 	curl -X POST -F image=@dog.jpg 'http://localhost:5000/predict'
# Submita a request via Python:
#	python simple_request.py


from keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import flask
import io

import cv2

import matplotlib.pyplot as plt

from image_segmentation_functions import Segmenter
from line_detection_functions import binarize, get_lines

# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
segmenter = None

def load_model():
	# load the pre-trained Keras model (here we are using a model
	# pre-trained on ImageNet and provided by Keras, but you can
	# substitute in your own networks just as easily)
	global segmenter
	segmenter = Segmenter()

def fig2data (fig):
    fig.canvas.draw ( )
 
    w,h = fig.canvas.get_width_height()
    buf = np.frombuffer( fig.canvas.tostring_rgb(), dtype=np.uint8 )
    # fig.close()
    buf = buf.reshape((h,w,3))
    #crop the whitespace
    bnd1 = np.argwhere( buf[:,:,2].min(axis=0)!=255 ).ravel()
    bnd0 = np.argwhere( buf[:,:,2].min(axis=1)!=255 ).ravel()

    return buf[bnd0[0]:bnd0[-1],
               bnd1[0]:bnd1[-1]]

def get_res_image(img,lines):
	f = plt.figure()
	plot = f.add_subplot(111)
	plot.axis('off')

	width = img.shape[1]
	plot.set_xlim(0,width)
	plot.imshow(img,
	           cmap='Greys_r')
	plot.hlines(lines,
	           0, width,
	           colors='r',
	           linestyles='--',
	           linewidth=1)

	res_img = fig2data(f)
	plt.clf()
	return res_img

def prepare_image(image):
	# if the image mode is not RGB, convert it
	if image.mode != "RGB":
		image = image.convert("RGB")

	# resize the input image and preprocess it
	image = image.resize((1280, 960))
	image = img_to_array(image)

	# return the processed image
	return image

def store_results(image,lines):
	pass

@app.route("/predict", methods=["POST"])
def predict():
	# initialize the data dictionary that will be returned from the
	# view
	data = {"success": False}

	# ensure an image was properly uploaded to our endpoint
	if flask.request.method == "POST":
		if flask.request.files.get("image"):
			# read the image in PIL format
			image = flask.request.files["image"].read()
			image = Image.open(io.BytesIO(image))

			# preprocess the image and prepare it for classification
			image = prepare_image(image)

			
			img_proc, mask = segmenter.process_image(image)
			bin_ = binarize(img_proc.astype('uint8'),
                plot=False,
                mask=cv2.erode(mask,
                               np.ones((5, 5)),
                               iterations=7
                )
			)
			bin_ = (bin_>30).astype('uint8')*bin_
			lines = get_lines(bin_, return_coord=True)
		
			store_results(bin_, lines)

			res_img = get_res_image(img_proc, lines)
			pil_img = Image.fromarray(res_img)
			byte_io = io.BytesIO()
			pil_img.save(byte_io, 'PNG')
			byte_io.seek(0)

			print("before sending")
			return flask.send_file(
					byte_io,
					mimetype='image/png'
				)

# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
	print(("* Loading Keras model and Flask starting server..."
		"please wait until server has fully started"))
	load_model()

app.run()
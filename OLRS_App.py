#############################################################################
# Campus-tour Guide: On-campus Landmarks Recognition System (OLRS)          #
#                                                                           #
#                                                                           #
# This is the **Application program** for the OLRS                          #
#                                                                           #
# The app will run on localhost (http://127.0.0.1:5000),                    #
#   please use the browser to access the web page                           #
#                                                                           #
#                                                                           #
#############################################################################

import torch
from flask import Flask, render_template, request
import os
import cv2
import numpy
from OLRS_Algo import CNN, predictCNN # import trained CNN model, and prediction algorithm function
import time

# define label dictionary label:["name","url"]
label_dict = {1:["Communal Building","https://www.google.com/maps/embed?pb=!1m14!1m8!1m3!1d1237.6886370426041!2d114.17982722105018!3d22.304389930481882!3m2!1i1024!2i768!4f13.1!3m3!1m2!1s0x340400e7d3ff3a4f%3A0xc92e4e49849b0894!2sCommunal%20Building%2C%20Hong%20Kong%20Polytechnic%20University!5e1!3m2!1sen!2shk!4v1679848365505!5m2!1sen!2shk"],
              2:["LibCafe","https://www.google.com/maps/embed?pb=!1m14!1m8!1m3!1d1237.697192179583!2d114.17983862569567!3d22.30342447795825!3m2!1i1024!2i768!4f13.1!3m3!1m2!1s0x340400e61ff7c00d%3A0x1da7fb3d040019f2!2sLib%20Cafe%20%40%20PolyU!5e1!3m2!1sen!2shk!4v1679848430576!5m2!1sen!2shk"],
              3:["Li Ka Shing Tower","https://www.google.com/maps/embed?pb=!1m14!1m8!1m3!1d1237.697192179583!2d114.17983862569567!3d22.30342447795825!3m2!1i1024!2i768!4f13.1!3m3!1m2!1s0x340400e6284300db%3A0x4d6e2110ced25d58!2sThe%20Hong%20Kong%20Polytechnic%20University%20Li%20Ka%20Shing%20Tower!5e1!3m2!1sen!2shk!4v1679848468647!5m2!1sen!2shk"],
              4:["Jockey Club Innovation Tower","https://www.google.com/maps/embed?pb=!1m14!1m8!1m3!1d1237.6824659426766!2d114.1795535095585!3d22.305086318323447!3m2!1i1024!2i768!4f13.1!3m3!1m2!1s0x340400e87bc70a43%3A0x41b4db8708c4f997!2sJockey%20Club%20Innovation%20Tower!5e1!3m2!1sen!2shk!4v1679848506700!5m2!1sen!2shk"],
              5:["Lee Shau Kee Building","https://www.google.com/maps/embed?pb=!1m14!1m8!1m3!1d601.9170086063797!2d114.18017448434357!3d22.30554904280531!3m2!1i1024!2i768!4f13.1!3m3!1m2!1s0x340400e7ea4cd421%3A0xc9fb261caeaaf54a!2sThe%20Hong%20Kong%20Polytechnic%20University%20Lee%20Shau%20Kee%20Building!5e1!3m2!1sen!2shk!4v1679848552038!5m2!1sen!2shk"],
              6:["Block VA","https://www.google.com/maps/embed?pb=!1m18!1m12!1m3!1d291.0864783915491!2d114.1793771125983!3d22.30445422411509!2m3!1f0!2f0!3f0!3m2!1i1024!2i768!4f13.1!3m3!1m2!1s0x340400e8bcc55287%3A0x9d68bde580ce1e36!2zU3RhZmYgQ2FudGVlbi9UZWEgSG91c2Ug55CG5aSn6Iy25a6k!5e1!3m2!1sen!2shk!4v1679848727388!5m2!1sen!2shk"],
              7:["Logo Square","https://www.google.com/maps/embed?pb=!1m18!1m12!1m3!1d291.08675378558684!2d114.17960711202136!3d22.30432208275497!2m3!1f0!2f0!3f0!3m2!1i1024!2i768!4f13.1!3m3!1m2!1s0x340400e62b17d551%3A0xb976ecf06412e438!2sChan%20Sui%20Kau%20and%20Chan%20Lam%20Moon%20Chun%20Square!5e1!3m2!1sen!2shk!4v1679848753832!5m2!1sen!2shk"],
              8:["Jockey Club Auditorium","https://www.google.com/maps/embed?pb=!1m18!1m12!1m3!1d291.085525492384!2d114.1803608127545!3d22.304911444664498!2m3!1f0!2f0!3f0!3m2!1i1024!2i768!4f13.1!3m3!1m2!1s0x340400e7cdc1e295%3A0xc663d4ec42aeb6cd!2sJockey%20Club%20Auditorium%2C%20Hong%20Kong%20Polytechnic%20University!5e1!3m2!1sen!2shk!4v1679848779612!5m2!1sen!2shk"],
              9:["Global Student Hub","https://www.google.com/maps/embed?pb=!1m14!1m12!1m3!1d291.08575193486007!2d114.1808327942858!3d22.30480279372899!2m3!1f0!2f0!3f0!3m2!1i1024!2i768!4f13.1!5e1!3m2!1sen!2shk!4v1679848882966!5m2!1sen!2shk"],
              10:["Industrial Centre","https://www.google.com/maps/embed?pb=!1m18!1m12!1m3!1d424.60401392051415!2d114.18059927862325!3d22.30545340561605!2m3!1f0!2f0!3f0!3m2!1i1024!2i768!4f13.1!3m3!1m2!1s0x340400e7c1849f1d%3A0x7de2449ade29d7a9!2sIndustrial%20Centre%20(PolyU)!5e1!3m2!1sen!2shk!4v1679848921643!5m2!1sen!2shk"],
              11:["Pao Yue-Kong Library","https://www.google.com/maps/embed?pb=!1m18!1m12!1m3!1d1235.8894191854044!2d114.18009791520088!3d22.30352922812425!2m3!1f0!2f0!3f0!3m2!1i1024!2i768!4f13.1!3m3!1m2!1s0x340400e621b21f81%3A0xf16a483a46632bc9!2sPao%20Yue-kong%20Library%2C%20The%20Hong%20Kong%20Polytechnic%20University!5e1!3m2!1sen!2shk!4v1679848941918!5m2!1sen!2shk"],
              12:["Main Entrance","https://www.google.com/maps/embed?pb=!1m18!1m12!1m3!1d412.03602137815585!2d114.17820972880558!3d22.302981175970682!2m3!1f0!2f0!3f0!3m2!1i1024!2i768!4f13.1!3m3!1m2!1s0x340400e8bc926631%3A0xb47b2e56fa16092a!2sTang%20Ping%20Yuan%20Square!5e1!3m2!1sen!2shk!4v1679849062286!5m2!1sen!2shk"],
              13:["Block X Sport Center","https://www.google.com/maps/embed?pb=!1m18!1m12!1m3!1d291.0843851459422!2d114.17997109289337!3d22.305458594568496!2m3!1f0!2f0!3f0!3m2!1i1024!2i768!4f13.1!3m3!1m2!1s0x340400e7d90b041b%3A0xc4bd27fddfd13f9b!2sThe%20Hong%20Kong%20Polytechnic%20University%20Block%20X%20Sports%20Centre!5e1!3m2!1sen!2shk!4v1679849086749!5m2!1sen!2shk"],
              14:["Lawn","https://www.google.com/maps/embed?pb=!1m18!1m12!1m3!1d360.9350028962638!2d114.1800261208732!3d22.304084433479446!2m3!1f0!2f0!3f0!3m2!1i1024!2i768!4f13.1!3m3!1m2!1s0x340400e62eb8b7c5%3A0x8abe108666d9cad5!2sLawn%20Caf%C3%A9!5e1!3m2!1sen!2shk!4v1679849144758!5m2!1sen!2shk"]}

UPLOAD_FOLDER = os.path.join('static', 'uploads')
app = Flask(__name__, template_folder='template', static_folder='static')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'ZHANG WENGYU'

# Load the CNN model
model = torch.load('modelCNN.pth') 
 
# initialize the web page
@app.route('/')
def index():
    return render_template('index.html', user_image = "static/polyu.jpg",img_label = "PolyU", img_map_url = "https://www.google.com/maps/embed?pb=!1m14!1m8!1m3!1d2191.716936626927!2d114.1795058!3d22.3040093!3m2!1i1024!2i768!4f13.1!3m3!1m2!1s0x340400e809c71ff1%3A0xeb7151a34a54910d!2sThe%20Hong%20Kong%20Polytechnic%20University%20(PolyU)!5e1!3m2!1sen!2shk!4v1679874374388!5m2!1sen!2shk")

# response to the POST request
@app.route('/',  methods=("POST", "GET"))
def uploadFile():
    if request.method == 'POST':
        
        # Clearing the folder, no uploaded file will be stored in the server once the prediction is done
        dir = 'static/uploads'
        for f in os.listdir(UPLOAD_FOLDER):
            os.remove(os.path.join(dir, f))

        # get the uploaded image
        uploaded_img = request.files['uploaded-file']

        # use the current time as the file name for temporary storage
        ts = str(int(time.time()))
        img_filename = ts +'.jpeg'
        
        # get file from the request
        image = numpy.fromstring(uploaded_img.read(), numpy.uint8)
        img = cv2.imdecode(image, cv2.IMREAD_UNCHANGED)

        # predict the image with the CNN prediction algorithm function
        pred = predictCNN(img,model)
        label = label_dict[pred][0] # get the label
        label_url = label_dict[pred][1] # get the url for the map
        img = cv2.resize(img, (1024, 1024), interpolation=cv2.INTER_CUBIC)
        print(label)
        
        # temporary storage for display
        fname = 'static/uploads/'+img_filename
        cv2.imwrite(fname, img)
        
        # Display image in web page
        result = render_template('index.html', user_image = fname, img_label = label, img_map_url = label_url)
        return result
  
if __name__=='__main__':
    # run the app on localhost (http://127.0.0.1:5000), use the browser to access the web page
    app.run(debug=True)
    # app.run(host='0.0.0.0', port=8500, debug=True)

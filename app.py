from kivy.app import App
from kivy.lang import Builder
from kivy.core.window import Window
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.properties import  StringProperty
from kivy.config import Config
from tkinter import *                   
from tkinter import filedialog as fd    
import numpy as np
import tensorflow as tf
import keras
import keras.utils as image
import keras.backend as K


def recall(y_true, y_pred):
    tp = K.sum(K.cast(y_true * y_pred, float))
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'))
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'))
    recall_keras = tp/ (tp+ fn + K.epsilon())
    return recall_keras
def precision(y_true, y_pred):
    tp = K.sum(K.cast(y_true * y_pred, float))
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'))
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'))
    precision_keras = tp / (tp + fp + K.epsilon())
    return precision_keras
def f1(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * ((p * r) / (p + r + K.epsilon()))

def prediction(model, path):

    predict_img = image.load_img(path, target_size = (224,224))
    predict_img = image.img_to_array(predict_img)
    predict_img = np.expand_dims(predict_img, axis=0)
    with tf.device('/device:CPU:0'):
        result = model.predict(predict_img)
    if result[0][0] == 1:
        prediction = 'No Crack Detected'
    else:
        prediction = 'Crack Detected'
    return prediction

Config.set('graphics', 'resizable', False)

Window.size = (260, 600)
Window.clearcolor = (0,0,0,1)

class Homepage(Screen):
    pass

class Metricspage(Screen):
    pass

class Detectionpage(Screen):
    text = StringProperty()
    path = StringProperty("/home/ayush/Documents/Machine_learning/L&T EduTech Hackathon at SHAASTRA IITM/crack/test/Positive/DJI_0677_10_16.jpg")

    def run_model(self):

        #open file dialog
        root = Tk()
        root.withdraw()
        root.filename = fd.askopenfilename(initialdir = "/",title = "Select file",filetypes = (("jpeg files","*.jpg"),("all files","*.*")))
        self.path = root.filename
        root.destroy()
        model = keras.models.load_model('mobilenet_crack_detection.h5',custom_objects={'precision': precision, 'recall': recall, 'f1': f1})
        
        pred = prediction(model,self.path)
        if pred == "Crack Detected":
            self.text = "Crack Detected"
        else:
            #have no on one line and crack on other line
            self.text = "\n"+ "             No"+"\n"+"Crack Detected"



        
        
class WindowManager(ScreenManager):
    pass
    

kv = Builder.load_file('components.kv')
sm = WindowManager()

 

class loginMain(App):
    def build(self):
        sm.add_widget(Homepage(name='home'))
        sm.add_widget(Detectionpage(name="detect"))
        sm.add_widget(Metricspage(name="metrics"))
        return sm

if __name__ == '__main__':
    loginMain().run()
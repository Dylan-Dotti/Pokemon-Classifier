import keras_deep_learning.projects.pokemon_classifier as pc
import keras_deep_learning.projects.pokemon_classifier.dataset as ds
import numpy as np
import tkinter as tk
import tkinter.filedialog as fd
from keras import models
from keras.preprocessing.image import load_img, img_to_array
from keras_deep_learning.projects.pokemon_classifier.dataset import get_categories
from PIL import Image, ImageTk


class PokemonClassifierFrame(tk.Frame):

    def __init__(self, master):
        super().__init__(master)
        model_path = pc.__file__.replace('__init__.py',
            'models\\full_classifier_pretrained.h5')
        self.model = models.load_model(model_path)

        categories = ds.get_categories()
        self.index_class_map = {i: c 
            for (i, c) in enumerate(categories)}

        self.img_path = None
        self.img_label = tk.Label(master, text='No Image Selected')
        self.select_file_button = tk.Button(self.master, 
            text='Select Image', command=self._on_select_file_pressed)
        self.classify_button = tk.Button(self.master,
            text='Who\'s that Pokemon?', command=self._on_classify_pressed)
        self.result_label = tk.Label(master)

        self.img_label.grid(row=0, column=0, padx=20, pady=5)
        self.select_file_button.grid(row=1, column=0, padx=20, pady=5)
        self.classify_button.grid(row=2, column=0, padx=20, pady=5)
    
    def get_top_results(self, results, count=3):
        results = results.copy()
        results.sort(reverse=True)
        print(results)
    
    def _on_select_file_pressed(self):
        self.img_path = fd.askopenfile(mode='r').name
        print('Selected:', self.img_path)
        img = Image.open(self.img_path).resize((250, 250), Image.ANTIALIAS)
        img = ImageTk.PhotoImage(img)
        self.result_label.grid_forget()
        self.img_label.grid_forget()
        self.img_label = tk.Label(self.master, image=img)
        self.img_label.img = img
        self.img_label.grid(row=0, column=0, padx=10, pady=10)
    
    def _on_classify_pressed(self):
        print(self.img_path)
        img = load_img(self.img_path, target_size=(150, 150))
        img_arr = img_to_array(img)
        img_tensor = np.array([img_arr])
        results = self.model.predict(img_tensor)[0].tolist()
        print(self.model.predict(img_tensor))
        res_index = results.index(max(results))
        self.result_label.grid_forget()
        self.result_label = tk.Label(self.master,
            text=self.index_class_map[res_index])
        self.result_label.grid(row=3, column=0, padx=10, pady=10)



window = tk.Tk()
window.title('Pokemon Classifier')

PokemonClassifierFrame(window).grid(row=0, column=0)

window.mainloop()
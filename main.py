import matplotlib.pyplot as pt
import numpy as np
import pandas as pd
from matplotlib.widgets import Slider

house_data=pd.read_csv("./data/data_for_lr.csv")

weights=np.random.uniform(0,1);
biases=np.random.uniform(0,1);


fig ,ax=pt.subplots()
fig.subplots_adjust(left=0.25,bottom=0.25)
axes=pt.axes([0.25,0.01,0.65,0.2],facecolor="red")
slide=Slider(axes,label="Weights",valmin=0,valmax=5,valinit=weights)
class Linear_Regerssion():
    def __init__(self,house_data,weights,biases):
        self.w=weights
        self.b=biases
        self.x=house_data.x
        self.y=house_data.y
    def predict(self):
        self.ys=self.w*self.x +self.b
        return self.ys
    def plot(self):
        ax.scatter(self.x,self.y,c="b",s=5,)
        self.line ,=ax.plot(self.x,self.ys,c="r")
        ax.set_xlabel("square feet")
        ax.set_ylabel("cost")
    def cost(self):
        square_difference=(np.array(self.y)-np.array(self.ys))**2
        cost_value=(np.sum(square_difference))/(2*len(self.ys))
        return cost_value
    
        

def Update_Weights_Manually(val):
    test.w=slide.val
    ys=test.predict()
    test.line.set_ydata(ys)
    
    print(test.cost())
    
    fig.canvas.draw_idle()
    



test=Linear_Regerssion(house_data,weights,biases)
test.predict()
test.plot()
ax.legend()
slide.on_changed(Update_Weights_Manually)
pt.show()
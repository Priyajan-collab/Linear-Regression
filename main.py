import matplotlib.pyplot as pt
import numpy as np
import pandas as pd
from matplotlib.widgets import Slider,Button

house_data=pd.read_csv("./data/data_for_lr.csv")

weights=np.random.uniform(0,1);
biases=np.random.uniform(0,1);
lr=0.00001


fig ,ax=pt.subplots()
fig.subplots_adjust(left=0.25,bottom=0.45)
axes=pt.axes([0.25,0.01,0.65,0.2],facecolor="red")
axes_button=pt.axes([0.45,0.2,0.1,0.1],facecolor="blue",)
btn=Button(axes_button,label="train")
slide=Slider(axes,label="Weights",valmin=0,valmax=5,valinit=weights)
class Linear_Regeression():
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
        self.line ,=ax.plot(self.x,self.ys,c="r",label="prediction line")
        ax.set_xlabel("square feet")
        ax.set_ylabel("cost")
       
        ax.legend()
    def cost(self):
        square_difference=(np.array(self.y)-np.array(self.ys))**2
        cost_value=(np.sum(square_difference))/(2*len(self.ys))
        print(cost_value)
        return cost_value
    def update_Weight_biases(self):
                derivative_cost=np.sum((np.array(self.y)-np.array(self.ys))*self.x)/len(self.ys)
                derivative_biases=np.sum(np.array(self.y)-np.array(self.ys))/len(self.ys)
                self.w= self.w +lr*derivative_cost
                self.b= self.b +lr*derivative_biases
                
               
    
                

def Update_Weights_Manually(val):
    test.w=slide.val
    ys=test.predict()
    test.line.set_ydata(ys)
    
    print(test.cost())
    
    fig.canvas.draw_idle()
    



test=Linear_Regeression(house_data,weights,biases)
test.predict()

test.plot()
def train(val):

        for i in range(100):
                test.predict()
                test.cost()
                test.update_Weight_biases()
                test.line.set_ydata(test.ys)
                fig.canvas.draw_idle()
                pt.pause(0.1)
ax.legend()
btn.on_clicked(train)
slide.on_changed(Update_Weights_Manually)
pt.show()
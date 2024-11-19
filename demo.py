import numpy as np
import pandas as pd
import matplotlib.pyplot as pt
from matplotlib.widgets import Slider


src="data/data_for_lr.csv"

house_data=pd.read_csv(src)
weights=np.random.uniform(0,1);
biases=np.random.uniform(0,1);
lr=0.0001
weights=2;
biases=1;
def predict(xs,weights):
        return (weights*xs+biases)

# ys=predict(house_data.x)
# def plot_line(ys):
#         ax.lines.clear()  
#         ax.plot(house_data.x, ys, color="red")
#         pt.pause(0.1)  
    
# def cost(y,ys):
#         errors=(np.array(y)-np.array(ys))**2
#         update_error=np.sum(errors)/(2*len(ys))
#         return update_error
   


fig ,ax=pt.subplots()
fig.subplots_adjust(left=0.25,bottom=0.25)
ax=fig.gca()

ax.set_xlim([0,100])
ax.set_ylim([0,100])
ax.scatter(house_data.x,house_data.y,s=5)
ys=predict(house_data.x,weights)
line,=ax.plot(house_data.x,ys,c="r")
# axes=ax.axes()
# plot_line(house_data.x)

# ax.scatter(house_data.x,house_data.y,s=5)
freq_axes=pt.axes([0.25,0.01,0.65,0.2],facecolor="red")
slider=Slider(freq_axes,"weights",valmin=0.1,valmax=3,valinit=0.1)
def update(val):
        weights=slider.val
        ys=predict(house_data.x,weights)
        line.set_ydata(ys)
        fig.canvas.draw_idle
        
slider.on_changed(update) 


pt.show()
import numpy as np
import pandas as pd
import matplotlib.pyplot as pt


src="data/data_for_lr.csv"

house_data=pd.read_csv(src)
weights=np.random.uniform(0,1);
biases=np.random.uniform(0,1);
lr=0.0001
def predict(xs):
        return (weights*xs+biases)

# ys=predict(house_data.x)
def plot_line(ys):
        ax.lines.clear()  
        ax.plot(house_data.x, ys, color="red")
        pt.pause(0.1)  
    
def cost(y,ys):
        errors=(np.array(y)-np.array(ys))**2
        update_error=np.sum(errors)/(2*len(ys))
        return update_error

def update_weight(y,ys):
        global weights
        updated_weights=np.sum(np.array(house_data.x)*(np.array(y)-np.array(ys)))/len(ys)
        weights+=lr*updated_weights
        
def update_bias(y,ys):
        global biases
        updated_biases=np.sum(np.array(y)-np.array(ys))/len(ys)
        biases+=lr*updated_biases       


fig =pt.figure()
ax=fig.gca()
ax.set_xlim([0,100])
ax.set_ylim([0,100])
ax.scatter(house_data.x,house_data.y,s=5)

# plot_line(house_data.x)

# ax.scatter(house_data.x,house_data.y,s=5)

for i in range(200):
        print("iteration",i);
        ys=predict(house_data.x)
        print(cost(house_data.y,ys))
        update_bias(house_data.y,ys)
        update_weight(house_data.y,ys)
   
        plot_line(ys)
        
        


pt.show()
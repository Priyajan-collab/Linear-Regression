import numpy as np
import pandas as pd
import matplotlib.pyplot as pt


src="data/data_for_lr.csv"

house_data=pd.read_csv(src)

w=np.random.uniform(0,1);
b=np.random.uniform(0,1);
def predict(xs):
        return (w*xs+b)

ys=predict(house_data.x)
def plot_line(xs):
        ax.plot(xs,ys,c="red")
    
def cost(y,ys):
        errors=(np.array(y)-np.array(ys))**2
       
        update_error=np.sum(errors)/(2*len(ys))
        return update_error

fig =pt.figure()

print(cost(house_data.y,ys))
ax=fig.gca()
ax.set_xlim([0,100])
ax.set_ylim([0,100])

plot_line(house_data.x)

ax.scatter(house_data.x,house_data.y,s=5)







pt.show()
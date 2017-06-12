import numpy as np
import  matplotlib.pyplot as plt
import sys
import os
plot_dir = sys.argv[1]
with open(plot_dir + '/mbox_loss.txt') as rd:
  reader = rd.read()
reader = reader.split('\n')
reader = reader[0: len(reader) - 2]
loss = []
for i in range(len(reader)):
  line = reader[i]
  data = line.split('(')[0]
  data = data.split('=')[1]
  loss.append(float(data))
with open(plot_dir + '/detections_eval.txt') as rd:
  reader = rd.read()
reader = reader.split('\n')
reader = reader[0: len(reader) - 2]
detect_eval = []
for i in range (len(reader)):
  line = reader[i]
  data = line.split('=')[1]
  detect_eval.append(float(data))
x_loss = range(len(loss))
x_eval = range(len(detect_eval))

fig, ax = plt.subplots()
fig.subplots_adjust(bottom=0.2, right=0.85)

newax = fig.add_axes(ax.get_position())
newax.patch.set_visible(False)

newax.yaxis.set_label_position('right')
newax.yaxis.set_ticks_position('right')

newax.spines['bottom'].set_position(('outward', 35))

ax.plot(x_loss, loss, 'r-')
ax.set_xlabel('iteration*10', color='red')
ax.set_ylabel('Loss', color='red')

x = np.linspace(0, 6*np.pi)
newax.plot(x_eval, detect_eval , 'g-')

newax.set_xlabel('iteration*200', color='green')
newax.set_ylabel('Detection_eval', color='green')
plt.show()

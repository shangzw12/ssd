import os
import xml.etree.ElementTree as ET
prefix = '/home/shangzw/data/tbs/tbs_300x300_HSIL_10x/Annotations/'
with open('./tmp_file') as rd:
  file_name=rd.read()
file_name = file_name.split('\n')
file_name = file_name[0: len(file_name)-2]
file_name_with_pre = [prefix+ i + '.xml' for i in file_name]
width = []
height = []
for file in file_name_with_pre:
  tree= ET.parse(file)
  root = tree.getroot()
  for child in root:
    if child.tag == 'size':
      width.append(child[0].text)
      height.append(child[1].text)

for i in range(len ( file_name) ) :
  print file_name[i]+' '+width[i]+' '+height[i]

      

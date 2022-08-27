# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 00:02:08 2020

@author: Matthew
"""

import os
import numpy as np
import cv2
from matplotlib import pyplot as pt
import pytesseract as tess
tess.pytesseract.tesseract_cmd = r'C:\Users\Matthew\AppData\Local\Tesseract-OCR\tesseract.exe'
import openpyxl
from openpyxl import Workbook
import math

img = cv2.imread("05.png",0)

sE_square1 = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
sE_square2 = cv2.getStructuringElement(cv2.MORPH_RECT,(20,20))
sE_square3 = cv2.getStructuringElement(cv2.MORPH_RECT,(50,50))
sE_plus = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
sE_plus2 = cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5))
sE_diamond = np.array([
  [0,0,1,1,1,0,0],
  [0,1,1,1,1,1,0],
  [1,1,1,1,1,1,1],
  [1,1,1,1,1,1,1],
  [1,1,1,1,1,1,1],
  [0,1,1,1,1,1,0],
  [0,0,1,1,1,0,0]], dtype=np.uint8)
sE_diamond2 = np.array([
  [0,0,0,0,1,0,0,0,0],
  [0,0,0,1,1,1,0,0,0],
  [0,0,1,1,1,1,1,0,0],
  [0,1,1,1,1,1,1,1,0],
  [1,1,1,1,1,1,1,1,1],
  [0,1,1,1,1,1,1,1,0],
  [0,0,1,1,1,1,1,0,0],
  [0,0,0,1,1,1,0,0,0],
  [0,0,0,0,1,0,0,0,0]], dtype=np.uint8)
sE_x = np.array([
  [1,0,1],
  [0,1,0],
  [1,0,1]], dtype=np.uint8)

headers = ["TITLE:",
           "DRAWING NO:",
           "DRAWING NUMBER:",
           "CONTRACTOR:",
           "DRAWN BY:",
           "CHECKED BY:",
           "APPROVED BY:",
           "CAD NO:",
           "LANG:",
           "PROJECT NO:",
           "PAGE:",
           "STATUS:",
           "STS:",
           "UNIT:",
           "DRAWING TITLE:",
           "DRAWN:",
           "CHECKED:",
           "APPROVED:",
           "COMPANY NAME:",
           "COMPANY:"]

headers_amend = ["ISSUE",
                 "DATE",
                 "CHANGE(S)",
                 "BY",
                 "CDK",
                 "REV",
                 "DATE",
                 "BY"]

headers_amend_2 = ["REV",
                   "DATE",
                   "BY"]

def calculateDistance(coord_arr,a):
    dist_arr = []
    midpoint_arr = []
    
    for x in range(0,(len(coord_arr)),1):
        midpoint = (int((coord_arr[x][0][0] + coord_arr[x][3][0])/2),int((coord_arr[x][0][1] + coord_arr[x][3][1])/2))
        midpoint_arr.append(midpoint)
    
    for y in range(0,len(midpoint_arr),1):      
        distance = math.sqrt((((midpoint_arr[a][0] - midpoint_arr[y][0]) ** 2) + ((midpoint_arr[a][1] - midpoint_arr[y][1]) ** 2)))
        distance = int(distance)
        
        if distance == 0:
            distance += 1111
                    
        dist_arr.append(distance)
    
    return dist_arr
    
def calculateMidpoint(coord_arr):
    dist_arr = []
    midpoint_arr = []
    
    for x in range(0,(len(coord_arr)),1):
        midpoint = (int((coord_arr[x][0][0] + coord_arr[x][3][0])/2),int((coord_arr[x][0][1] + coord_arr[x][3][1])/2))
        midpoint_arr.append(midpoint)
        
    return midpoint_arr

def getRow(bU_arr):
  rows = []
  current_row = []
  y = int(bU_arr[0][7]) + (int(bU_arr[0][9])//2) #Mid-y
  for b_idx, b in enumerate(bU_arr):
    current_y = int(b[7]) + (int(b[9])//2)
    if np.abs(y - current_y) < 30:
      current_row.append(b)
    else:
      rows.append(current_row)
      current_row = []
      current_row.append(b)
      y = int(b[7]) + (int(b[9])//2)
  rows.append(current_row)
  return rows

img_bi = cv2.threshold(img,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
mask = ~img_bi
# cv2.imwrite("step0-img_bi.jpg", img_bi) 

[nrow,ncol] = img_bi.shape

kernel_len = np.array(img).shape[1]//100
hor_SE = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len,1))
vert_SE = cv2.getStructuringElement(cv2.MORPH_RECT, (1,kernel_len))

process_img = cv2.morphologyEx(mask, cv2.MORPH_OPEN, sE_diamond2)

process_img = cv2.morphologyEx(process_img, cv2.MORPH_OPEN, sE_square1, iterations=3)
cv2.imwrite("step1_process_img.jpg", process_img)
step1 = cv2.medianBlur(process_img,13)
cv2.imwrite("step1.jpg", step1) 

# Remove figure
step2 = cv2.bitwise_or(img_bi,step1)
cv2.imwrite("step2.jpg", step2)

step4 = cv2.bitwise_xor(step1,img_bi)
step4 = cv2.erode(~step4, sE_square1, iterations=3)
step4 = cv2.dilate(step4, sE_square3, iterations=4)
step4 = cv2.dilate(step4, sE_square2, iterations=12)

drawings = cv2.bitwise_and(~step4, img) + step4

step5 = cv2.bitwise_and(img_bi, step4)
cv2.imwrite("step5.jpg", step5)

# Get lines from table
process_img = cv2.erode(~step5, hor_SE, iterations=3)
hor_line = cv2.dilate(process_img, hor_SE, iterations=4)
# cv2.imwrite("step5_hor_line.jpg", hor_line)
process_img = cv2.erode(~step5, vert_SE, iterations=3)
vert_line = cv2.dilate(process_img,vert_SE, iterations=4)
# cv2.imwrite("step5_vert_line.jpg", vert_line)
step5_5 = hor_line + vert_line
# cv2.imwrite("step5_5.jpg", step5_5)

step6 = cv2.bitwise_or(step5,step5_5)
cv2.imwrite("step5_5.jpg", step6)
img2 = step6

b_arr = []
coord_arr = []

conf = r'--oem 3 --psm 6'
boxes = tess.image_to_data(img2,config=conf)

for b_idx,b in enumerate(boxes.splitlines()):
    if b_idx!=0:
        b = b.split()
        #b_arr.append(b)
        #print(b)
    if len(b)==12:
        x,y,w,h = int(b[6]),int(b[7]),int(b[8]),int(b[9])
        cv2.rectangle(img2,(x,y),(w+x,h+y),(0,0,255),2)
        if len(b[11]) >= 1 and b[11] != '|':
            b_arr.append(b)
            coord_arr.append(((x,y),(x,(y+w)),((x+h),y),((x+h),(y+w))))
            cv2.putText(img2,str(len(b_arr)),(x,y),cv2.FONT_HERSHEY_COMPLEX,1,(40,40,100),2)
            #cv2.putText(img2,b[11],(x,y),cv2.FONT_HERSHEY_COMPLEX,1,(40,40,100),2)
cv2.imwrite("Experiment2.jpg", img2)      


bU_amend_arr = []
bU_amend_final_arr = []

index_arr = []
index_arr_2 = []
index_arr_2_final = []

temp = []
temp_2 = []

bU_arr = []
bUH_arr = []

matching_arr = []
matching_temp1_arr = []
matching_temp2_arr = []

#Fixing overscanning in a stupid way
for z in range(0, len(b_arr),1):
    for x in ["|PROJECT"]:
        if any(x in s for s in b_arr[z]) == True:
            b_arr[z][11] = "PROJECT"
            #print(bU_arr[i])

for z in range(0, len(b_arr),1):
    for x in ["|COMPANY"]:
        if any(x in s for s in b_arr[z]) == True:
            b_arr[z][11] = "COMPANY"
            #print(bU_arr[i])
            
for z in range(0, len(b_arr),1):
    for x in ["APPROVFD"]:
        if any(x in s for s in b_arr[z]) == True:
            b_arr[z][11] = "APPROVED"

for i in range(0,len(b_arr),1):
    for x in ["DRAWING","DRAWN","CHECKED","APPROVED","PROJECT","CAD","COMPANY"]:
        if any(x in s for s in b_arr[i]) == True:
            for y in ["NUMBER:","BY:","NO:","NAME:","NO.:"]:
                if any(y in s for s in b_arr[i]) == True:
                    b_arr[i][11] = (str(x) + " " + str(y)) 
                    #print(b_arr[i])
                    
            if " " not in b_arr[i][11]:
                if "CAD NO:" not in b_arr[i][11]:
                    #test = calculateDistanceHeading(coord_arr, i)
                    #completeDist_arr.append(test)
                    # print(b_arr[i]) 
                    index_arr.append(i)
                    
                    if b_arr[i] == b_arr[len(b_arr) - 1]:
                        i -= 0
                    #else:
                        #print(b_arr[i+1])
                        
bUH_arr = b_arr                      

for i in range(0, len(index_arr),1):
    if b_arr[int(index_arr[i])][11] in ["DRAWING","DRAWN","CHECKED","APPROVED","PROJECT","CAD","COMPANY"]:
        try:
          if b_arr[int(index_arr[i])+1][11] in ["NUMBER:","DRAWN:","TITLE:","BY:","BY:","NO:","NAME:","NO.:"]:
              if str(b_arr[int(index_arr[i])][11]) + " " + str(b_arr[int(index_arr[i])+1][11]) in ["DRAWING NUMBER:","DRAWING TITLE:","DRAWN BY:","CHECKED BY:","APPROVED BY:","PROJECT NO:","CAD NO:","COMPANY NAME:","DRAWING NO.:","DRAWING NO:"]:
                  bUH_arr[int(index_arr[i])][11] = (str(b_arr[int(index_arr[i])][11]) + " " + str(b_arr[int(index_arr[i])+1][11]))
                  bUH_arr[int(index_arr[i])+1][11] = " "
        except:
          continue

for i in range(0,len(b_arr),1):
    if b_arr[i][11] in ["AMENDMENTS"]:
        # print(i)
        completeDist = calculateDistance(coord_arr,i)        
        midpoint_arr = calculateMidpoint(coord_arr)
        if (midpoint_arr[i][0] < (int((ncol)*0.2))) or (int((ncol)*0.4)) < midpoint_arr[i][0] < (int((ncol)*0.6)) or ((int((ncol)*0.8)) < midpoint_arr[i][0] < (int(ncol)))  or ((int((ncol)*0.6)) < midpoint_arr[i][0] < (int((ncol)*0.8))):
            radius = 450
        elif ((int((ncol)*0.2)) < midpoint_arr[i][0] < (int((ncol)*0.4))):    
            radius = 1000
        
        for k in range(0,len(completeDist),1):
            if (completeDist[k] < radius):
                index_arr_2.append(k)
                # print(completeDist[k])
# print(index_arr_2)

#Managing offset failure
for y in range(len(index_arr_2)):
    if b_arr[int(index_arr_2[y])][11] in ["TITLE:","DRAWING TITLE:"]:
        temp.append(y)
        
if temp:
    for i in range(0,int(temp[0]),1):
        temp_2.append(index_arr_2[i]) 
        
if not temp:
    temp_2 = []        
        
if temp_2:
    for i in range(0, len(temp_2),1):
        index_arr_2_final.append(int(index_arr_2[i]))    
#Remove [11]
    for i in range(0, (len(index_arr_2_final)),1):
        bU_amend_arr.append(b_arr[int(index_arr_2_final[i])])
#Remove [11]
if not temp_2:
    for i in range(0, (len(index_arr_2)),1):
        bU_amend_arr.append(b_arr[int(index_arr_2[i])])
    
if temp_2:
    for idx in range(len(b_arr)):
        if idx not in index_arr_2_final:
            if bUH_arr[idx][11] not in ["AMENDMENTS"," "]:
                if int(b_arr[idx][9]) > 20 and b_arr[idx][11] not in ['=']:
                    bU_arr.append(b_arr[idx])  

if not temp_2:
    for idx in range(len(b_arr)):
        if idx not in index_arr_2:
            if bUH_arr[idx][11] not in ["AMENDMENTS"," "]:
                if int(b_arr[idx][9]) > 20 and b_arr[idx][11] not in ['=']:
                    bU_arr.append(b_arr[idx])            

############################################                    
'''
rows = []
  current_row = []
  y = int(bU_arr[0][7]) + (int(bU_arr[0][9])//2) #Mid-y
  for b_idx, b in enumerate(bU_arr):
    current_y = int(b[7]) + (int(b[9])//2)
    if np.abs(y - current_y) < 30:
      current_row.append(b)
    else:
      rows.append(current_row)
      current_row = []
      current_row.append(b)
      y = int(b[7]) + (int(b[9])//2)
  rows.append(current_row)
  return rows
'''

                  

rows = getRow(bU_arr)
all_data = []


header = None
for r_idx, row in enumerate(rows):
  for box in row:
    if box[11] in headers:
      header = r_idx
      break
  if header != None:
    break
rows = rows[header:]

for row_idx in range(0,len(rows), 2):
  for hb_idx, header_box in enumerate(rows[row_idx]):
    header = header_box[11]
    data = ''
    try:
      for data_box in rows[row_idx+1]:
        data_x = int(data_box[6]) + (int(data_box[8])//2)
        if data_x > int(header_box[6]):
          if hb_idx < len(rows[row_idx])-1: #Check if header not last
            if data_x < int(rows[row_idx][hb_idx+1][6]):
              data += ' ' + data_box[11]
          else: #Last header of row
            data += ' ' + data_box[11]
    except:
      continue
    all_data.append([header, data])
    
############################################  

for x in range(0,len(b_arr),1):
    if b_arr[x][11] in ["AMENDMENTS"]:
        all_data.append(["AMENDMENT","DATA"])

rows = getRow(bU_amend_arr)

#CHECK FOR BOTH AMENDMENT TABLE TYPES

header = None
for r_idx, row in enumerate(rows):
  for box in row:
    if box[11] in headers_amend:
      header = r_idx
      break
  if header != None:
    break
rows = rows[header:]



       
#THIS IS FOR 3x3 AMENDMENT TABLE, 3x5 AMENDMENT TABLE FAILS FROM THIS
for row_idx in range(0,len(rows), 1):
  for hb_idx, header_box in enumerate(rows[row_idx]):     

    data = ''
    try:
      header = rows[0][hb_idx][11]  
      for data_box in rows[row_idx+1]:
        data_x = int(data_box[6]) + (int(data_box[8])//2)
        if data_x > int(header_box[6]):
          if hb_idx < len(rows[row_idx])-1: #Check if header not last
            if data_x < int(rows[row_idx][hb_idx+1][6]):
              data += ' ' + data_box[11]        
          else: #Last header of row
            data += ' ' + data_box[11]
    except:
      continue
    all_data.append([header, data])
 
#print(all_data)

#############################################################
'''
  rows = getRow(bU_amend_arr)
    
    #CHECK FOR BOTH AMENDMENT TABLE TYPES
    
  header = None
  for r_idx, row in enumerate(rows):
    for box in row:
      if box[11] in headers_amend:
        header = r_idx
        break
    if header != None:
      break
  rows = rows[header:]
          
  #THIS IS FOR 3x3 AMENDMENT TABLE, 3x5 AMENDMENT TABLE FAILS FROM THIS
  for row_idx in range(0,len(rows), 1):
    for hb_idx, header_box in enumerate(rows[row_idx]):     
    
      data = ''
    try:
      header = rows[0][hb_idx][11]  
      for data_box in rows[row_idx+1]:
        data_x = int(data_box[6]) + (int(data_box[8])//2)
        if data_x > int(header_box[6]):
          if hb_idx < len(rows[row_idx])-1: #Check if header not last
            if data_x < int(rows[row_idx][hb_idx+1][6]):
              data += ' ' + data_box[11]        
            else: #Last header of row
              data += ' ' + data_box[11]
    except:
      continue
    all_data.append([header, data])
'''
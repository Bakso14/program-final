import cv2
import numpy as np
import time
import os
from math import atan2, cos, sin, pi
from numpy import diff,sign
from scipy.spatial import distance
import math
def nothing(x):
    pass

def getOrientation(pts, img):
    sz = len(pts)
    data_pts = np.empty((sz, 2), dtype=np.float64)
    for i in range(data_pts.shape[0]):
        data_pts[i,0] = pts[i,0,0]
        data_pts[i,1] = pts[i,0,1]
    # Perform PCA analysis
    mean = np.empty((0))
    mean, eigenvectors, eigenvalues = cv2.PCACompute2(data_pts, mean)
    # Store the center of the object
    cntr = (int(mean[0,0]), int(mean[0,1]))
    #cv2.circle(img, cntr, 3, (255, 0, 255), 2)
    angle = atan2(eigenvectors[0,1], eigenvectors[0,0]) # orientation in radians
    length = 100
    x2 = cntr[0] + length*cos(angle)
    y2 = cntr[1] + length*sin(angle)
    #cv2.line(img,cntr,(int(x2),int(y2)),(0,255,0),1,cv2.LINE_AA)
    return angle


cannyH = 'canny high'
cannyL = 'canny low'
co = 'koreksi orientasi'
th_sse = 'threshold SSE'
img = np.zeros((25,512,3), np.uint8)
cv2.namedWindow('thresh')
cv2.createTrackbar(cannyH, 'thresh',15,255,nothing)
cv2.createTrackbar(cannyL, 'thresh',46,255,nothing)

cv2.createTrackbar(co, 'thresh',1,360,nothing)
cv2.createTrackbar(th_sse, 'thresh',100,500,nothing)

kernel = np.ones((15,15),np.uint8)
kernel_er = np.ones((15,15),np.uint8)

final_clstr = []
numero_clstr = []

jumlah_frame = 0


my_file = open("D:\Mata_kuliah_s2\Thesis\Mulai\Video_jalan\Taman_Alumni\list.txt", "r")
content = my_file.read()
content_list = content.split("\n")
my_file.close()

#SAVE
path_parent = "D:\Mata_kuliah_s2\Thesis\Mulai\Program\paka_dataset"
nama_folder = "hasil_data_its_taman_alumni"
path_coba = os.path.join(path_parent,nama_folder)
#os.mkdir(path_coba)

start = time.time()
for foto in range(len(content_list)):
#while(1):
    #foto = 100

    jumlah_frame = jumlah_frame + 1
    
    #cv2.imshow('thresh',img)
    high=cv2.getTrackbarPos(cannyH, 'thresh')
    low=cv2.getTrackbarPos(cannyL, 'thresh')
    
    src = cv2.imread(content_list[foto])

    scale_percent = 50
    width = int(src.shape[1] * scale_percent / 100)
    height = int(src.shape[0] * scale_percent / 100)
    dsize = (width, height)
    
    frame= cv2.resize(src,dsize)
    
    crop = frame[200:360,0:640]
    
    gray = cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(3,3),0) 
    edges = cv2.Canny(blur,high,low,apertureSize = 3,)
    
    # Copy edges to the images that will display the results in BGR
    gambar_kosong = np.zeros(edges.shape[:2], np.uint8)
    cdstP_sementara = cv2.cvtColor(gambar_kosong, cv2.COLOR_GRAY2BGR)
    cdstP = cv2.cvtColor(gambar_kosong, cv2.COLOR_GRAY2BGR)
    contours = []
    linesP = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, None, 100, 10)
    
    v = 0
    if linesP is not None:
        cluster_orientation = np.zeros((2,len(linesP)),np.int16) #connected component orientation
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv2.line(cdstP, (l[0], l[1]), (l[2], l[3]), (255,255,255), 3, cv2.LINE_AA)
            cv2.line(cdstP_sementara, (l[0], l[1]), (l[2], l[3]), (255,255,255), 3, cv2.LINE_AA)
            cdstP_sementara = cv2.cvtColor(cdstP_sementara,cv2.COLOR_BGR2GRAY)
            single_contours = cv2.findContours(cdstP_sementara, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0]
            cdstP_sementara = cv2.cvtColor(gambar_kosong, cv2.COLOR_GRAY2BGR)
            contours.append(single_contours[0])
            a = getOrientation(contours[i], crop)
            a_derajat = 360*a/(2*pi)
            if ((a_derajat < -15 and a_derajat > -30 ) or (a_derajat > 9.5 and a_derajat < 25)):
                cluster_orientation[0,v] = a_derajat
                cluster_orientation[1,v] = i
                v = v+1
                #cv2.drawContours(crop, contours, i, (0, 0, 255), 1) 
    cdstP = cv2.cvtColor(cdstP,cv2.COLOR_BGR2GRAY)
    
    #Pengklasteran berdasarkan orientasi dan SSE
    koreksi_o = cv2.getTrackbarPos(co, 'thresh')
    threshold_SSE = cv2.getTrackbarPos(th_sse, 'thresh')*1000

    R = 10 # maksimal dalam clstr
    N = 5 # maksimal frame cc dalam clstr
    
    #penghapusan clstr tidak memenuhi syarat : component melebihi N frame, clstr tidak berisi
    a = 0
    while a < len(final_clstr):
        b = 1
        while b < len(final_clstr[a]):
            if (jumlah_frame-numero_clstr[a][b]) > N:
                del final_clstr[a][b]
                del numero_clstr[a][b]
                b = b - 1
            b = b + 1
        
        if len(final_clstr[a]) < 2:
            del final_clstr[a]
            del numero_clstr[a]
            a = a - 1
        a = a + 1
            
            
    #print(len(final_clstr))
    if len(final_clstr) < 1 :
        if v!=0:
            final_clstr.append([cluster_orientation[0,0]])
            numero_clstr.append([1])
    tanda = 0
    for connected_component in range(v):
        for a in range(len(final_clstr)):
            if cluster_orientation[0,connected_component] < (final_clstr[a][0] + koreksi_o) and cluster_orientation[0,connected_component] > (final_clstr[a][0] - koreksi_o):
                final_clstr[a].append(contours[cluster_orientation[1,connected_component]])
                numero_clstr[a].append(jumlah_frame)
                if len(final_clstr[a]) > 1 and len(final_clstr[a]) < R:
                    gabung = np.concatenate((final_clstr[a][1:len(final_clstr[a])]), axis=0)
                    z = np.polyfit(gabung[:,0,1],gabung[:,0,0],3 ,full = True)
                    if z[1] < threshold_SSE:
                        tanda = 0
                        final_clstr[a][0] = (final_clstr[a][0] * (len(final_clstr[a])-1)+cluster_orientation[0,connected_component])/len(final_clstr[a])
                        break
                    else:
                        del final_clstr[a][len(final_clstr[a])-1]
                        del numero_clstr[a][len(final_clstr[a])-1]
                        tanda = 1
                else:
                    del final_clstr[a][len(final_clstr[a])-1]
                    del numero_clstr[a][len(final_clstr[a])-1]
                    tanda = 1
            else:
               tanda = 1
        if tanda == 1 :
            final_clstr.append([cluster_orientation[0,connected_component]])
            numero_clstr.append([1])
            final_clstr[len(final_clstr)-1].append(contours[cluster_orientation[1,connected_component]])
            numero_clstr[len(final_clstr)-1].append(jumlah_frame)
            tanda = 0
            
            
    #menghapus clastr menumpuk
    koreksi_sudut = 7
    clstr_a = 0
    while clstr_a < len(final_clstr):
        clstr_b = 0
        while clstr_b < len(final_clstr):
            if clstr_a != clstr_b :
                if final_clstr[clstr_b][0] < (final_clstr[clstr_a][0] + koreksi_sudut) and final_clstr[clstr_b][0] > (final_clstr[clstr_a][0] - koreksi_sudut):
                    final_clstr[clstr_a].extend(final_clstr[clstr_b][1:len(final_clstr[clstr_b])])
                    numero_clstr[clstr_a].extend(numero_clstr[clstr_b][1:len(numero_clstr[clstr_b])])
                    final_clstr[clstr_a][0] = (final_clstr[clstr_a][0] + final_clstr[clstr_b][0])/2 
                    final_clstr[clstr_b][0] = 0
            clstr_b = clstr_b + 1
        clstr_a = clstr_a + 1
                    

    #Menampilkan garis hasil klaster
    ins = 0
    while ins < len(final_clstr):
        if len(final_clstr[ins]) > 1 and final_clstr[ins][0] != 0 :#and ((final_clstr[ins][0] < -15 and final_clstr[ins][0] > -30 ) or (final_clstr[ins][0] > 10 and final_clstr[ins][0] < 25)) :
            gabung = np.concatenate((final_clstr[ins][1:len(final_clstr[ins])]), axis=0)
            z = np.polyfit(gabung[:,0,1],gabung[:,0,0],3 ,full = True)
            p = np.poly1d(z[0])
            x_a = np.arange(min(gabung[:,0,1]),max(gabung[:,0,1]))
            x_a = x_a.reshape((-1, 1))
            y_a = p(x_a)
            y_a = y_a.astype(np.int32)
            y_a = y_a.reshape((-1, 1))
            garis = np.concatenate((y_a,x_a), axis=1)
            garis = garis.reshape((-1, 1, 2))
            color = (255, 0, 0)   
            isClosed = False
            thickness = 2
            
            #local_minmax = diff(sign(diff(y_a[:,0]))).nonzero()[0] + 1 # local min+max
            
            nilai_min = (x_a[np.where(x_a == min(x_a))],y_a[np.where(x_a == min(x_a))])
            nilai_max = (x_a[np.where(x_a == max(x_a))],y_a[np.where(x_a == max(x_a))])

            #print(ins,"Hasil",len(local_minmax))
            #if len(local_minmax) <= 1 or len(local_minmax) > 30:
            jarak_titik = distance.euclidean(nilai_min, nilai_max)
            if jarak_titik > 120 and jarak_titik < 450:
                image = cv2.polylines(crop, [garis], isClosed, color, thickness)
                i_str = str(final_clstr[ins][0])
                font = cv2.FONT_HERSHEY_SIMPLEX 
                org = (y_a[:,0][int((len(y_a)-1)/2)],x_a[:,0][int((len(y_a)-1)/2)])
                fontScale = 0.5
                color = (0, 255, 0) 
                cv2.putText(crop, i_str , org, font, fontScale, color, 1, cv2.LINE_AA)
                #cv2.drawContours(crop, [gabung], 0, (0, 0, 255), 1)   
                    
            else:
                del final_clstr[ins]
                del numero_clstr[ins]
                ins = ins - 1
                
        else:
            del final_clstr[ins]
            del numero_clstr[ins]
            ins = ins - 1
            
        ins = ins+1
    
    edges_3_d = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    #cv2.imshow('edges',edges)
    #cv2.imshow('crop',crop)
    cdstP_3_d = cv2.cvtColor(cdstP, cv2.COLOR_GRAY2BGR)
    numpy_vertical = np.vstack((crop, edges_3_d,cdstP_3_d))
    cv2.imshow('Numpy Vertical', numpy_vertical)

    #save gambar hasil
    os.chdir(path_coba)
    a_str = str(foto)
    cv2.imwrite("hasil"+a_str+".png",numpy_vertical)
    path_ini = r'D:\Mata_kuliah_s2\Thesis\Mulai\Program\paka_dataset\program'
    os.chdir(path_ini)
    
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break


print(time.time()-start)
print(len(content_list)/(time.time()-start))
cv2.destroyAllWindows()

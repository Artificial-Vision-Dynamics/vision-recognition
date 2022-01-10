import numpy as np
import cv2 as cv2
from matplotlib import pyplot as plt
import os
import scipy.stats

# Change directory
os.chdir('D:')
os.chdir(r'D:\OneDrive - Universidad Politécnica de Madrid\Curso 21-22\1er Semestre\6 - Visión por computador\Trabajo\Programming\vision-recognition')

# Variables
matchear = 1
methodName = 'orb'
partesCuerpo = [[800,500],[300,2250],[1500,2250],[500,3650],[1400,3650]]
radio = 300

# Imágenes
folder = 'orb'  # directory = '../' + folder
name = 'persona'
nameImg1 = folder + '/' + name + '5.jpg'
nameImg2 = folder + '/' + name + '4.jpg' # Referencia
img1 = cv2.imread(nameImg1, 0)
img2 = cv2.imread(nameImg2, 0)

# Seleccionar el método y un BruteForce matcher object
if methodName == 'orb':     # ORB
    orb = cv2.ORB_create()
    method = orb
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
elif methodName == 'sift':  # SIFT
    sift = cv2.SIFT_create(15000)  #xfeatures2d.
    method = sift
    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=False)

# Detectar keypoints y descriptors
img1KP, img1Des = method.detectAndCompute(img1, None)
img2KP, img2Des = method.detectAndCompute(img2, None)
#print(len(img2KP))
#print(img2Des.shape)
    
img1DrawKP = cv2.drawKeypoints(img1, img1KP, None, color=(0, 250, 0), flags=0)
img2DrawKP = cv2.drawKeypoints(img2, img2KP, None, color=(0, 250, 0), flags=0)

#nameSave = folder + '/' + name + '_' + methodName + '.jpg'
#cv.imwrite(nameSave,imgKP)

if matchear == 1:
    # Matchear los descriptors de ambas imágenes
    matches = bf.match(img1Des, img2Des)
    # img1KP_pt = np.float32([(img1KP[e].pt) for e in range(0, len(img1KP))])
    # img2KP_pt = np.float32([(img2KP[e].pt) for e in range(0, len(img2KP))])
    # matches = cv2.getAffineTransform(img1KP_pt[0:3], img2KP_pt[0:3])
    #matches = cv2.estimateAffine2D(img1KP, img2KP, True)

    
    # Ordenar los matches por importancia 
    matches = sorted(matches, key=lambda x: x.distance)[0:20]

    # RANSAC eliminar los matches malos y guardar su posición en x, y
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)
    matchesParte = [0] * 3
    matchesFin = [matches[0]]
    # KP1Fin = [img1KP[0]]
    # KP2Fin = [img2KP[0]]
    posParte = np.zeros((5, 2))
    numPartes = np.zeros(5)
    for i,match in enumerate(matches):
        points1[i, :] = img1KP[match.queryIdx].pt
        points2[i, :] = img2KP[match.queryIdx].pt
        # Si está en el circulo de una parte del cuerpo, lo asignamos
        for j,parte in enumerate(partesCuerpo):
            if np.sqrt(np.sum((points2[i]-parte)**2)) < radio:
                # print(points2[i,:])
                aux = [points1[i,0], points1[i,1], j+1]
                matchesParte = np.vstack([matchesParte, aux])
                posParte[j,:] += points1[i,:]
                numPartes[j] += 1
                matchesFin.append(match)
                # KP1Fin.append(img1KP[i])
                # KP2Fin.append(img2KP[i])
        
    # Ordenar la posición de los matches por su parte de cuerpo que corresponden
    matchesOrdenados = sorted(matchesParte, key=lambda x: x[2])
    #print(matchesOrdenados)
    matchesOrdenadosPt = np.float32([(matchesOrdenados[e][0:3]) for e in range(1, len(matchesOrdenados))])  # Quitamos el primer 0,0
    
    posParteFin = np.zeros((5, 2))
    aux1 = 3
    aux2 = 0
    for k, match in enumerate(matchesOrdenadosPt):
        if match[2] != aux1 or k+1 == len(matchesOrdenadosPt):
            print(aux1)
            ppt = matchesOrdenadosPt[aux2:k]
            for n in range(0,2):   
                liminf = scipy.stats.scoreatpercentile(ppt[n], 25)
                limsup = scipy.stats.scoreatpercentile(ppt[n], 75)
                print("El 25% percentil es =", liminf, "y el 75% percentil es =", limsup)
                trimean = scipy.stats.mstats.tmean(ppt[n], (liminf, limsup))
                print(trimean)
                posParteFin[aux1-1][n] = trimean
            aux1 = int(match[2])
            aux2 = k
            
    # Media de los puntos importantes
    
    for k,parte in enumerate(posParte):
        if numPartes[k] == 0:
            posParteFin[k, :] = [a/b*c for a, b, c in zip(partesCuerpo[k],
                                      [img2.shape[0], 1], [img1.shape[0], 1])]
            #posParteFin[k, :] = partesCuerpo[k] / [img2.shape[0],1] * [img1.shape[0],1] #[0, 0]
        else:
            print(posParteFin)
            #posParteFin[k, :] = [parte[0]/numPartes[k], parte[1]/numPartes[k]]
    print(posParteFin)
    
    # Dibujar puntos cuerpos
    fig, ax = plt.subplots(1)
    # Mostrar referencia
    ax.set_aspect('equal')
    ax.imshow(img2DrawKP)
    for xx, yy in partesCuerpo:
        draw_circle = plt.Circle((xx+img1.shape[1], yy), radio, fill=False)
        ax.add_artist(draw_circle)
        ax.plot(xx+img1.shape[1], yy, 'bo')
        if xx != partesCuerpo[0][0]:
            ax.plot([partesCuerpo[0][0] + img1.shape[1], xx + img1.shape[1]],
                    [partesCuerpo[0][0], yy], 'b')
    # Dibujar puntos encontrados
    for xx, yy in posParteFin:
        ax.plot(posParteFin[:, 0], posParteFin[:, 1], 'ro')
        if xx != posParteFin[0][0]:
            ax.plot([posParteFin[0][0], xx],
                    [posParteFin[0][0], yy], 'r')
    

    
    # h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
    # # Usar homography
    # height, width, channels = img2.shape
    # img1Reg = cv2.warpPerspective(img1, h, (width, height))
    # plt.imshow("Reg",img1Reg)
        
        
        
    # Plotear los matches
    # prueba1 = tuple(KP1Fin)
    # prueba2 = tuple(KP2Fin)
    result = cv2.drawMatches(img1, img1KP,
                             img2, img2KP, matchesFin, None)  # img2, flags=2

    # Display the best matching points
    plt.rcParams['figure.figsize'] = [14.0, 7.0]
    plt.title('Best Matching Points')
    plt.imshow(result)
    plt.show()

    # Print total number of matching points between the training and query images
    print("\nNumber of Matching Keypoints Between The Training and Query Images: ", len(matches))

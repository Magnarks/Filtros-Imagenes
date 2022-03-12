import cv2
import numpy as np
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
import imutils
archivo=""
modelo_deteccion_facial= cv2.dnn.readNetFromCaffe("C:/Users/diego/Documents/Python/Graficos/models/deploy.prototxt.txt", "C:/Users/diego/Documents/Python/Graficos/models/res10_300x300_ssd_iter_140000_fp16.caffemodel")

def Caricatura():
    FrameImagenes.pack_forget()
    BotonDescarga.grid_remove()
    FrameImagenes2.pack_forget()
    BotonDescarga2.grid_remove()
    FrameImagenes3.pack_forget()
    BotonDescarga3.grid_remove()
    FrameImagenes4.pack_forget()
    BotonDescarga4.grid_remove()
    #Importamos la imagen
    global imagen
    global archivo
    global imagen2
    global imagen3
    global img_descarga1
    global img_descarga2
    archivo=filedialog.askopenfilename(initialdir="images", filetypes=[("Archivo de Imagen", ".jpg .png")])
    imagen= cv2.imread(archivo)
    #La pasamos a escala de grises
    gris= cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    gris= cv2.medianBlur(gris, 7)
    #creamos edges o capas
    capas= cv2.adaptiveThreshold(gris, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 7, 7)
    #Cartonizacion
    imagen2=colorQuantization(imagen, 9)
    blur= cv2.bilateralFilter(imagen, d=7, sigmaColor=200, sigmaSpace=200)
    imagen3= cv2.bitwise_and(blur, blur, mask=capas)
    img_descarga1= imagen2
    img_descarga2= imagen3
    #mostramos las imagenes
    FrameImagenes2.pack()
    imagen= imutils.resize(imagen, height=1280)
    resimagen = imutils.resize(imagen, width=720)
    resimagen = cv2.cvtColor(resimagen, cv2.COLOR_BGR2RGB)
    tkim = Image.fromarray(resimagen)
    tkimg = ImageTk.PhotoImage(tkim)
    original=Label(FrameImagenes2)
    original.grid(row=0, column=0)
    original.configure(image=tkimg)
    original.image = tkimg
    imagen2= imutils.resize(imagen2, height=1280)
    resimagen2 = imutils.resize(imagen2, width=720)
    resimagen2 = cv2.cvtColor(resimagen2, cv2.COLOR_BGR2RGB)
    tkim2 = Image.fromarray(resimagen2)
    tkimg2 = ImageTk.PhotoImage(tkim2)
    convertida=Label(FrameImagenes2)
    convertida.grid(row=0, column=1)
    convertida.configure(image=tkimg2)
    convertida.image = tkimg2
    imagen3= imutils.resize(imagen3, height=1280)
    resimagen3 = imutils.resize(imagen3, width=720)
    resimagen3 = cv2.cvtColor(resimagen3, cv2.COLOR_BGR2RGB)
    tkim3 = Image.fromarray(resimagen3)
    tkimg3 = ImageTk.PhotoImage(tkim3)
    convertida2=Label(FrameImagenes2)
    convertida2.grid(row=1, column=0, columnspan=2)
    convertida2.configure(image=tkimg3)
    convertida2.image = tkimg3
    BotonDescarga2.grid(row=1, column=0, padx=5, pady=5, columnspan=4)

def colorQuantization(image, k):
    data=np.float32(image).reshape((-1,3))
    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,20,0.001)
    ret,label,center=cv2.kmeans(data,k,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    center=np.uint8(center)
    result=center[label.flatten()]
    result=result.reshape(imagen.shape)
    return result

def nothing(x):
    pass

def Filtro_Lapiz(img, ksize, gamma):
    gris= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur= cv2.GaussianBlur(gris, (ksize,ksize),0)
    img_dividida= cv2.divide(gris, blur, scale=256)
    if gamma == 0:
        gamma = 0.01
    elif gamma < 0:
        raise Exception("No puede ser negativa", "Valor de Gamma no puede ser un numero negativo (rango = 0-1)")
    elif gamma > 1:
        raise Exception("No puede ser superior a 1", "Valor de Gamma no puede ser un numero mayor a 1 (rango = 0-1)")
    invgamma= 1/gamma
    lut= np.array([((i/255)**invgamma)*255 for i in range(0,256)])
    filtro_lapiz= cv2.LUT(img_dividida.astype("uint8"), lut.astype("uint8"))
    
    return filtro_lapiz

def Dibujo_TiempoReal():
    FrameImagenes.pack_forget()
    BotonDescarga.grid_remove()
    FrameImagenes2.pack_forget()
    BotonDescarga2.grid_remove()
    FrameImagenes3.pack_forget()
    BotonDescarga3.grid_remove()
    FrameImagenes4.pack_forget()
    BotonDescarga4.grid_remove()
    global imagen
    global archivo
    global imagen_descarga
    global imagen_dibujada
    aviso.pack()
    archivo=filedialog.askopenfilename(initialdir="images", filetypes=[("Archivo de Imagen", ".jpg .png")])
    imagen= cv2.imread(archivo)
    cv2.namedWindow("Panel de Control")
    cv2.createTrackbar("ksize", "Panel de Control",1,100,nothing)
    cv2.createTrackbar("gamma", "Panel de Control",1,100,nothing)
    while True:
        BotonDescarga.grid(row=1, column=0, padx=5, pady=5, columnspan=4)
        #tomando el ksize
        k= cv2.getTrackbarPos("ksize", "Panel de Control")
        # k size debe ser un numero odd
        k= 2*k+1
        # Calibrar gamma a escala de 0-1
        g= cv2.getTrackbarPos("gamma", "Panel de Control")
        g= g/100
        imagen_r= imagen
        imagen_dibujada= Filtro_Lapiz(imagen,k,g)
        cv2.imshow("Panel de Control", imagen_dibujada)
        imagen_descarga= imagen_dibujada
        #mostrando la imagen final
        FrameImagenes.pack()
        imagen_r= imutils.resize(imagen_r, height=1280)
        resimagen = imutils.resize(imagen_r, width=720)
        resimagen = cv2.cvtColor(resimagen, cv2.COLOR_BGR2RGB)
        tkim = Image.fromarray(resimagen)
        tkimg = ImageTk.PhotoImage(tkim)
        original=Label(FrameImagenes)
        original.grid(row=0, column=0)
        original.configure(image=tkimg)
        original.image = tkimg
        imagen_descarga= imutils.resize(imagen_descarga, height=1280)
        resimagen2 = imutils.resize(imagen_descarga, width=720)
        resimagen2 = cv2.cvtColor(resimagen2, cv2.COLOR_BGR2RGB)
        tkim2 = Image.fromarray(resimagen2)
        tkimg2 = ImageTk.PhotoImage(tkim2)
        convertida=Label(FrameImagenes)
        convertida.grid(row=0, column=1)
        convertida.configure(image=tkimg2)
        convertida.image = tkimg2    
        if cv2.waitKey(1)==27:
            break
        if cv2.getWindowProperty('Panel de Control',cv2.WND_PROP_VISIBLE) < 1:        
            break
    cv2.destroyAllWindows()

def ControlBrillo():
    FrameImagenes.pack_forget()
    BotonDescarga.grid_remove()
    FrameImagenes2.pack_forget()
    BotonDescarga2.grid_remove()
    FrameImagenes3.pack_forget()
    BotonDescarga3.grid_remove()
    FrameImagenes4.pack_forget()
    BotonDescarga4.grid_remove()
    global imagen
    global archivo
    global img_ctrl
    global imagen_descarga
    aviso.pack()
    archivo=filedialog.askopenfilename(initialdir="images", filetypes=[("Archivo de Imagen", ".jpg .png")])
    imagen= cv2.imread(archivo)
    cv2.namedWindow("Control de Brillo")
    Brillo= cv2.createTrackbar("Brillo", "Control de Brillo", 127, 255, nothing)
    valor= np.ones_like(imagen, dtype="uint8")
    while True:
        BotonDescarga3.grid(row=1, column=0, padx=5, pady=5, columnspan=4)
        imagen_r= imagen
        Brillo= cv2.getTrackbarPos("Brillo", "Control de Brillo")
        Barra= Brillo-127
        if Barra>=0:
            valor= np.ones_like(imagen, dtype="uint8")*Barra
            img_ctrl= cv2.add(imagen,valor)
            imagen_descarga= img_ctrl
        else:
            Brillo= 127- Brillo
            valor= np.ones_like(imagen, dtype="uint8")*Brillo
            img_ctrl= cv2.subtract(imagen, valor)
            imagen_descarga= img_ctrl
        cv2.imshow("Control de Brillo", img_ctrl)
        #mostrando la imagen final
        FrameImagenes3.pack()
        imagen_r= imutils.resize(imagen_r, height=1280)
        resimagen = imutils.resize(imagen_r, width=720)
        resimagen = cv2.cvtColor(resimagen, cv2.COLOR_BGR2RGB)
        tkim = Image.fromarray(resimagen)
        tkimg = ImageTk.PhotoImage(tkim)
        original=Label(FrameImagenes3)
        original.grid(row=0, column=0)
        original.configure(image=tkimg)
        original.image = tkimg
        imagen_descarga= imutils.resize(imagen_descarga, height=1280)
        resimagen2 = imutils.resize(imagen_descarga, width=720)
        resimagen2 = cv2.cvtColor(resimagen2, cv2.COLOR_BGR2RGB)
        tkim2 = Image.fromarray(resimagen2)
        tkimg2 = ImageTk.PhotoImage(tkim2)
        convertida=Label(FrameImagenes3)
        convertida.grid(row=0, column=1)
        convertida.configure(image=tkimg2)
        convertida.image = tkimg2 
        if cv2.waitKey(1)==27:
            break
        if cv2.getWindowProperty('Control de Brillo',cv2.WND_PROP_VISIBLE) < 1:        
            break
    cv2.destroyAllWindows()

def ControlBrilloVideos(image, Brillo):
    Barra= Brillo-127

    if Barra>=0:
        valor= np.ones_like(image, dtype="uint8")*Barra
        img_ctrl= cv2.add(image,valor)
    else:
        Brillo= 127- Brillo
        valor= np.ones_like(image, dtype="uint8")*Brillo
        img_ctrl= cv2.subtract(image, valor)
    return img_ctrl

def VideoBrillo():
    FrameImagenes.pack_forget()
    BotonDescarga.grid_remove()
    FrameImagenes2.pack_forget()
    BotonDescarga2.grid_remove()
    FrameImagenes3.pack_forget()
    BotonDescarga3.grid_remove()
    FrameImagenes4.pack_forget()
    BotonDescarga4.grid_remove()
    global imagen
    global archivo
    global img_ctrl
    aviso.pack()
    archivo=filedialog.askopenfilename(initialdir="videos", filetypes=[("Archivo de Video", ".mp4 .avi")])
    imagen= cv2.imread(archivo)
    vid= cv2.VideoCapture(archivo)
    cv2.namedWindow("Control de Brillo")
    Brillo= cv2.createTrackbar("Brillo", "Control de Brillo", 127, 255, nothing)
    valor= np.ones_like(imagen, dtype="uint8")
    #Para Descargar el video
    frame_width = int(vid.get(3))
    frame_height = int(vid.get(4))
    size = (frame_width, frame_height)
    resultado = cv2.VideoWriter('filename.avi', cv2.VideoWriter_fourcc(*'MJPG'), 10, size)

    while True:
        ret, frame= vid.read()
        if ret == False:
            break
            
        Brillo= cv2.getTrackbarPos("Brillo", "Control de Brillo")    
        img_ctrl= ControlBrilloVideos(frame, Brillo)
        cv2.imshow("Control de Brillo", img_ctrl)
        if cv2.waitKey(20) == 27: #El numero indica los fotrogramas
            resultado.write(img_ctrl)
            break
    cv2.destroyAllWindows()
    vid.release()
    resultado.release()

def FusionImagenes():
    FrameImagenes.pack_forget()
    BotonDescarga.grid_remove()
    FrameImagenes2.pack_forget()
    BotonDescarga2.grid_remove()
    FrameImagenes3.pack_forget()
    BotonDescarga3.grid_remove()
    FrameImagenes4.pack_forget()
    BotonDescarga4.grid_remove()
    global imagen
    global imagen2
    global archivo
    global archivo2
    global mezcla
    aviso.pack()
    archivo=filedialog.askopenfilename(initialdir="images", filetypes=[("Archivo de Imagen", ".jpg .png")])
    imagen= cv2.imread(archivo)
    archivo2=filedialog.askopenfilename(initialdir="images", filetypes=[("Archivo de Imagen", ".jpg .png")])
    imagen2= cv2.imread(archivo2)
    cv2.namedWindow("Control de Fusion")
    fusion= cv2.createTrackbar("fusion", "Control de Fusion", 7, 10, nothing)
    fusion2= cv2.createTrackbar("fusion2", "Control de Fusion", 3, 10, nothing)
    while True:
        BotonDescarga4.grid(row=1, column=0, padx=5, pady=5, columnspan=4)
        imagen_r= imagen
        fusion= cv2.getTrackbarPos("fusion", "Control de Fusion")
        res_fusion= fusion/10
        fusion2= cv2.getTrackbarPos("fusion2", "Control de Fusion")
        res_fusion2= fusion2/10
        mezcla= cv2.addWeighted(imagen, res_fusion, imagen2, res_fusion2, 0)
        cv2.imshow("Control de Fusion", mezcla)
        imagen_descarga= mezcla
        #mostrando la imagen final
        FrameImagenes4.pack()
        imagen_r= imutils.resize(imagen_r, height=1280)
        resimagen = imutils.resize(imagen_r, width=720)
        resimagen = cv2.cvtColor(resimagen, cv2.COLOR_BGR2RGB)
        tkim = Image.fromarray(resimagen)
        tkimg = ImageTk.PhotoImage(tkim)
        original=Label(FrameImagenes4)
        original.grid(row=0, column=0)
        original.configure(image=tkimg)
        original.image = tkimg
        imagen_descarga= imutils.resize(imagen_descarga, height=1280)
        resimagen2 = imutils.resize(imagen_descarga, width=720)
        resimagen2 = cv2.cvtColor(resimagen2, cv2.COLOR_BGR2RGB)
        tkim2 = Image.fromarray(resimagen2)
        tkimg2 = ImageTk.PhotoImage(tkim2)
        convertida=Label(FrameImagenes4)
        convertida.grid(row=0, column=1)
        convertida.configure(image=tkimg2)
        convertida.image = tkimg2  
        if cv2.waitKey(1)==27:
            break
        if cv2.getWindowProperty('Control de Fusion',cv2.WND_PROP_VISIBLE) < 1:        
            break
    cv2.destroyAllWindows()

def Face_Blur():
    FrameImagenes.pack_forget()
    BotonDescarga.grid_remove()
    FrameImagenes2.pack_forget()
    BotonDescarga2.grid_remove()
    FrameImagenes3.pack_forget()
    BotonDescarga3.grid_remove()
    FrameImagenes4.pack_forget()
    BotonDescarga4.grid_remove()
    FrameImagenes5.pack_forget()
    global imagen
    global archivo
    archivo=filedialog.askopenfilename(initialdir="images", filetypes=[("Archivo de Imagen", ".jpg .png")])
    imagen= cv2.imread(archivo)
    #blob desde la imagen
    image= imagen.copy()
    #mask image
    face_mask= np.zeros(image.shape[:2],dtype="uint8") #2-D (escala de grises)
    blob= cv2.dnn.blobFromImage(image,1,(300,300),(104,177,123), swapRB=True)
    modelo_deteccion_facial.setInput(blob)
    detecciones= modelo_deteccion_facial.forward()
    h, w= image.shape[:2]
    for i in range(detecciones.shape[2]):
        coincidencias= detecciones[0,0,i,2]
        if coincidencias>0.5:
            box= detecciones[0,0,i,3:7]*np.array([w,h,w,h])
            box= box.astype(int)
            cv2.rectangle(face_mask, (box[0],box[1]), (box[2],box[3]),255,-1)
    inv_face_mask= cv2.bitwise_not(face_mask)
    #Aplicar mask image a imagen original
    bg_image= cv2.bitwise_and(image, image, mask=face_mask)
    fg_image= cv2.bitwise_and(image, image, mask=inv_face_mask)
    #Blur images con gaussian blur
    bg_image_blur= cv2.GaussianBlur(bg_image,(51,51),0)
    fg_image_blur= cv2.GaussianBlur(fg_image,(51,51),0)
    #addicion
    face_blur= cv2.add(fg_image, bg_image_blur)
    fore_blur= cv2.add(bg_image, fg_image_blur)
    #mostrando la imagen final
    FrameImagenes5.pack()
    imagen= imutils.resize(imagen, height=1280)
    resimagen = imutils.resize(imagen, width=720)
    resimagen = cv2.cvtColor(resimagen, cv2.COLOR_BGR2RGB)
    tkim = Image.fromarray(resimagen)
    tkimg = ImageTk.PhotoImage(tkim)
    original=Label(FrameImagenes5)
    original.grid(row=0, column=0)
    original.configure(image=tkimg)
    original.image = tkimg
    face_blur= imutils.resize(face_blur, height=1280)
    resimagen2 = imutils.resize(face_blur, width=720)
    resimagen2 = cv2.cvtColor(resimagen2, cv2.COLOR_BGR2RGB)
    tkim2 = Image.fromarray(resimagen2)
    tkimg2 = ImageTk.PhotoImage(tkim2)
    convertida=Label(FrameImagenes5)
    convertida.grid(row=0, column=1)
    convertida.configure(image=tkimg2)
    convertida.image = tkimg2 

def Fore_Blur():
    FrameImagenes.pack_forget()
    BotonDescarga.grid_remove()
    FrameImagenes2.pack_forget()
    BotonDescarga2.grid_remove()
    FrameImagenes3.pack_forget()
    BotonDescarga3.grid_remove()
    FrameImagenes4.pack_forget()
    BotonDescarga4.grid_remove()
    FrameImagenes5.pack_forget()
    global imagen
    global archivo
    archivo=filedialog.askopenfilename(initialdir="images", filetypes=[("Archivo de Imagen", ".jpg .png")])
    imagen= cv2.imread(archivo)
    #blob desde la imagen
    image= imagen.copy()
    #mask image
    face_mask= np.zeros(image.shape[:2],dtype="uint8") #2-D (escala de grises)
    blob= cv2.dnn.blobFromImage(image,1,(300,300),(104,177,123), swapRB=True)
    modelo_deteccion_facial.setInput(blob)
    detecciones= modelo_deteccion_facial.forward()
    h, w= image.shape[:2]
    for i in range(detecciones.shape[2]):
        coincidencias= detecciones[0,0,i,2]
        if coincidencias>0.5:
            box= detecciones[0,0,i,3:7]*np.array([w,h,w,h])
            box= box.astype(int)
            cv2.rectangle(face_mask, (box[0],box[1]), (box[2],box[3]),255,-1)
    inv_face_mask= cv2.bitwise_not(face_mask)
    #Aplicar mask image a imagen original
    bg_image= cv2.bitwise_and(image, image, mask=face_mask)
    fg_image= cv2.bitwise_and(image, image, mask=inv_face_mask)
    #Blur images con gaussian blur
    bg_image_blur= cv2.GaussianBlur(bg_image,(51,51),0)
    fg_image_blur= cv2.GaussianBlur(fg_image,(51,51),0)
    #addicion
    face_blur= cv2.add(fg_image, bg_image_blur)
    fore_blur= cv2.add(bg_image, fg_image_blur)
    #mostrando la imagen final
    FrameImagenes5.pack()
    imagen= imutils.resize(imagen, height=1280)
    resimagen = imutils.resize(imagen, width=720)
    resimagen = cv2.cvtColor(resimagen, cv2.COLOR_BGR2RGB)
    tkim = Image.fromarray(resimagen)
    tkimg = ImageTk.PhotoImage(tkim)
    original=Label(FrameImagenes5)
    original.grid(row=0, column=0)
    original.configure(image=tkimg)
    original.image = tkimg
    fore_blur= imutils.resize(fore_blur, height=1280)
    resimagen2 = imutils.resize(fore_blur, width=720)
    resimagen2 = cv2.cvtColor(resimagen2, cv2.COLOR_BGR2RGB)
    tkim2 = Image.fromarray(resimagen2)
    tkimg2 = ImageTk.PhotoImage(tkim2)
    convertida=Label(FrameImagenes5)
    convertida.grid(row=0, column=1)
    convertida.configure(image=tkimg2)
    convertida.image = tkimg2 

def Face_Blur2(img):
    #blob desde la imagen
    image= img.copy()
    #mask image
    face_mask= np.zeros(image.shape[:2],dtype="uint8") #2-D (escala de grises)
    blob= cv2.dnn.blobFromImage(image,1,(300,300),(104,177,123), swapRB=True)
    modelo_deteccion_facial.setInput(blob)
    detecciones= modelo_deteccion_facial.forward()
    h, w= image.shape[:2]
    for i in range(detecciones.shape[2]):
        coincidencias= detecciones[0,0,i,2]
        if coincidencias>0.5:
            box= detecciones[0,0,i,3:7]*np.array([w,h,w,h])
            box= box.astype(int)
            cv2.rectangle(face_mask, (box[0],box[1]), (box[2],box[3]),255,-1)
    inv_face_mask= cv2.bitwise_not(face_mask)
    #Aplicar mask image a imagen original
    bg_image= cv2.bitwise_and(image, image, mask=face_mask)
    fg_image= cv2.bitwise_and(image, image, mask=inv_face_mask)
    #Blur images con gaussian blur
    bg_image_blur= cv2.GaussianBlur(bg_image,(51,51),0)
    #addicion
    face_blur= cv2.add(fg_image, bg_image_blur)
    return face_blur

def Fore_Blur2(img):
    #blob desde la imagen
    image= img.copy()
    #mask image
    face_mask= np.zeros(image.shape[:2],dtype="uint8") #2-D (escala de grises)
    blob= cv2.dnn.blobFromImage(image,1,(300,300),(104,177,123), swapRB=True)
    modelo_deteccion_facial.setInput(blob)
    detecciones= modelo_deteccion_facial.forward()
    h, w= image.shape[:2]
    for i in range(detecciones.shape[2]):
        coincidencias= detecciones[0,0,i,2]
        if coincidencias>0.5:
            box= detecciones[0,0,i,3:7]*np.array([w,h,w,h])
            box= box.astype(int)
            cv2.rectangle(face_mask, (box[0],box[1]), (box[2],box[3]),255,-1)
    inv_face_mask= cv2.bitwise_not(face_mask)
    #Aplicar mask image a imagen original
    bg_image= cv2.bitwise_and(image, image, mask=face_mask)
    fg_image= cv2.bitwise_and(image, image, mask=inv_face_mask)
    #Blur images con gaussian blur
    bg_image_blur= cv2.GaussianBlur(bg_image,(51,51),0)
    fg_image_blur= cv2.GaussianBlur(fg_image,(51,51),0)
    #addicion
    fore_blur= cv2.add(bg_image, fg_image_blur)
    
    return fore_blur

def Blur_Tiempo_Real():
    cap= cv2.VideoCapture(1)

    while True:
        ret, frame= cap.read()
        
        if ret == False:
            break
        blur_image= Face_Blur2(frame)

        cv2.imshow("Blur", blur_image)
        
        if cv2.waitKey(1)==27:
            break
    cap.release()
    cv2.destroyAllWindows()

def Blur_Tiempo_Real2():
    cap= cv2.VideoCapture(1)

    while True:
        ret, frame= cap.read()
        
        if ret == False:
            break
        fore_image= Fore_Blur2(frame)
        cv2.imshow("Blur2", fore_image)
        
        if cv2.waitKey(1)==27:
            break
    cap.release()
    cv2.destroyAllWindows()

def Descarga():
    guardado=filedialog.asksaveasfilename(initialdir="images", filetypes=[("Archivo de Imagen", ".jpg .png")], initialfile="nombratuarchivo.jpg")
    cv2.imwrite(guardado, imagen_dibujada)
    BotonDescarga.grid_remove()

def Descarga2():
    guardado=filedialog.asksaveasfilename(initialdir="images", filetypes=[("Archivo de Imagen", ".jpg .png")], initialfile="nombratuarchivo.jpg")
    cv2.imwrite(guardado, img_descarga1)
    guardado2=filedialog.asksaveasfilename(initialdir="images", filetypes=[("Archivo de Imagen", ".jpg .png")], initialfile="nombratuarchivo.jpg")
    cv2.imwrite(guardado2, img_descarga2)
    BotonDescarga2.grid_remove()

def Descarga3():
    guardado=filedialog.asksaveasfilename(initialdir="images", filetypes=[("Archivo de Imagen", ".jpg .png")], initialfile="nombratuarchivo.jpg")
    cv2.imwrite(guardado, img_ctrl)
    BotonDescarga3.grid_remove()
    aviso.pack_forget()

def Descarga4():
    guardado=filedialog.asksaveasfilename(initialdir="images", filetypes=[("Archivo de Imagen", ".jpg .png")], initialfile="nombratuarchivo.jpg")
    cv2.imwrite(guardado, mezcla)
    BotonDescarga4.grid_remove()
    aviso.pack_forget()

main= Tk()
main.config(background="black")
main.state("zoomed")
Label(main, text="Programa de Filtros", font="arial 20 bold", fg="#00FF00", bg="black").pack()
FrameBotones=Frame()
FrameBotones.config(background="black")
FrameBotones.pack()
FrameImagenes=Frame()
FrameImagenes.config(background="black")
FrameImagenes.pack()
FrameImagenes2=Frame()
FrameImagenes2.config(background="black")
FrameImagenes2.pack()
FrameImagenes3=Frame()
FrameImagenes3.config(background="black")
FrameImagenes3.pack()
FrameImagenes4=Frame()
FrameImagenes4.config(background="black")
FrameImagenes4.pack()
FrameImagenes5=Frame()
FrameImagenes5.config(background="black")
FrameImagenes5.pack()
BotonCaricatura=Button(FrameBotones, text="Pintura", command=Caricatura)
BotonCaricatura.grid(row=0, column=0, padx=5, pady=5)
BotonDibujo=Button(FrameBotones, text="Dibujo", command=Dibujo_TiempoReal)
BotonDibujo.grid(row=0, column=1, padx=5, pady=5)
BotonBrillo=Button(FrameBotones, text="Cambiar Brillo", command=ControlBrillo)
BotonBrillo.grid(row=0, column=2, padx=5, pady=5)
BotonBrilloVideo=Button(FrameBotones, text="Cambiar Brillo (Video)", command=VideoBrillo)
BotonBrilloVideo.grid(row=0, column=3, padx=5, pady=5)
BotonFusion=Button(FrameBotones, text="Fusionar Imagenes", command=FusionImagenes)
BotonFusion.grid(row=0, column=4, padx=5, pady=5)
BotonFacialBlur=Button(FrameBotones, text="Blur Caras", command=Face_Blur)
BotonFacialBlur.grid(row=0, column=5, padx=5, pady=5)
BotonForeBlur=Button(FrameBotones, text="Blur Fondo", command=Fore_Blur)
BotonForeBlur.grid(row=0, column=6, padx=5, pady=5)
BotonBlur=Button(FrameBotones, text="Blur Tiempo Real", command=Blur_Tiempo_Real)
BotonBlur.grid(row=0, column=7, padx=5, pady=5)
BotonBlur2=Button(FrameBotones, text="FondoBlur Tiempo Real", command=Blur_Tiempo_Real2)
BotonBlur2.grid(row=0, column=8, padx=5, pady=5)
BotonDescarga=Button(FrameBotones, text="Descargar Imagen", command=Descarga)
BotonDescarga.grid(row=1, column=0, padx=5, pady=5, columnspan=4)
BotonDescarga.grid_remove()
BotonDescarga2=Button(FrameBotones, text="Descargar Imagen", command=Descarga2)
BotonDescarga2.grid(row=1, column=0, padx=5, pady=5, columnspan=4)
BotonDescarga2.grid_remove()
BotonDescarga3=Button(FrameBotones, text="Descargar Imagen", command=Descarga3)
BotonDescarga3.grid(row=1, column=0, padx=5, pady=5, columnspan=4)
BotonDescarga3.grid_remove()
BotonDescarga4=Button(FrameBotones, text="Descargar Imagen", command=Descarga4)
BotonDescarga4.grid(row=1, column=0, padx=5, pady=5, columnspan=4)
BotonDescarga4.grid_remove()
Label(main, text="*La imagen se puede mostrar pequeña en el programa, pero al descargar estara en el formato original*", font="arial 10 bold", fg="#00FF00", bg="black").pack()
aviso=Label(main, text="*Puede Cerrar la ventana pulsando Esc*", font="arial 10 bold", fg="#00FF00", bg="black")
aviso.pack()
aviso.pack_forget()
main.mainloop()
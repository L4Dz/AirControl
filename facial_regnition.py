import cv2

# 1. Carregar o classificador pré-treinado do OpenCV
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# 2. Iniciar a captura de vídeo (0 indica a câmara padrão)
cap = cv2.VideoCapture(0)

while True:
    # 3. Ler cada frame capturado pela câmara
    ret, frame = cap.read()
    if not ret:
        print("Não foi possível aceder à câmara.")
        break
        
    # 4. Converter o frame para escala de cinzentos
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 5. Detetar os rostos no frame
    faces = face_cascade.detectMultiScale(
        gray, 
        scaleFactor=1.1, 
        minNeighbors=5, 
        minSize=(30, 30)
    )
    
    # 6. Desenhar um retângulo azul em volta de cada rosto detetado
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
    # 7. Apresentar o resultado no ecrã
    cv2.imshow('Detetor de Rostos', frame)
    
    # 8. Pressionar a tecla 'q' para encerrar o programa
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libertar os recursos da câmara e fechar as janelas
cap.release()
cv2.destroyAllWindows()
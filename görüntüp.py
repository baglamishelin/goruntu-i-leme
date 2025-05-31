import cv2
import numpy as np

# Orijinal resmi oku ve yeniden boyutlandır
resim = cv2.imread("img.jpg")


# Ortalama filtresi (blur)
ortalamafilter = cv2.blur(resim, (5, 5))

# Ortanca filtresi
ortancafilter = cv2.medianBlur(resim, 3)

# Griye çevir
gray = cv2.cvtColor(resim, cv2.COLOR_BGR2GRAY)

# Sobel filtresi
sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
sobelfilter = cv2.magnitude(sobelx, sobely).astype(np.uint8)

# Prewitt filtresi (OpenCV'de hazır yok, kernel ile yapılır)
prewitt_kernelx = np.array([[ -1, 0, 1],
                            [ -1, 0, 1],
                            [ -1, 0, 1]])
prewitt_kernely = np.array([[ -1, -1, -1],
                            [  0,  0,  0],
                            [  1,  1,  1]])
prewitt_x = cv2.filter2D(gray, -1, prewitt_kernelx)
prewitt_y = cv2.filter2D(gray, -1, prewitt_kernely)
prewittfilter = cv2.magnitude(prewitt_x.astype(np.float32), prewitt_y.astype(np.float32)).astype(np.uint8)

# Gaussian blur
yumusatmafilter = cv2.GaussianBlur(resim, (3, 3), 0)

# Keskinleştirme filtresi
keskin_kernel = np.array([[0, -1, 0],
                          [-1, 5, -1],
                          [0, -1, 0]])
keskinlestirmefilter = cv2.filter2D(resim, -1, keskin_kernel)

# Dönme (270 derece saat yönü tersi)
dondurulmus = cv2.rotate(resim, cv2.ROTATE_90_COUNTERCLOCKWISE)

# Kaydırma
rows, cols = resim.shape[:2]
M = np.float32([[1, 0, 60], [0, 1, 40]])
kaydirilmis = cv2.warpAffine(resim, M, (cols, rows))

# Ayna işlemleri
ayna_y = cv2.flip(resim, 1)
ayna_d = cv2.flip(resim, 0)

# Histogram eşitleme
gray_hist = cv2.cvtColor(resim, cv2.COLOR_BGR2GRAY)
histogramesitlemee = cv2.equalizeHist(gray_hist)

# Kontrast germe
kontrastgermee = cv2.normalize(gray_hist, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

# Manuel eşikleme
_, manuels = cv2.threshold(gray_hist, 100, 255, cv2.THRESH_BINARY)

# Otsu eşikleme
_, otsuesikleme = cv2.threshold(gray_hist, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Kapur entropi için OpenCV fonksiyonu yok, aynı sonucu Otsu ile gösteriyoruz
dummy_kapur = otsuesikleme

# Erosion ve Dilation
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
erosionimg = cv2.erode(otsuesikleme, kernel)
dilationimg = cv2.dilate(otsuesikleme, kernel)

# Ağırlık merkezi
def agirlik_merkezi(img):
    M = cv2.moments(img)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        return (cx, cy)
    return None
agirlik_merkez = agirlik_merkezi(otsuesikleme)
agirlik_goster = cv2.cvtColor(otsuesikleme, cv2.COLOR_GRAY2BGR)
if agirlik_merkez:
    cv2.circle(agirlik_goster, agirlik_merkez, 5, (0, 0, 255), -1)

# İskelet çıkarma
def iskelet_cikarma(img):
    skel = np.zeros(img.shape, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    done = False
    while not done:
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()
        if cv2.countNonZero(img) == 0:
            done = True
    return skel

iskelet = iskelet_cikarma(otsuesikleme)

# Görüntüleri göster
cv2.imshow("Orijinal", resim)
cv2.imshow("Ortalama", ortalamafilter)
cv2.imshow("Ortanca", ortancafilter)
cv2.imshow("Sobel", sobelfilter)
cv2.imshow("Prewitt", prewittfilter)
cv2.imshow("Yumuşatma", yumusatmafilter)
cv2.imshow("Keskinleştirme", keskinlestirmefilter)
cv2.imshow("Dondurma", dondurulmus)
cv2.imshow("Kaydirma", kaydirilmis)
cv2.imshow("Ayna Yatay", ayna_y)
cv2.imshow("Ayna Dikey", ayna_d)
cv2.imshow("Histogram Eşitleme", histogramesitlemee)
cv2.imshow("Kontrast Germe", kontrastgermee)
cv2.imshow("Manuel Eşikleme", manuels)
cv2.imshow("Otsu Eşikleme", otsuesikleme)
cv2.imshow("Kapur (Otsu ile benzer)", dummy_kapur)
cv2.imshow("Erosion", erosionimg)
cv2.imshow("Dilation", dilationimg)
cv2.imshow("Ağırlık Merkezi", agirlik_goster)
cv2.imshow("İskelet", iskelet)

cv2.waitKey(0)
cv2.destroyAllWindows()
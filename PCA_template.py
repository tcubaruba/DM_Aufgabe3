import matplotlib.image as npim
import matplotlib.pyplot as plt
from scipy.linalg import svd
from scipy.linalg import eigh
import numpy as np

print("\n", "*"*15, "AUFGABE 3-3", "*"*15, "\n")


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.144])


def svd_decomposition(img, vals):
    U, s, V = svd(img)
    U = U[:, :vals]
    s = s[:vals]
    V = V[:vals, :]
    reconst_matrix = np.dot(U, np.dot(np.diag(s), V))
    return reconst_matrix


def pca_decomposition(img, vals):
    meanImg = np.mean(img)
    pic = img - meanImg
    # cov = np.cov(pic)
    cov = np.dot(pic, pic.T)
    E, V = eigh(cov)
    E = np.flip(E)
    V = np.fliplr(V)
    E, V = E[:vals], V[:, :vals]
    U = np.dot(V, V.T)
    reconst_matrix = np.dot(U, pic)
    reconst_matrix += meanImg
    return reconst_matrix


img = npim.imread("katze.png")
gray = rgb2gray(img)
print(gray.shape)
Mhat1 = svd_decomposition(gray, 20)
Mhat2 = svd_decomposition(gray, 10)
Mhat3 = svd_decomposition(gray, 5)

fig, [ax1, ax2, ax3, ax4] = plt.subplots(1, 4)
ax1.axis('off')
ax1.imshow(gray, cmap=plt.get_cmap('gray'))
ax1.set_title('true image')
ax2.axis('off')
ax2.imshow(Mhat1, cmap=plt.get_cmap('gray'))
ax2.set_title('20 dims')
ax3.axis('off')
ax3.imshow(Mhat2, cmap=plt.get_cmap('gray'))
ax3.set_title('10 dims')
ax4.axis('off')
ax4.imshow(Mhat3, cmap=plt.get_cmap('gray'))
ax4.set_title('5 dims')
plt.show()

pca1 = pca_decomposition(gray, 20)
pca2 = pca_decomposition(gray, 10)
pca3 = pca_decomposition(gray, 5)

fig, [ax1, ax2, ax3, ax4] = plt.subplots(1, 4)
ax1.axis('off')
ax1.imshow(gray, cmap=plt.get_cmap('gray'))
ax1.set_title('true image')
ax2.axis('off')
ax2.imshow(pca1, cmap=plt.get_cmap('gray'))
ax2.set_title('20 dims')
ax3.axis('off')
ax3.imshow(pca2, cmap=plt.get_cmap('gray'))
ax3.set_title('10 dims')
ax4.axis('off')
ax4.imshow(pca3, cmap=plt.get_cmap('gray'))
ax4.set_title('5 dims')
plt.show()

print("SVD und PCA with 20 components are the same? ", np.allclose(Mhat1, pca1))

print("SVD und PCA with 20 components are QUITE the same? (Tolerance 0.005) ", np.allclose(Mhat1, pca1, atol=0.005))

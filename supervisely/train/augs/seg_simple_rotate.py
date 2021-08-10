import imgaug.augmenters as iaa

seq = iaa.Sequential([
	iaa.Sometimes(0.2, iaa.arithmetic.SaltAndPepper(p=(0, 0.23), per_channel=False)),
	iaa.Sometimes(0.2, iaa.color.MultiplyAndAddToBrightness(mul=(0.7, 1.3), add=(-30, 30), to_colorspace='YCrCb', from_colorspace='RGB', random_order=True)),
	iaa.Sometimes(0.2, iaa.blur.GaussianBlur(sigma=(0, 3))),
	iaa.Sometimes(0.2, iaa.contrast.GammaContrast(gamma=(0.7, 1.7), per_channel=False)),
	iaa.Sometimes(0.2, iaa.arithmetic.JpegCompression(compression=(85, 100))),
	iaa.Sometimes(0.2, iaa.geometric.Rotate(rotate=(-30, 30), order=1, cval=0, mode='constant', fit_output=False))
], random_order=False)

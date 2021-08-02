import imgaug.augmenters as iaa

seq = iaa.Sequential([
	iaa.Sometimes(0.2, iaa.imgcorruptlike.GaussianNoise(severity=(1, 3))),
	iaa.Sometimes(0.1, iaa.imgcorruptlike.MotionBlur(severity=(1, 5))),
	iaa.Sometimes(0.1, iaa.imgcorruptlike.GaussianBlur(severity=(1, 4))),
	iaa.Sometimes(0.05, iaa.imgcorruptlike.Contrast(severity=(1, 2))),
	iaa.Sometimes(0.05, iaa.imgcorruptlike.Brightness(severity=(1, 3))),
	iaa.Sometimes(0.1, iaa.imgcorruptlike.JpegCompression(severity=(1, 3))),
	iaa.Sometimes(0.2, iaa.geometric.Rotate(rotate=(-37, 37), order=1, cval=0, mode='reflect', fit_output=False))
], random_order=True)

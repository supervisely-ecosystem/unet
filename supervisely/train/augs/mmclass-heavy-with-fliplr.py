import imgaug.augmenters as iaa

seq = iaa.Sequential([
	iaa.Sometimes(0.2, iaa.imgcorruptlike.GaussianNoise(severity=(1, 3))),
	iaa.Sometimes(0.1, iaa.imgcorruptlike.MotionBlur(severity=(1, 5))),
	iaa.Sometimes(0.1, iaa.imgcorruptlike.GaussianBlur(severity=(1, 4))),
	iaa.Sometimes(0.05, iaa.imgcorruptlike.Frost(severity=(1, 3))),
	iaa.Sometimes(0.05, iaa.imgcorruptlike.Snow(severity=(1, 2))),
	iaa.Sometimes(0.05, iaa.imgcorruptlike.Fog(severity=(1, 3))),
	iaa.Sometimes(0.05, iaa.imgcorruptlike.Contrast(severity=(1, 2))),
	iaa.Sometimes(0.05, iaa.imgcorruptlike.Brightness(severity=(1, 3))),
	iaa.Sometimes(0.1, iaa.imgcorruptlike.JpegCompression(severity=(1, 3))),
	iaa.Sometimes(0.05, iaa.arithmetic.Cutout(nb_iterations=(1, 5), size=0.2, squared=False, fill_mode='gaussian', cval=128.0, fill_per_channel=True)),
	iaa.Sometimes(0.5, iaa.flip.Fliplr(p=1.0)),
	iaa.Sometimes(0.2, iaa.geometric.Rotate(rotate=(-37, 37), order=1, cval=0, mode='reflect', fit_output=False)),
	iaa.Sometimes(0.1, iaa.geometric.ShearX(shear=(-30, 30), order=1, cval=0, mode='reflect', fit_output=False)),
	iaa.Sometimes(0.1, iaa.geometric.ShearY(shear=(-30, 30), order=1, cval=0, mode='reflect', fit_output=False)),
	iaa.Sometimes(0.05, iaa.geometric.ScaleX(scale=(0.5, 1.5), order=1, cval=0, mode='constant', fit_output=False)),
	iaa.Sometimes(0.05, iaa.geometric.ScaleY(scale=(0.5, 1.5), order=1, cval=0, mode='reflect', fit_output=False))
], random_order=True)

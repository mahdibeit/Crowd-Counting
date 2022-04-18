Full code is taken from:

https://github.com/ZhihengCV/Bayesian-Crowd-Counting

Parts that are explained in the paper are modified accordingly:

Loss function python files are in folder losses:

	Perceptual loss function:

		Autoecoder.ipynb
		Perceptual_loss.py

	Auxilliary loss function with depth:

		Aux_loss_depth.py

	Main bayesian loss function:

		post_prob.py
		bay_loss.py

Model python files:

	VGG simple model

		vgg.py

	VGG with SE attention:

		vgg_att_SE.py

	VGG with depth:

		vgg_att_depth.py

	SASNET and UNET models:

		SASNet_Full.py
		SASNet_Semi.py
		UNet.py
		unet_parts.py

	VGG with dropout:

		vgg_dropout.py

train files with boosting:

	Boosting_regression_trainer.py
	Boosting_train.py

Data related processing:

	Preprocessing step:
		
		patch_preprocessing_step.py
		preprocess_dataset.py

	Patch testing step:

		patch_test.py

	Depth and RGB dataloader:

		crowd_sh_depth.py
		
# Image_overlay

This is a small project . The aim of the task is to perform image overlay on a target group of images .

The steps are :

1. Prepare the dataset : the dataset used in this case is the https://huggingface.co/datasets/Abrumu/Fashion_controlnet_dataset . The dataset is downloaded by using the datasets module and only the target label is used as image.
2. Performing semantic segmentation using deeplabv3 model.The deep labv3 model is a pretrained model and is widely used in semantic segmentation tasks .NOTE : in using the deeplabv3 I am unable to seperate the garments but The segmentation results yield the whole human body .
3. Performing skew correlation analysis on the dataset for which we want to perform overlay . We use the segmented images to get a correlation analysis thereby determing the best fits .
4. Now we use the CAGAN model to perform overlay .
5. We check for the F1 score and the fit of the model

Future updates:

1. making the model more robust by segmenting only the garments.
2. Possibly training the model using deep fashion dataset
3. using editGAN for more features
4. deploying the model
5. *** making an API service for this ***

# Sonic Fanart randomforest prediction

#### Table of contents
- [Heading](#description)

<!-- toc -->

## Description

Sonic the Hedgehog is a very well known video game franchise. It's then of no surprise that many fans have tried drawing their own pieces of art featuring the blue blur.
Deviantart is a popular website to put your own art on the Internet, so I've used a (now deleted) dataset with urls to Deviantart.
This dataset is composed of approximately 1170 fan arts by many Deviantart users.
On Deviantart, when you post art, you have many stats you can see, like the number of views, the number of people that liked or commented, etc.

This application aims to predict how many views a fan art will get, based on the number of likes (called favorites on Deviantart) and comments it got.

## Features:

-parameters selection by the user
-data table visualization
-learning curve and validation curve + accuracy of said model
-prediction in graphical form
-scraping dataset urls
-downloading of fanarts directly from application
-displaying fanarts in app

## How it works:

In the first tab, you can inputs most of the parameters that will be needed for the app to work, namely:
-the proportion of data used to train the model. By default, it takes 70% of our data into account for the training, and keep 30% of it for testing.
-the minimal and maximum number of views. Because we use a randomforest classifier to tell us in which class our fanart will be, all fanarts below the minimal number will be in the class "not a lot of views". All fanarts above will be considered having a "lot of views". The fanarts in-between have a "regular number of views".
-RandomForest Classifiers have many parameters. You can change parameters for the validation curve and see when the parameter is optimal.

In the second tab, you can display the data generated with your inputs.

In the third tab, you can see the results. You can display both the validation curve and learning curve, and see what the accuracy of the model is. The model is then saved automatically.
Then, you can input numbers of favorites and comments a fanart has, and it will predict in which class it may be, based on the model we fitted previously.

Finally, in the fourth tab, you can input integers to download the images in-between those ids. For example to download image 1 to 2, you put 0 and 2 (it will download images with id in-between 0 and 2: image 0 and image 1). It takes ~4 seconds to download an image, because websites can detect the scraping when we're too fast.
Once downloaded, the app will say that fanarts were downloaded. You can either choose to display them, or to download them as a zip file.

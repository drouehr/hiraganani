# hiraganani
----
hiragana ocr dataset and model

a set of 11,384 127x128 grayscale PNG images of the 46 japanese [hiragana](https://en.wikipedia.org/wiki/Hiragana) characters (without [dakuten or handakuten](https://en.wikipedia.org/wiki/Dakuten_and_handakuten)).

<img src="https://i.imgur.com/2wtgknL.png" width=500>

-----

## dataset description

the images are located in `training_images` under the naming scheme `<index>_<sampleNumber>.png`, where `index` is the index of the character (as shown in the table below) and `sampleNumber` is the sample index for that character. for example, `20_9.png` is the 10th sample for the character な (NA).

breakdown of sample numbers for each character in `training_images`:

|index|character|romaji|count|
|-|-|-|-|
|0|あ|A|260|
|1|い|I|208|
|2|う|U|247|
|3|え|E|246|
|4|お|O|271|
|5|か|KA|257|
|6|き|KI|278|
|7|く|KU|241|
|8|け|KE|255|
|9|こ|KO|244|
|10|さ|SA|240|
|11|し|SHI|248|
|12|す|SU|266|
|13|せ|SE|254|
|14|そ|SO|242|
|15|た|TA|214|
|16|ち|CHI|241|
|17|つ|TSU|256|
|18|て|TE|237|
|19|と|TO|206|
|20|な|NA|255|
|21|に|NI|261|
|22|ぬ|NU|257|
|23|ね|NE|239|
|24|の|NO|228|
|25|は|HA|253|
|26|ひ|HI|261|
|27|ふ|FU|250|
|28|へ|HE|239|
|29|ほ|HO|241|
|30|ま|MA|261|
|31|み|MI|232|
|32|む|MU|246|
|33|め|ME|238|
|34|も|MO|227|
|35|や|YA|257|
|36|ゆ|YU|262|
|37|よ|YO|248|
|38|ら|RA|259|
|39|り|RI|252|
|40|る|RU|232|
|41|れ|RE|250|
|42|ろ|RO|243|
|43|わ|WA|275|
|44|を|WO|251|
|45|ん|N|256|

### (low: 206, high: 278, total: 11384)

also included in `test_images` are 400 additional images of random characters that do not overlap with the training set.

----- 

## web interfaces

the samples were collected using the web interface (`/web/draw_hiragana`), which contains a 400x400 canvas where the user is asked to draw a random hiragana character. on submission, the following preprocessing steps are performed:
  - find the bounding box of the drawn character
  - create a temporary canvas with dimensions (127, 128)
  - apply a black background to the temporary canvas
  - scale the character bounding box to a maximum of ((127\*0.9), (128\*0.9)) to leave a 5\% margin on all sides
  - center and overlay the scaled character on the temporary canvas

this ensures that the collected samples are at a consistent scale relative to the canvas, and that any variations in the height and width of each character are the result of variations in writing style. the resulting image is sent as a PNG buffer to the server and saved under an appropriate name.

the set was manually filtered to remove any images where the character was not properly scaled or centered (i.e. when the preprocessing above did not work as expected), artifacts were present or the character was not drawn at least somewhat correctly.

as a side note, i had to enlist the help of a couple of friends (none of whom are japanese) to write up samples with me, so please pardon me if the dataset features some of the worst handwriting you've ever seen.

`/web/predict_hiragana` is a similar web interface that instead polls the canvas every 500ms, performs the preprocessing steps above and sends the image data to a `/predict` endpoint on the server. the server then uses the trained model to classify the image and returns the predicted character to the client.

`/web/server.js` contains a skeleton implementation for an expressjs server used to host the web interfaces, containing the `/predict` endpoint to classify images using a preloaded model.

-----

## model training

a sample keras training script for an ocr model `train_test.py` is included along with a sample model trained using it. also available are python and javascript implementations for testing a model using `/test_images`.

<img src="https://i.imgur.com/3iJS5e5.png" width=800>

the sample model is a CNN with 3 conv2d layers, trained over 24 epochs. 

during training, image augmentation is performed randomly on each sample, including:
  - rotation between ±6.5°
  - ±5% horizontal and vertical shift
  - shear transformations between ±5°
  
this model has an accuracy of 99.25% (397/400) on the test set. the training script also generates feature maps, accuracy/loss plots and a confusion matrix for each model. 

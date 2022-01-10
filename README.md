# The Simpsons face detection and characters recognition

This project uses:
<ul>
  <li>HOG features</li>
  <li>2 neural networks as classifiers, one for detection and one for recognition</li>
  <li>sliding window for detection</li>
</ul>

The data was collected from a dataset containing 4400 images taken from The Simpsons series. The positives examples were the faces of the characters, and the negative examples were extracted randomly at different scales from the images in the dataset.The dataset was augmented with positive examples. Each image of a face was flipped, translated and rotated using methods from openCV. Hence, for every positive example I generated another 53 examples.

Below you can see how the detector performs on the test data. Some of them look good, some not so much. =)))

<div align='center' min-width=820>
  <img src='data/salveazaFisiere/detections/detections_bart_simpson_0.jpg' float='left'>
  <img src='data/salveazaFisiere/detections/detections_homer_simpson_5.jpg' float='right' >
  <img src='data/salveazaFisiere/detections/detections_lisa_simpson_7.jpg' float='right' >
  <img src='data/salveazaFisiere/detections/detections_marge_simpson_6.jpg' float='right' >
  <img src='data/salveazaFisiere/detections/detections_lisa_simpson_5.jpg' float='right' >
  <img src='data/salveazaFisiere/detections/detections_homer_simpson_9.jpg' float='right' >
</div>

<br>Finally, the best classifier got MAP of 0.87 for detections. The scores for recognition were not tha impressive. Bellow you can check out the graphs. 

<div align='center' min-width=820>
  <img src='data/salveazaFisiere/average_precision_all_faces.png' width=300 float='left'>
  <img src='data/salveazaFisiere/average_precision_bart.png' width=300 float='left'>
  <img src='data/salveazaFisiere/average_precision_homer.png' width=300 float='left'>
  <img src='data/salveazaFisiere/average_precision_lisa.png' width=300 float='left'>
  <img src='data/salveazaFisiere/average_precision_marge.png' width=300 float='left'>
</div>

<br>For further details on the solution check out the doc.
<br>The libraries and versions required to run the project:
<ul>
  <li>python==3.9.5</li>
  <li>numpy==1.19.5</li>
  <li>opencv_python==4.5.4.58</li>
  <li>tqdm==4.62.3</li>  
  <li>pickleshare==0.7.5</li>
  <li>scikit-image==0.19.1</li>
  <li>scikit-learn==1.0.1</li>
</ul>

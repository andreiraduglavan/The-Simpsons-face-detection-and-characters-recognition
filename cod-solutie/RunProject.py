from numpy import character
from Parameters import *
from FacialDetector import *
import pdb
from Visualize import *


params: Parameters = Parameters()
params.dim_window = 36  # exemplele pozitive (fete de oameni cropate) au 36x36 pixeli
params.dim_hog_cell = 4
params.overlap = 0.3
params.threshold = 0 # toate ferestrele cu scorul > threshold si maxime locale devin detectii


params.scaling_ratio = 0.9
params.use_flip_images = True  # adauga imaginile cu fete oglindite
train=False
trainFalsePositives=False

if params.use_flip_images:
    params.number_positive_examples *= 2

facial_detector: FacialDetector = FacialDetector(params)
#exemple pozitive


if train:
    positive_features_path = os.path.join(params.dir_save_files, 'descriptoriExemplePozitive_' + str(params.dim_hog_cell) + '_' +
                            str(params.number_positive_examples) + '.npy')
    characters_labels_path = os.path.join(params.dir_save_files, 'labelsExemplePozitive_' + str(params.dim_hog_cell) + '_' +
                            str(params.number_positive_examples) + '.npy')
    if os.path.exists(positive_features_path):
        positive_features = np.load(positive_features_path)
        characters_labels = np.load(characters_labels_path)
        print('Am incarcat descriptorii pentru exemplele pozitive')
    else:
        print('Construim descriptorii pentru exemplele pozitive:')
        positive_features, characters_labels = facial_detector.get_positive_descriptors()
        np.save(characters_labels_path,characters_labels)
        np.save(positive_features_path, positive_features)
        print('Am salvat descriptorii pentru exemplele pozitive in fisierul %s' % positive_features_path)
    params.number_positive_examples=len(positive_features)

    # exemple negative
    negative_features_path = os.path.join(params.dir_save_files, 'descriptoriExempleNegative_' + str(params.dim_hog_cell) + '_' +
                            str(params.number_negative_examples) + '.npy')
    if os.path.exists(negative_features_path):
        negative_features = np.load(negative_features_path)
        np.random.shuffle(negative_features)
        negative_features=negative_features[:141000]
        print('Am incarcat descriptorii pentru exemplele negative')
    else:
        print('Construim descriptorii pentru exemplele negative:')
        negative_features = facial_detector.get_negative_descriptors()
        np.save(negative_features_path, negative_features)
        print('Am salvat descriptorii pentru exemplele negative in fisierul %s' % negative_features_path)
    print(np.shape(positive_features))
    print(np.shape(negative_features))
    #clasificator
    training_examples = np.concatenate((np.squeeze(positive_features), np.squeeze(negative_features)), axis=0)
    train_labels = np.concatenate((np.ones(positive_features.shape[0]), np.zeros(negative_features.shape[0])))
    characters_indexes=[index for index in range(len(characters_labels)) if np.argmax(characters_labels[index])!=4]
    characters_labels=characters_labels[characters_indexes]
    positive_features=positive_features[characters_indexes]
    print(len(positive_features), len(characters_labels))
    facial_detector.train_NN(training_examples, train_labels)
    facial_detector.train_recognition_NN(positive_features, characters_labels)
else:
    facial_detector.train_NN([],[])
    facial_detector.train_recognition_NN([],[])

detections, scores, file_names, descriptors, false_positives = facial_detector.run()

try:
    os.mkdir(os.path.join(params.dir_save_files, 'evaluate/task1/'))
except:
    pass

np.save(os.path.join(params.dir_save_files, 'evaluate/task1/detections_all_faces.npy'), detections)
np.save(os.path.join(params.dir_save_files, 'evaluate/task1/scores_all_faces.npy'), scores)
np.save(os.path.join(params.dir_save_files, 'evaluate/task1/file_names_all_faces.npy'), file_names)


facial_detector.run_recognition(detections, descriptors, file_names)

if os.path.exists(params.path_annotations):
    facial_detector.eval_detections(detections, scores, file_names)
    show_detections_with_ground_truth(detections, scores, file_names, params)
else:
    show_detections_without_ground_truth(detections, scores, file_names, params)
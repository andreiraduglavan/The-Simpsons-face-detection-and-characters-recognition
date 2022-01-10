from Parameters import *
import numpy as np
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import glob
import cv2 as cv
import pdb
import pickle
import ntpath
from copy import deepcopy
import timeit
from skimage.feature import hog
from YWO import isYellow

characters_dict={'bart':0, 'homer':1, 'lisa':2, 'marge':3, 'unknown':4, 'background':5}

def show_image(image):
    cv.imshow("",image)
    cv.waitKey(0)
    cv.destroyAllWindows()

class FacialDetector:
    def __init__(self, params:Parameters):
        self.params = params

    def get_positive_descriptors(self):
        print("AM GENERAT MAI INTAI EXEMPLELE POZITIVE")

        dir__path=self.params.dir_pos_examples
        num_images = len(os.listdir(dir__path))
        positive_descriptors = []
        characters_labels=[]
        print('Calculam descriptorii pt %d imagini pozitive...' % num_images)
        for file in os.listdir(dir__path):
            print('Procesam exemplul pozitiv numarul %s...' % file)
            img = cv.imread(os.path.join(dir__path, file), cv.IMREAD_GRAYSCALE)
            character=file.split('_')[0]
            character_label=np.eye(6)[characters_dict[character]]
            imgFlip=cv.flip(img, flipCode=1)
            height, width = np.shape(img)
            for img in [img, imgFlip]:
                for angle in [-15,0,15]:
                    R=cv.getRotationMatrix2D((width//2,height//2), angle, 1)
                    imageRotated=cv.warpAffine(img,R, (width, height), borderMode=cv.BORDER_REPLICATE)
                    for x in [-width/5,0,width/5]:
                            for y in [-height/5,0, height/5]:
                                T=np.float32([[1,0,x],[0,1,y]])
                                imageTranslated=cv.warpAffine(imageRotated,T, (width, height), borderMode=cv.BORDER_REPLICATE)
                                features=hog(imageTranslated, pixels_per_cell=(self.params.dim_hog_cell,self.params.dim_hog_cell), cells_per_block=(2,2), feature_vector=True)
                                positive_descriptors.append(features)
                                characters_labels.append(character_label)
            # completati codul functiei in continuare
            # TODO: sterge
            '''features = hog(img, pixels_per_cell=(self.params.dim_hog_cell, self.params.dim_hog_cell), cells_per_block=(2, 2), feature_vector=True)
            
            positive_descriptors.append(features)
            characters_labels.append(character_label)
            if self.params.use_flip_images:
                features = hog(np.fliplr(img), pixels_per_cell=(self.params.dim_hog_cell, self.params.dim_hog_cell),
                               cells_per_block=(2, 2), feature_vector=True)
                positive_descriptors.append(features)
                characters_labels.append(character_label)'''

        positive_descriptors = np.array(positive_descriptors)
        return positive_descriptors, characters_labels

    def get_negative_descriptors(self):


        print("AM GENERAT MAI INTAI EXEMPLELE NEGATIVE")

        images_path = os.path.join(self.params.dir_neg_examples, '*.jpg')
        files = glob.glob(images_path)
        num_images = len(files)
        negative_descriptors = []
        print('Calculam descriptorii pt %d imagini negative...' % num_images)
        for i in range(num_images):
            print('Procesam exemplul negativ numarul %d...' % i)
            img = cv.imread(files[i], cv.IMREAD_GRAYSCALE)
            height, width = np.shape(img)
            # completati codul functiei in continuare
            # TODO: sterge
            if height==36 and width==36:
                features = hog(img, pixels_per_cell=(self.params.dim_hog_cell, self.params.dim_hog_cell), cells_per_block=(2, 2), feature_vector=True)
                print(len(features))
                negative_descriptors.append(features)
            '''if self.params.use_flip_images:
                features = hog(np.fliplr(img), pixels_per_cell=(self.params.dim_hog_cell, self.params.dim_hog_cell),
                               cells_per_block=(2, 2), feature_vector=True)
                negative_descriptors.append(features)'''

        negative_descriptors = np.array(negative_descriptors)
        return negative_descriptors

    def train_classifier(self, training_examples, train_labels, ignore_restore=True):
        svm_file_name = os.path.join(self.params.dir_save_files, 'best_model_%d_%d_%d' %
                                     (self.params.dim_hog_cell, self.params.number_negative_examples,
                                      self.params.number_positive_examples))
        if os.path.exists(svm_file_name) and ignore_restore:
            self.params.best_model = pickle.load(open(svm_file_name, 'rb'))
            return

        best_accuracy = 0
        best_c = 0
        best_model = None
        Cs = [10 ** -5, 10 ** -4,  10 ** -3,  10 ** -2, 10 ** -1, 10 ** 0]
        Cs=[1]
        for c in Cs:
            print('Antrenam un clasificator pentru c=%f' % c)
            model = LinearSVC(C=c, verbose=True, max_iter=50000)
            model.fit(training_examples, train_labels)
            acc = model.score(training_examples, train_labels)
            if acc > best_accuracy:
                best_accuracy = acc
                best_c = c
                best_model = deepcopy(model)

        print('Performanta clasificatorului optim pt c = %f' % best_c)
        # salveaza clasificatorul
        pickle.dump(best_model, open(svm_file_name, 'wb'))

        # vizualizeaza cat de bine sunt separate exemplele pozitive de cele negative dupa antrenare
        # ideal ar fi ca exemplele pozitive sa primeasca scoruri > 0, iar exemplele negative sa primeasca scoruri < 0
        scores = best_model.decision_function(training_examples)
        self.params.best_model = best_model
        positive_scores = scores[train_labels > 0]
        negative_scores = scores[train_labels <= 0]


        plt.plot(np.sort(positive_scores))
        plt.plot(np.zeros(len(negative_scores) + 20))
        plt.plot(np.sort(negative_scores))
        plt.xlabel('Nr example antrenare')
        plt.ylabel('Scor clasificator')
        plt.title('Distributia scorurilor clasificatorului pe exemplele de antrenare')
        plt.legend(['Scoruri exemple pozitive', '0', 'Scoruri exemple negative'])
        plt.show()

    def train_perceptron(self, training_examples, train_labels, ignore_restore=True):
        perceptron_file=os.path.join(self.params.dir_save_files, 'model_perceptron')
        if os.path.exists(perceptron_file) and ignore_restore:
            self.params.best_model = pickle.load(open(perceptron_file, 'rb'))
            return
        
        model=Perceptron(random_state=True, verbose=True, tol=1e-4, max_iter=200, n_iter_no_change=1)
        model.fit(training_examples, train_labels)

        pickle.dump(model, open(perceptron_file, 'wb'))

        scores = model.decision_function(training_examples)
        self.params.best_model = model
        positive_scores = scores[train_labels > 0]
        negative_scores = scores[train_labels <= 0]


        plt.plot(np.sort(positive_scores))
        plt.plot(np.zeros(len(negative_scores) + 20))
        plt.plot(np.sort(negative_scores))
        plt.xlabel('Nr example antrenare')
        plt.ylabel('Scor clasificator perceptron')
        plt.title('Distributia scorurilor clasificatorului pe exemplele de antrenare')
        plt.legend(['Scoruri exemple pozitive', '0', 'Scoruri exemple negative'])
        plt.show()

    def train_NN(self, training_examples, train_labels, load_model=True):
        NN_file=os.path.join(self.params.dir_save_files, 'model_NN_2on1')
        if os.path.exists(NN_file) and load_model:
            self.params.best_model = pickle.load(open(NN_file, 'rb'))
            return
        
        model=MLPClassifier(activation='relu', solver='adam', shuffle=True, max_iter=1000, verbose=True, warm_start=True, n_iter_no_change=3)
        model.fit(training_examples, train_labels)
        score=model.score(training_examples, train_labels)
        print(score)
        pickle.dump(model, open(NN_file, 'wb'))

        self.params.best_model = model

    def train_recognition_NN(self, training_examples, train_labels, load_model=True):
        recognition_NN_file=os.path.join(self.params.dir_save_files, 'model_characters_NN')
        if os.path.exists(recognition_NN_file) and load_model:
            self.params.recognition_model = pickle.load(open(recognition_NN_file, 'rb'))
            return
        
        recognition_model=MLPClassifier(activation='relu', solver='adam', shuffle=True, max_iter=50, verbose=True, n_iter_no_change=1)
        recognition_model.fit(training_examples, train_labels)

        pickle.dump(recognition_model, open(recognition_NN_file, 'wb'))

        self.params.recognition_model = recognition_model

    def intersection_over_union(self, bbox_a, bbox_b):
        x_a = max(bbox_a[0], bbox_b[0])
        y_a = max(bbox_a[1], bbox_b[1])
        x_b = min(bbox_a[2], bbox_b[2])
        y_b = min(bbox_a[3], bbox_b[3])

        inter_area = max(0, x_b - x_a + 1) * max(0, y_b - y_a + 1)


        box_a_area = (bbox_a[2] - bbox_a[0] + 1) * (bbox_a[3] - bbox_a[1] + 1)
        box_b_area = (bbox_b[2] - bbox_b[0] + 1) * (bbox_b[3] - bbox_b[1] + 1)

        iou = inter_area / float(box_a_area + box_b_area - inter_area)

        return iou


    def non_maximal_suppression(self, image_detections, image_scores, image_descriptors, image_size, iou_threshold = 0.3):
        """
        Detectiile cu scor mare suprima detectiile ce se suprapun cu acestea dar au scor mai mic.
        Detectiile se pot suprapune partial, dar centrul unei detectii nu poate
        fi in interiorul celeilalte detectii.
        :param image_detections:  numpy array de dimensiune NX4, unde N este numarul de detectii.
        :param image_scores: numpy array de dimensiune N
        :param image_size: tuplu, dimensiunea imaginii
        :return: image_detections si image_scores care sunt maximale.
        """

        # xmin, ymin, xmax, ymax
        x_out_of_bounds = np.where(image_detections[:, 2] > image_size[1])[0]
        y_out_of_bounds = np.where(image_detections[:, 3] > image_size[0])[0]
        print(x_out_of_bounds, y_out_of_bounds)
        image_detections[x_out_of_bounds, 2] = image_size[1]
        image_detections[y_out_of_bounds, 3] = image_size[0]
        sorted_indices = np.flipud(np.argsort(image_scores))
        sorted_image_detections = image_detections[sorted_indices]
        sorted_scores = image_scores[sorted_indices]
        sorted_descriptors = image_descriptors[sorted_indices]

        is_maximal = np.ones(len(image_detections)).astype(bool)

        for i in range(len(sorted_image_detections) - 1):
            if is_maximal[i] == True: # don't change to 'is True' because is a numpy True and is not a python True :)
                for j in range(i + 1, len(sorted_image_detections)):
                    if is_maximal[j] == True: # don't change to 'is True' because is a numpy True and is not a python True :)
                        if self.intersection_over_union(sorted_image_detections[i],
                                                        sorted_image_detections[j]) > iou_threshold:
                            is_maximal[j] = False
                        else:  # verificam daca centrul detectiei este in mijlocul detectiei cu scor mai mare
                            c_x = (sorted_image_detections[j][0] + sorted_image_detections[j][2]) / 2
                            c_y = (sorted_image_detections[j][1] + sorted_image_detections[j][3]) / 2
                            if sorted_image_detections[i][0] <= c_x <= sorted_image_detections[i][2] and \
                                    sorted_image_detections[i][1] <= c_y <= sorted_image_detections[i][3]:
                                is_maximal[j] = False


        return sorted_image_detections[is_maximal], sorted_scores[is_maximal], sorted_descriptors[is_maximal]

    def run(self):
        """
        Aceasta functie returneaza toate detectiile ( = ferestre) pentru toate imaginile din self.params.dir_test_examples
        Directorul cu numele self.params.dir_test_examples contine imagini ce
        pot sau nu contine fete. Aceasta functie ar trebui sa detecteze fete atat pe setul de
        date MIT+CMU dar si pentru alte imagini
        Functia 'non_maximal_suppression' suprimeaza detectii care se suprapun (protocolul de evaluare considera o detectie duplicata ca fiind falsa)
        Suprimarea non-maximelor se realizeaza pe pentru fiecare imagine.
        :return:
        detections: numpy array de dimensiune NX4, unde N este numarul de detectii pentru toate imaginile.
        detections[i, :] = [x_min, y_min, x_max, y_max]
        scores: numpy array de dimensiune N, scorurile pentru toate detectiile pentru toate imaginile.
        file_names: numpy array de dimensiune N, pentru fiecare detectie trebuie sa salvam numele imaginii.
        (doar numele, nu toata calea).
        """

        test_images_path = os.path.join(self.params.dir_test_examples, '*.jpg')
        test_files = glob.glob(test_images_path)
        detections = None  # array cu toate detectiile pe care le obtinem
        scores = np.array([])  # array cu toate scorurile pe care le obtinem
        file_names = np.array(
            [])  # array cu fisiele, in aceasta lista fisierele vor aparea de mai multe ori, pentru fiecare
        # detectie din imagine, numele imaginii va aparea in aceasta lista
        #w = self.best_model.coef_.T
        #bias = self.best_model.intercept_[0]
        num_test_images = len(test_files)
        descriptors_to_return = []
        false_positives=[]
        for i in range(num_test_images):
            start_time = timeit.default_timer()
            print('Procesam imaginea de testare %d/%d..' % (i, num_test_images))
            img = cv.imread(test_files[i])
            imgGray=cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            image_scores = []
            image_detections = []
            image_descriptors = []
            original_image = img.copy()
            scales=[1.5-0.1*x for x in range(11)]
            for scale in scales:
                for scaleHeight in [1,0.8,0.6]:
                    img_scaled=cv.resize(img, (0,0), fx=scale, fy=scale*scaleHeight)
                    img_scaled_gray=cv.resize(imgGray, (0,0), fx=scale, fy=scale*scaleHeight)
                    if np.shape(img_scaled)[0]<37 or np.shape(img_scaled)[1]<37:
                        break
                    # TODO: completati codul functiei in continuare
                    hog_descriptor = hog(img_scaled_gray, pixels_per_cell=(self.params.dim_hog_cell, self.params.dim_hog_cell),
                                        cells_per_block=(2, 2), feature_vector=False)
                    num_cols = img_scaled_gray.shape[1] // self.params.dim_hog_cell - 1
                    num_rows = img_scaled_gray.shape[0] // self.params.dim_hog_cell - 1
                    num_cell_in_template = self.params.dim_window // self.params.dim_hog_cell - 1

                    for y in range(0, num_rows - num_cell_in_template):
                        for x in range(0, num_cols - num_cell_in_template):
                            descr = hog_descriptor[y: y + num_cell_in_template, x: x + num_cell_in_template].flatten()
                            crop= img_scaled[y*self.params.dim_hog_cell: y*self.params.dim_hog_cell + self.params.dim_window, x*self.params.dim_hog_cell: x*self.params.dim_hog_cell + self.params.dim_window]
                            yellowValue=np.sum(isYellow(crop))
                            if yellowValue>648:
                                #score = np.dot(descr, w)[0] + bias
                                logits=self.params.best_model.predict_log_proba([descr])[0]
                                pred=np.argmax(logits)
                                score=np.max(logits)
                                if pred==1:
                                    x_min = int((x * self.params.dim_hog_cell)/scale)
                                    y_min = int((y * self.params.dim_hog_cell)/scale/scaleHeight)
                                    x_max = int((x * self.params.dim_hog_cell + self.params.dim_window)/scale)
                                    y_max = int((y * self.params.dim_hog_cell + self.params.dim_window)/scale/scaleHeight)
                                    bbox=[x_min, y_min, x_max, y_max]
                                    image_detections.append(bbox)
                                    image_scores.append(score)
                                    image_descriptors.append(descr)

            
            if len(image_scores) > 0:
                image_detections, image_scores, image_descriptors = self.non_maximal_suppression(np.array(image_detections),
                                                                              np.array(image_scores),np.array(image_descriptors),
                                                                              original_image.shape, iou_threshold=0)
        
            if len(image_scores) > 0:
                if detections is None:
                    detections = image_detections
                else:
                    detections = np.concatenate((detections, image_detections))
                scores = np.append(scores, image_scores)
                short_file_name = ntpath.basename(test_files[i])
                image_names = [short_file_name for _ in range(len(image_scores))]
                file_names = np.append(file_names, image_names)
                for descriptor in image_descriptors:
                    descriptors_to_return.append(descriptor)
            end_time = timeit.default_timer()
            print('Timpul de procesarea al imaginii de testare %d/%d este %f sec.'
                  % (i, num_test_images, end_time - start_time))

        return detections, scores, file_names, descriptors_to_return, false_positives
    
    def run_recognition(self, detections, descriptors, file_names):
            detections_bart=[]
            scores_bart=[]
            file_names_bart=[]
            detections_homer=[]
            scores_homer=[]
            file_names_homer=[]
            detections_lisa=[]
            scores_lisa=[]
            file_names_lisa=[]
            detections_marge=[]
            scores_marge=[]
            file_names_marge=[]
            
            for i in range(len(descriptors)):
                descr=descriptors[i]
                logits=self.params.recognition_model.predict_log_proba([descr])[0]
                pred=np.argmax(logits)
                score=np.max(logits)
                
                if pred==0:
                    detections_bart.append(detections[i])
                    scores_bart.append(score)
                    file_names_bart.append(file_names[i])
                if  pred==1:
                    detections_homer.append(detections[i])
                    scores_homer.append(score)
                    file_names_homer.append(file_names[i])
                if pred==2:
                    detections_lisa.append(detections[i])
                    scores_lisa.append(score)
                    file_names_lisa.append(file_names[i])
                if pred==3:
                    detections_marge.append(detections[i])
                    scores_marge.append(score)
                    file_names_marge.append(file_names[i])
            try:
                os.mkdir(os.path.join(self.params.dir_save_files, 'evaluate/task2/'))
            except:
                pass
            np.save(os.path.join(self.params.dir_save_files, 'evaluate/task2/detections_bart.npy'), detections_bart)
            np.save(os.path.join(self.params.dir_save_files, 'evaluate/task2/scores_bart.npy'), scores_bart)
            np.save(os.path.join(self.params.dir_save_files, 'evaluate/task2/file_names_bart.npy'), file_names_bart)

            np.save(os.path.join(self.params.dir_save_files, 'evaluate/task2/detections_homer.npy'), detections_homer)
            np.save(os.path.join(self.params.dir_save_files, 'evaluate/task2/scores_homer.npy'), scores_homer)
            np.save(os.path.join(self.params.dir_save_files, 'evaluate/task2/file_names_homer.npy'), file_names_homer)

            np.save(os.path.join(self.params.dir_save_files, 'evaluate/task2/detections_lisa.npy'), detections_lisa)
            np.save(os.path.join(self.params.dir_save_files, 'evaluate/task2/scores_lisa.npy'), scores_lisa)
            np.save(os.path.join(self.params.dir_save_files, 'evaluate/task2/file_names_lisa.npy'), file_names_lisa)

            np.save(os.path.join(self.params.dir_save_files, 'evaluate/task2/detections_marge.npy'), detections_marge)
            np.save(os.path.join(self.params.dir_save_files, 'evaluate/task2/scores_marge.npy'), scores_marge)
            np.save(os.path.join(self.params.dir_save_files, 'evaluate/task2/file_names_marge.npy'), file_names_marge)


    def compute_average_precision(self, rec, prec):
        # functie adaptata din 2010 Pascal VOC development kit
        m_rec = np.concatenate(([0], rec, [1]))
        m_pre = np.concatenate(([0], prec, [0]))
        for i in range(len(m_pre) -  1, -1, 1):
            m_pre[i] = max(m_pre[i], m_pre[i + 1])
        m_rec = np.array(m_rec)
        i = np.where(m_rec[1:] != m_rec[:-1])[0] + 1
        average_precision = np.sum((m_rec[i] - m_rec[i - 1]) * m_pre[i])
        return average_precision


    def eval_detections(self, detections, scores, file_names):
        ground_truth_file = np.loadtxt(self.params.path_annotations, dtype='str')
        ground_truth_file_names = np.array(ground_truth_file[:, 0])
        ground_truth_detections = np.array(ground_truth_file[:, 1:5], np.int)

        num_gt_detections = len(ground_truth_detections)  # numar total de adevarat pozitive
        gt_exists_detection = np.zeros(num_gt_detections)
        # sorteazam detectiile dupa scorul lor
        sorted_indices = np.argsort(scores)[::-1]
        file_names = file_names[sorted_indices]
        scores = scores[sorted_indices]
        detections = detections[sorted_indices]

        num_detections = len(detections)
        true_positive = np.zeros(num_detections)
        false_positive = np.zeros(num_detections)
        duplicated_detections = np.zeros(num_detections)

        for detection_idx in range(num_detections):
            indices_detections_on_image = np.where(ground_truth_file_names == file_names[detection_idx])[0]

            gt_detections_on_image = ground_truth_detections[indices_detections_on_image]
            bbox = detections[detection_idx]
            max_overlap = -1
            index_max_overlap_bbox = -1
            for gt_idx, gt_bbox in enumerate(gt_detections_on_image):
                overlap = self.intersection_over_union(bbox, gt_bbox)
                if overlap > max_overlap:
                    max_overlap = overlap
                    index_max_overlap_bbox = indices_detections_on_image[gt_idx]

            # clasifica o detectie ca fiind adevarat pozitiva / fals pozitiva
            if max_overlap >= 0.3:
                if gt_exists_detection[index_max_overlap_bbox] == 0:
                    true_positive[detection_idx] = 1
                    gt_exists_detection[index_max_overlap_bbox] = 1
                else:
                    false_positive[detection_idx] = 1
                    duplicated_detections[detection_idx] = 1
            else:
                false_positive[detection_idx] = 1

        cum_false_positive = np.cumsum(false_positive)
        cum_true_positive = np.cumsum(true_positive)

        rec = cum_true_positive / num_gt_detections
        prec = cum_true_positive / (cum_true_positive + cum_false_positive)
        average_precision = self.compute_average_precision(rec, prec)
        plt.plot(rec, prec, '-')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Average precision %.3f' % average_precision)
        plt.savefig(os.path.join(self.params.dir_save_files, 'precizie_medie.png'))
        plt.show()

from training_and_evaluation import *


def visualize_predictions(prediction, test_generator):
    rand_samp = random.sample(range(len(prediction)), 10)
    for index in range(6):
        probability = prediction[rand_samp[index]]
        image_path = TEST_DIR + "/" + test_generator.filenames[rand_samp[index]]
        image = mpimg.imread(image_path, 0)
        # BGR TO RGB conversion using CV2
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pixels = np.array(image)
        plt.imshow(pixels)
        print("prob:{}".format(probability))
        print(test_generator.filenames[rand_samp[index]])

        plt.title('%.2f' % ((probability) * 100) + '% COVID')
        plt.show()


def simple_classification_report(y_true, y_pred):
    print(metrics.classification_report(y_true, y_pred))


def matrix_metrix(y_true, y_pred):
    CM = metrics.confusion_matrix(y_true, y_pred)
    TN = CM[0][0]
    FN = CM[1][0]
    TP = CM[1][1]
    FP = CM[0][1]
    Population = TN + FN + TP + FP
    Prevalence = round((TP + FP) / Population, 3)
    Accuracy = round((TP + TN) / Population, 3)
    Precision = round(TP / (TP + FP), 3)
    NPV = round(TN / (TN + FN), 3)
    FDR = round(FP / (TP + FP), 3)
    FOR = round(FN / (TN + FN), 3)
    check_Pos = Precision + FDR
    check_Neg = NPV + FOR
    Recall = round(TP / (TP + FN), 3)
    FPR = round(FP / (TN + FP), 3)
    Specificity = 1 - FPR
    FNR = round(FN / (TP + FN), 3)
    TNR = round(TN / (TN + FP), 3)
    check_Pos2 = Recall + FNR
    check_Neg2 = FPR + TNR
    LRPos = round(Recall / FPR, 3)
    LRNeg = round(FNR / TNR, 3)
    DOR = round(LRPos / LRNeg)
    F1 = round(2 * ((Precision * Recall) / (Precision + Recall)), 4)
    F2 = round((1 + 2 ** 2) * ((Precision * Recall) / ((2 ** 2 * Precision) + Recall)), 4)
    MCC = round(((TP * TN) - (FP * FN)) / math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)), 4)
    BM = Recall + TNR - 1
    MK = Precision + NPV - 1
    AUC = round(roc_auc_score(y_true, prediction), 3)

    met_dict = {
        'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN, 'Prevalence': Prevalence,
        'Accuracy': Accuracy, 'Precision': Precision, 'Recall': Recall,
        'F1': F1, 'F2': F2,
        # 'AUC':AUC,
        'NPV': NPV, 'FPR': FPR,
        'TNR': Specificity, 'FNR': FNR, 'TNR': TNR, 'FDR': FDR, 'FOR': FOR, 'check_Pos': check_Pos,
        'check_Neg': check_Neg, 'check_Pos2': check_Pos2, 'check_Neg2': check_Neg2, 'LR+': LRPos,
        'LR-': LRNeg, 'DOR': DOR, 'MCC': MCC, 'BM': BM, 'MK': MK
    }

    return met_dict


def metrics_in_depth(y_true, y_pred):
    for key, value in matrix_metrix(y_true, y_pred).items():
        print("{:<10} {:<10}".format(key, value))


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), fontsize=20,
                 horizontalalignment="center",
                 color="yellow" if cm[i, j] > thresh else "red")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def get_roc_curve(y_true, y_pred):
    print("AUC: " + str(metrics.roc_auc_score(y_true, y_pred)))
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    # print(fpr)
    # print(tpr)
    pyplot.plot(fpr, tpr, linestyle='--', label='ROC curve')
    pyplot.xlabel('False Positive Rate')
    pyplot.ylabel('True Positive Rate')
    pyplot.legend()
    pyplot.show()


def get_pr_curve(y_true, y_pred):
    print("Average Precision: " + str(sklearn.metrics.average_precision_score(y_true, y_pred)))
    precision, recall, thresholds = sklearn.metrics.precision_recall_curve(y_true, y_pred)
    pyplot.plot(recall, precision, linestyle='--', label='Precision versus Recall')
    pyplot.xlabel('Recall')
    pyplot.ylabel('Precision')
    pyplot.legend()
    pyplot.show()


def metric_against_thresholds(metric, thresholds=[round(i * .01, 2) for i in range(1, 99)], ):
    vals = []
    for t in thresholds:
        y_pred = [1 * (x >= t) for x in prediction]

        vals.append(float(matrix_metrix(y_true, y_pred)[metric]))

    return vals


def plot_metrics_against_thresholds(thresholds=[round(i * .01, 2) for i in range(1, 99)],
                                    desired_metrics=["Accuracy", "Precision", "Recall", "F1", "F2"]):
    pyplot.figure(figsize=(14, 14))
    pyplot.xlabel("Classification Threshold", fontsize=15)
    pyplot.ylabel("Metric Value", fontsize=15)
    for i in range(len(desired_metrics)):
        pyplot.plot(thresholds, metric_against_thresholds(desired_metrics[i]), lw=2, label=desired_metrics[i])
    pyplot.legend(title="Metrics", fontsize='large')
    pyplot.grid()
    pyplot.show()


def get_optimal_cutoffs(y_true, y_pred):
    TN, FP, FN, TP = metrics.confusion_matrix(y_true, y_pred).ravel()
    sensitivity = round(TP / (TP + FN), 2)
    specificity = round(TN / (TN + FP), 2)
    auc = metrics.roc_auc_score(y_true, y_pred)

    youden = sensitivity + specificity - 1
    print("youden: " + str(youden))

    closest_to_01 = np.sqrt(((1 - sensitivity) ** 2) + ((1 - specificity) ** 2))
    print("Closest to (0,1): " + str(np.around(closest_to_01, decimals=3)))

    concordance_probability = sensitivity * specificity
    print("Concordance Probability: " + str((np.around(concordance_probability, decimals=3))))

    index_of_union = np.abs(sensitivity - auc) + np.abs(specificity - auc)
    print("Index of Union: " + str((np.around(index_of_union, decimals=3))))


def full_report(model):
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    test_generator = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=IMAGE_SIZE,
        shuffle=False,
        class_mode='binary'
    )
    y_true = 1 - test_generator.classes
    prediction = (1 - model.predict(test_generator))
    y_pred = [1 * (x[0] >= .65) for x in prediction]

    cm = metrics.confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm=cm, classes=['NON-COVID', 'COVID'], title='Confusion Matrix')
    plot_confusion_matrix(cm=cm, classes=['NON-COVID', 'COVID'], title='Confusion Matrix', normalize=True)

    visualize_predictions(prediction, test_generator)
    simple_classification_report(y_true, y_pred)
    metrics_in_depth(y_true, y_pred)
    plot_metrics_against_thresholds()
    get_optimal_cutoffs(y_true, y_pred)
import itertools
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from keras.callbacks import ModelCheckpoint

def create_row(
            dataset: str,
            model: str,
            stratify: bool,
            hide: bool,
            var_name: str, 
            metric: str, 
            value: float):
    """
    Formats model results as a row of comma separated values.

    Args: 
        dataset (str): dataset name
        model (str): type of neural network used
        stratify (bool): flag to identify split strategy
        hide (bool): flag to identify if a class is missing
        var_name (str): variable name
        metric (str): used metric
        value (float): numerical value
    
    Returns:
        row (str): comma separated values 
    """
    row = [
        dataset,        
        model,        
        str(stratify),  # is_stratified
        str(hide),      # missing_class
        var_name,       # variable
        metric,         # metric
        str(value),     # value
    ]

    return ','.join(row)


def naming_figures(dataset: str, 
                   model: str, 
                   stratify: bool, 
                   hide: bool):
    
    """
    Creates figure names.

    Args: 
        dataset (str): dataset name
        model (str): type of neural network used
        stratify (bool): flag to identify split strategy
        hide (bool): flag to identify if a class is missing
    
    Returns:
        mod_title (str): formated string for model title 
    """

    mod_title = f'{dataset}-{model}'
    if stratify:
        if hide:
            mod_title = mod_title+'_TT'
        else:
            mod_title = mod_title+'_TF'
    else:
        if hide:
            mod_title = mod_title+'_FT'
        else:
            mod_title = mod_title+'_FF'

    return mod_title


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          prefix='',
                          filepath=None,):
    """
    Plot the confusion matrix.
    Normalization is applied by setting `normalize=True`.

    Code soure: 
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    
    """
    plt.figure(figsize = (8,6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.binary)
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

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    if filepath is None:
        plt.savefig(f'{prefix}_{title}.png')
    else:
        plt.savefig(f'{filepath}/{prefix}_{title}.png')


def remove_class(X_train, 
                 y_train, 
                 classes, 
                 hide=False, 
                 stratitify=False, 
                 small_data_size=0.2):
    """
    Splits a dataset into a large portion and a small portion while optionally 
    removing one class. It takes the training data features (X_train), 
    training data labels (y_train), a list of classes (classes), and several 
    optional parameters. It returns the large portion of the training data, 
    the small portion of the training data, and the value of the removed class 
    (if any).
    
    Args:
        X_train (np.array): The training data features.
        y_train (np.array): The training data labels.
        classes (np.array): The list of classes in the dataset.
        hide (bool, optional): Whether to remove a class from the dataset. 
        Defaults to False.
        stratify (bool, optional): Whether to perform stratified splitting. 
        Defaults to False.
        small_data_size (float, optional): The proportion of the dataset to be 
        allocated for the small portion. Defaults to 0.2.
    
    Returns:
        (X_largeT, y_largeT) (tuple): large portion of training data
        (X_smallT, y_smallT) (tuple): small portion of training data
        class_no (int or None): removed class value, or None if no class is removed.
    """
    
    classes = np.array(classes)
    
    if stratitify:
        X_train1, X_train2, y_train1, y_train2 = train_test_split(
            X_train, 
            y_train, 
            test_size=small_data_size, 
            random_state= 44, 
            stratify=y_train)
    else:
        X_train1, X_train2, y_train1, y_train2 = train_test_split(
            X_train, 
            y_train, 
            test_size=small_data_size, 
            random_state= 44)

    

    if hide:
        class_no = np.random.choice(classes, size= 1, replace=False)[0]
        idx = np.where(y_train1 == class_no)[0]

        X_largeT = np.delete(X_train1, idx, axis = 0)
        y_largeT = np.delete(y_train1, idx)
        X_smallT = np.append(X_train2, X_train1[idx], axis = 0)
        y_smallT = np.append(y_train2, y_train1[idx])

        print(f'Deleted class: {class_no}')
    else:
        print('All classes used.')
        X_largeT=X_train1
        X_smallT=X_train2
        y_largeT=y_train1
        y_smallT=y_train2
        class_no = None

    return (X_largeT, y_largeT), (X_smallT, y_smallT), class_no


def generate_new_weights(layer_weights, layer_biases):
    """
    Generates new weights and biases for a neural network. 
    Adds a random number to the inputed weights and biases. 

    Args:
        layer_weights (np.array): Weights of a neural network
        layer_biases (np.array): Biases of a neural network
    
    Returns: 
        new_w (np.array): New modified Weights of a neural network
        new_b (np.array): New modified Biases of a neural network
    """
    random_weight= np.random.uniform(size = len(layer_weights))
    random_bias= np.random.uniform(size = len(layer_biases))

    new_w = [weight*random_weight[i] for i, weight in enumerate(layer_weights)]
    new_b = [bias*random_bias[i] for i, bias in enumerate(layer_biases)]

    return np.array(new_w), np.array(new_b)


def add_noise_to_layer(model):
    """
    Adds noise to weight tensors in a neural network model with multiple layers.
    
    Args:
        model (tf.keras.Model): The neural network model to set new weights in.
                
    Returns:
        None
    """
    # Iterate over each layer in the model and set the new weights
    for layer in model.layers:
        weights= layer.get_weights()
        if len(weights) > 0:
            new_weight, new_bias = generate_new_weights(weights[0], weights[1])
            layer.set_weights([new_weight, new_bias])



def create_clones(model, clones=5):
    """
    Creates clones of a neural network model with multiple layers.

    Args:
        model (tf.keras.Model): The neural network model to set new weights in.

    Returns:
        modified_models (list): A list containing clones of the original model.


    """
    modified_models = []
    
    for i in range(clones):
        modified_model = tf.keras.models.clone_model(model)
        modified_model.set_weights(model.get_weights()) 
        add_noise_to_layer(modified_model)
        modified_models.append(modified_model)

    return modified_models


def clone_a_model(model):
    """
    Creates a clone of a given model by replicating its architecture and 
    copying its weights. It takes a model as input and returns a new model 
    with the same architecture and weights.

    Args:
        model (tf.keras.Model): The model to be cloned.

    Returns:
        new_model (tf.keras.Model): A new model that is a clone of the input model.
    """
    new_model =  tf.keras.models.clone_model(model)
    weights = model.get_weights()
    new_model.set_weights(weights)

    return new_model
    

es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
def fit_and_evaluate_models(new_models_list, X_train, y_train, X_test, y_test, 
                                 loss='categorical_crossentropy',
                                 metrics=['accuracy'], 
                                 optimizer='adam',
                                 batch_size=128,
                                 epochs=100,
                                 verbose=0,
                                 validation_split= 0.1,
                                 callbacks= [es_callback]
                                 ):
    """
    Fits and evaluates multiple machine learning models on a given dataset. 
    It takes a list of models, training and testing data, along with 
    optional parameters, and returns the evaluation scores and training 
    histories for each model.

    Args:
        new_models_list (List): A list of machine learning models to be trained 
        and evaluated.
        X_train (np.array): The training data features.
        y_train (np.array): The training data labels.
        X_test (np.array): The testing data features.
        y_test (np.array): The testing data labels.
        loss (str or callable, optional): The loss function to be optimized 
        during training. Defaults to 'categorical_crossentropy'.
        metrics (list of str or callable, optional): The evaluation metrics to 
        be computed during training and testing. Defaults to ['accuracy'].
        optimizer (str or tf.keras.optimizers.Optimizer instance, optional): 
        The optimizer algorithm to be used during training. Defaults to 'adam'.
        batch_size (int, optional): The number of samples per gradient update. 
        Defaults to 128.
        epochs (int, optional): The number of epochs to train the models. 
        Defaults to 100.
        verbose (int, optional): Verbosity mode (0, 1, or 2). Defaults to 0.
        validation_split (float, optional): Fraction of the training data to be 
        used as validation data. Defaults to 0.1.
        callbacks (list, optional): List of callback functions to be applied 
        during training. Defaults to [es_callback].
        
    Returns:
        scores (list): A list of evaluation scores for each model 
        on the testing data.
        histories (list): A list of training histories for each model.

    """
    scores = []
    histories = []

    for model in new_models_list:
        model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
        history = model.fit(X_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=verbose,
                  validation_split= validation_split,
                  callbacks=callbacks,
                  )
        histories.append(history)
        score = model.evaluate(X_test, y_test)
        scores.append(score)
    
    return scores, histories


def get_weighted_average(modified_models, history):
    """
    Calculates the weighted average for a list of modified models 

    Args: 
        modified_models (List): List of neural network models (tf.keras.Model).
    
    Returns:
        WA_weights (np.array): List of average weights for a new neural net.

    """
    val_acc = [history[i].history['val_accuracy'][-1] for i in range(len(history))]
    
    WA_weights = 0
    
    for i, model in enumerate(modified_models):
        weights= model.get_weights()

        if len(weights) > 0:
            WA_weights += np.array(weights, dtype='object') * val_acc[i]/np.sum(val_acc)
      

    return WA_weights


def evolutionary_strategy(model, clones, X_train, y_train, X_test, y_test):
    """
    Applies evolutionary strategy to a machine learning model and creates
    a average model based on its offspring.
    
    Args:
        model (tf.keras.Model):
        clones (int): number of desired mutations 
        X_train (np.array): The training data features.
        y_train (np.array): The training data labels.
        X_test (np.array): The testing data features.
        y_test (np.array): The testing data labels.
    
    Returns:
        avg_model (tf.keras.Model):
        history (list): The training history of the evolutionary strategy.
    """
    new_models = create_clones(model=model, clones=clones)
    _, history = fit_and_evaluate_models(new_models, 
                                         X_train, 
                                         y_train, 
                                         X_test, 
                                         y_test)
    WA_weights = get_weighted_average(new_models, history)

    avg_model = tf.keras.models.clone_model(model)
    avg_model.set_weights(WA_weights)

    return avg_model, history


def get_confusion_matrix(model, 
                         X_test, 
                         y_test, 
                         classes, 
                         title = "Confusion Matrix",
                         prefix = 'TT',
                         plot = True,
                         filepath = None,
                         ):
    """
    Calculates the confusion matrix for a given model's predictions 
    on a test dataset. It takes the model, test data features (`X_test`), 
    test data labels (`y_test`), a list of classes (`classes`), and several 
    optional parameters. It returns a dictionary containing the confusion 
    matrix and the model's predictions.

    Args:
        model (tf.keras.Model): The model used for making predictions.
        X_test (array-like): The test data features.
        y_test (array-like): The test data labels.
        classes (array-like): The list of classes in the dataset.
        title (str, optional): The title of the confusion matrix plot. 
        Defaults to "Confusion Matrix".
        prefix (str, optional): A prefix to be added to the plot file name 
        if saved. Defaults to "TT".
        plot (bool, optional): Whether to plot the confusion matrix. 
        Defaults to True.
        filepath (str, optional): The file path to save the plot if plot is 
        True. Defaults to None.
        
    Returns:
        results (dict): A dictionary containing the confusion matrix 
        and the model's predictions.
    """
    model_preds = model.predict(X_test)
    model_pred = [np.argmax(model_preds[i]) for i in range(y_test.shape[0])]
    model_true = np.array([np.argmax(y_test[i]) for i in range(y_test.shape[0])])
    cm_model = confusion_matrix(model_true, model_pred)

    if plot:
        plot_confusion_matrix(cm_model, 
                              classes = classes, 
                              title = title, 
                              prefix=prefix,
                              filepath=filepath)

    return {'confusion matrix': cm_model,
            'predictions': model_pred}


def evaluate_evolutionary_strategy(model, 
                                   clones, 
                                   X_train, 
                                   y_train, 
                                   X_test, 
                                   y_test,
                                   ):
    """
    Evaluates evolutionary strategy applied to a given machine learning model. 
    It takes the model, a number of clones, training and testing data, 
    and returns the results in a dictionary format.

    Args:
        model (tf.keras.Model): The original machine learning model to be evaluated.
        clones (int): The number of desired mutations for the evolutionary strategy.
        X_train (np.array): The training data features.
        y_train (np.array): The training data labels.
        X_test (np.array): The testing data features.
        y_test (np.array): The testing data labels.
    
    Returns:
        results (dict): A dictionary containing the results of the evolutionary 
        strategy evaluation.
    """
    
    results = {}

    for clone in clones:
        M, _ = evolutionary_strategy(model, 
                                     clone, 
                                     X_train, 
                                     y_train, 
                                     X_test, 
                                     y_test)
        M.compile(loss='categorical_crossentropy', 
                  optimizer='adam', 
                  metrics=['accuracy'])
        loss, accuracy = M.evaluate(X_test, y_test)
        results.update({clone:
                       {'model':M,
                        'loss':loss, 
                        'accuracy':accuracy
                        }
                    })
    
    return results